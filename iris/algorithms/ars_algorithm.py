# Copyright 2024 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Algorithm class for Augmented Random Search Blackbox algorithm."""

import collections
import datetime
import math
import pathlib
import pickle as pkl
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

from absl import logging
from flax import linen as nn
from iris import checkpoint_util
from iris import normalizer
from iris import worker_util
from iris.algorithms import algorithm
from iris.algorithms import stateless_perturbation_generators
import jax
import jax.numpy as jnp
import numpy as np

PRNGKey = jax.Array

_DUMMY_REWARD = -1_000_000_000.0


class MLP(nn.Module):
  """Defines an MLP model for learned hyper-params."""

  hidden_sizes: Sequence[int] = (32, 16)
  output_size: int = 2

  @nn.compact
  def __call__(self, x: jnp.ndarray, state: Any):
    for feat in self.hidden_sizes:
      x = nn.Dense(feat)(x)
      x = nn.tanh(x)
    x = nn.Dense(self.output_size)(x)
    return nn.sigmoid(x), state

  def initialize_carry(self, rng: PRNGKey, params: jnp.ndarray) -> Any:
    del rng, params
    return None


class AugmentedRandomSearch(algorithm.BlackboxAlgorithm):
  """Augmented random search algorithm for blackbox optimization."""

  def __init__(self,
               std: float | Callable[[int], float],
               step_size: float | Callable[[int], float],
               top_percentage: float = 1.0,
               orthogonal_suggestions: bool = False,
               quasirandom_suggestions: bool = False,
               top_sort_type: str = "max",
               obs_norm_data_buffer: Optional[normalizer.MeanStdBuffer] = None,
               **kwargs) -> None:
    """Initializes the augmented random search algorithm.

    Args:
      std: Standard deviation for normal perturbations around current
        optimization parameter vector. A std schedule as a function of iteration
        number can also be given.
      step_size: Step size for gradient ascent. A step size schedule as a
        function of iteration number can also be given.
      top_percentage: Fraction of top performing perturbations to use for
        gradient estimation.
      orthogonal_suggestions: Whether to orthogonalize the perturbations.
      quasirandom_suggestions: Whether quasirandom perturbations should be used;
        valid only if orthogonal_suggestions = True.
      top_sort_type: How to sort evaluation results for selecting top
        directions. Valid options are: "max" and "diff".
      obs_norm_data_buffer: Buffer to sync statistics from all workers for
        online mean std observation normalizer.
      **kwargs: Other keyword arguments for base class.
    """
    super().__init__(**kwargs)
    self._iteration = 0
    self._std = std
    self._step_size = step_size
    self._last_std_used = 1.0
    self._num_top = int(top_percentage * self._num_suggestions)
    self._num_top = max(1, self._num_top)
    self._orthogonal_suggestions = orthogonal_suggestions
    self._quasirandom_suggestions = quasirandom_suggestions
    self._top_sort_type = top_sort_type
    self._obs_norm_data_buffer = obs_norm_data_buffer

  def initialize(self, state: Dict[str, Any]) -> None:
    """Initializes the algorithm from initial worker state."""
    self._opt_params = state["init_params"]

    # Initialize Observation normalization buffer with init data from the worker
    if self._obs_norm_data_buffer is not None:
      self._obs_norm_data_buffer.data = state["obs_norm_buffer_data"]

  def process_evaluations(
      self, eval_results: Sequence[worker_util.EvaluationResult]) -> None:
    """Processes the list of Blackbox function evaluations return from workers.

    Gradient is computed by taking a weighted sum of directions and
    difference of their value from the current value. The current parameter
    vector is then updated in the gradient direction with specified step size.

    Args:
      eval_results: List containing Blackbox function evaluations based on the
        order in which the suggestions were sent. ARS performs antithetic
        gradient estimation. The suggestions are sent for evaluation in pairs.
        The eval_results list should contain an even number of entries with the
        first half entries corresponding to evaluation result of positive
        perturbations and the last half corresponding to negative perturbations.
    """

    # Retrieve delta direction from the param suggestion sent for evaluation.
    pos_eval_results = eval_results[:self._num_suggestions]
    neg_eval_results = eval_results[self._num_suggestions:]
    filtered_pos_eval_results = []
    filtered_neg_eval_results = []
    for (peval, neval) in zip(pos_eval_results, neg_eval_results):
      if (peval.params_evaluated.size) and (
          neval.params_evaluated.size):
        filtered_pos_eval_results.append(peval)
        filtered_neg_eval_results.append(neval)
    params = np.array([r.params_evaluated for r in filtered_pos_eval_results])
    directions = (params - self._opt_params) / self._last_std_used

    eval_results = filtered_pos_eval_results + filtered_neg_eval_results

    # Get top evaluation results
    pos_evals = np.array([r.value for r in filtered_pos_eval_results])
    neg_evals = np.array([r.value for r in filtered_neg_eval_results])
    if self._top_sort_type == "max":
      max_evals = np.max(np.vstack([pos_evals, neg_evals]), axis=0)
    else:
      max_evals = np.abs(pos_evals - neg_evals)
    idx = (-max_evals).argsort()[:self._num_top]
    pos_evals = pos_evals[idx]
    neg_evals = neg_evals[idx]
    all_top_evals = np.hstack([pos_evals, neg_evals])
    evals = pos_evals - neg_evals

    # Get delta directions corresponding to top evals
    directions = directions[idx, :]

    # Estimate gradients
    gradient = np.dot(evals, directions) / evals.shape[0]
    if not np.isclose(np.std(all_top_evals), 0.0):
      gradient /= np.std(all_top_evals)

    # Apply gradients
    step_size = self._step_size
    if callable(self._step_size):
      step_size = self._step_size(self._iteration)
    self._iteration += 1
    self._opt_params += step_size * gradient

    if self._obs_norm_data_buffer is not None:
      for r in eval_results:
        self._obs_norm_data_buffer.merge(r.obs_norm_buffer_data)

  def get_param_suggestions(self,
                            evaluate: bool = False) -> Sequence[Dict[str, Any]]:
    """Suggests a list of inputs to evaluate the Blackbox function on.

    Suggestions are sampled from a gaussian distribution around the current
    parameter vector. For each suggestion, a dict containing keyword arguments
    for the worker is sent.

    Args:
      evaluate: Whether to evaluate current optimization variables for reporting
        training progress.

    Returns:
      A list of suggested inputs for the workers to evaluate.
    """
    if evaluate:
      param_suggestions = [self._opt_params] * self._num_evals
    else:
      dimensions = self._opt_params.shape[0]
      if self._orthogonal_suggestions:
        if self._quasirandom_suggestions:
          param_suggestions = (
              stateless_perturbation_generators.RandomHadamardMatrixGenerator(
                  self._num_suggestions, dimensions
              ).generate_matrix()
          )
        else:
          # We generate random iid perturbations and orthogonalize them. In the
          # case when the number of suggestions to be generated is greater than
          # param dimensionality, we generate multiple orthogonal perturbation
          # blocks. Rows are othogonal within a block but not across blocks.
          ortho_pert_blocks = []
          for _ in range(math.ceil(float(self._num_suggestions / dimensions))):
            perturbations = self._np_random_state.normal(
                0, 1, (self._num_suggestions, dimensions))
            ortho_matrix, _ = np.linalg.qr(perturbations.T)
            ortho_pert_blocks.append(np.sqrt(dimensions) * ortho_matrix.T)
          param_suggestions = np.vstack(ortho_pert_blocks)
          param_suggestions = param_suggestions[:self._num_suggestions, :]
      else:
        param_suggestions = self._np_random_state.normal(
            0, 1, (self._num_suggestions, dimensions))
      self._last_std_used = self._std
      if callable(self._std):
        self._last_std_used = self._std(self._iteration)
      param_suggestions = np.vstack([
          self._opt_params + self._last_std_used * param_suggestions,
          self._opt_params - self._last_std_used * param_suggestions
      ])

    suggestions = []
    for params in param_suggestions:
      suggestion = {"params_to_eval": params}
      if self._obs_norm_data_buffer is not None:
        suggestion["obs_norm_state"] = self._obs_norm_data_buffer.state
        suggestion["update_obs_norm_buffer"] = not evaluate
      suggestions.append(suggestion)
    return suggestions

  @property
  def state(self) -> Dict[str, Any]:
    return self._get_state()

  def _get_state(self) -> Dict[str, Any]:
    state = {"params_to_eval": self._opt_params}
    if self._obs_norm_data_buffer is not None:
      state["obs_norm_state"] = self._obs_norm_data_buffer.state
    return state

  @state.setter
  def state(self, new_state: Dict[str, Any]) -> None:
    self._set_state(new_state)

  def _set_state(self, new_state: Dict[str, Any]) -> None:
    self._opt_params = new_state["params_to_eval"]
    if self._obs_norm_data_buffer is not None:
      self._obs_norm_data_buffer.state = new_state["obs_norm_state"]

  def restore_state_from_checkpoint(self, new_state: Dict[str, Any]) -> None:
    self.state = new_state


class LearnableAugmentedRandomSearch(AugmentedRandomSearch):
  """Learnable augmented random search algorithm for blackbox optimization."""

  def __init__(
      self,
      model: Callable[[], nn.Module] = MLP,
      model_path: Optional[str] = None,
      top_percentage: float = 1.0,
      orthogonal_suggestions: bool = False,
      quasirandom_suggestions: bool = False,
      top_sort_type: str = "max",
      obs_norm_data_buffer: Optional[normalizer.MeanStdBuffer] = None,
      seed: int = 42,
      reward_buffer_size: int = 10,
      **kwargs,
  ) -> None:
    """Initializes the learnable augmented random search algorithm.

    Args:
      model: The model class to use when loading the meta-policy.
      model_path: The checkpoint path to load the meta-policy from.
      top_percentage: Fraction of top performing perturbations to use for
        gradient estimation.
      orthogonal_suggestions: Whether to orthogonalize the perturbations.
      quasirandom_suggestions: Whether quasirandom perturbations should be used;
        valid only if orthogonal_suggestions = True.
      top_sort_type: How to sort evaluation results for selecting top
        directions. Valid options are: "max" and "diff".
      obs_norm_data_buffer: Buffer to sync statistics from all workers for
        online mean std observation normalizer.
      seed: The seed to use.
      reward_buffer_size: the size of the reward buffer that stores a history of
        rewards.
      **kwargs: Other keyword arguments for base class.
    """
    super().__init__(**kwargs)
    super().__init__(**kwargs)
    self._iteration = 0
    self._seed = seed
    self._model_path = model_path
    self._model = model()
    self._last_std_used = 1.0
    self._num_top = int(top_percentage * self._num_suggestions)
    self._num_top = max(1, self._num_top)
    self._orthogonal_suggestions = orthogonal_suggestions
    self._quasirandom_suggestions = quasirandom_suggestions
    self._top_sort_type = top_sort_type
    self._obs_norm_data_buffer = obs_norm_data_buffer
    self._tree_weights = None
    self._reward_buffer_size = reward_buffer_size
    self._reward_buffer = collections.deque(maxlen=self._reward_buffer_size)
    self._populate_reward_buffer()
    self._step_size = 0.02
    self._std = 1.0

  def _populate_reward_buffer(self):
    """Populate reward buffer with very negative values."""
    self._reward_buffer.extend([_DUMMY_REWARD] * self._reward_buffer_size)

  def _restore_state_from_checkpoint(self, logdir: str):
    try:
      state = checkpoint_util.load_checkpoint_state(logdir)
      iteration = 0  # No iteration information is extracted
      return state, iteration
    except ValueError:
      logging.warning(
          "Failed to load directly as a checkpoint, try searching subfolders"
          " with checkpoints."
      )
    return None, 0

  def get_param_suggestions(
      self, evaluate: bool = False
  ) -> Sequence[Dict[str, Any]]:
    """Suggests a list of inputs to evaluate the Blackbox function on.

    Suggestions are sampled from a gaussian distribution around the current
    parameter vector. For each suggestion, a dict containing keyword arguments
    for the worker is sent.

    Args:
      evaluate: Whether to evaluate current optimization variables for reporting
        training progress.

    Returns:
      A list of suggested inputs for the workers to evaluate.
    """
    if evaluate:
      param_suggestions = [self._opt_params] * self._num_evals
    else:
      dimensions = self._opt_params.shape[0]
      if self._orthogonal_suggestions:
        if self._quasirandom_suggestions:
          param_suggestions = (
              stateless_perturbation_generators.RandomHadamardMatrixGenerator(
                  self._num_suggestions, dimensions
              ).generate_matrix()
          )
        else:
          # We generate random iid perturbations and orthogonalize them. In the
          # case when the number of suggestions to be generated is greater than
          # param dimensionality, we generate multiple orthogonal perturbation
          # blocks. Rows are othogonal within a block but not across blocks.
          ortho_pert_blocks = []
          for _ in range(math.ceil(float(self._num_suggestions / dimensions))):
            perturbations = self._np_random_state.normal(
                0, 1, (self._num_suggestions, dimensions)
            )
            ortho_matrix, _ = np.linalg.qr(perturbations.T)
            ortho_pert_blocks.append(np.sqrt(dimensions) * ortho_matrix.T)
          param_suggestions = np.vstack(ortho_pert_blocks)
          param_suggestions = param_suggestions[: self._num_suggestions, :]
      else:
        param_suggestions = self._np_random_state.normal(
            0, 1, (self._num_suggestions, dimensions)
        )
      self._last_std_used = self._std
      param_suggestions = np.vstack([
          self._opt_params,
          self._opt_params + self._last_std_used * param_suggestions,
          self._opt_params - self._last_std_used * param_suggestions,
      ])

    suggestions = []
    for params in param_suggestions:
      suggestion = {"params_to_eval": params}
      if self._obs_norm_data_buffer is not None:
        suggestion["obs_norm_state"] = self._obs_norm_data_buffer.state
        suggestion["update_obs_norm_buffer"] = not evaluate
      suggestions.append(suggestion)
    return suggestions

  def process_evaluations(
      self, eval_results: Sequence[worker_util.EvaluationResult]
  ) -> None:

    self._reward_buffer.append(eval_results[0].value)
    rewards = np.asarray(self._reward_buffer)
    model_input = np.concatenate([[self._iteration], rewards])

    if self._tree_weights is None:
      self._state = self._restore_state_from_checkpoint(self._model_path)
      self._tree_weights = self._model.init(
          jax.random.PRNGKey(seed=self._seed), model_input, self._state
      )

    hyper_params, self._state = self._model.apply(
        self._tree_weights, model_input, self._state
    )
    step_size, std = hyper_params
    self._step_size = step_size
    self._std = std
    super().process_evaluations(eval_results)


class MultiAgentAugmentedRandomSearch(AugmentedRandomSearch):
  """Augmented random search algorithm for blackbox optimization."""

  def __init__(self,
               std: float,
               step_size: float,
               top_percentage: float = 1.0,
               orthogonal_suggestions: bool = False,
               quasirandom_suggestions: bool = False,
               top_sort_type: str = "max",
               obs_norm_data_buffer: Optional[normalizer.MeanStdBuffer] = None,
               agent_keys: Optional[List[str]] = None,
               restore_state_from_single_agent: bool = False,
               **kwargs) -> None:
    """Initializes the augmented random search algorithm for multi-agent training.

    Args:
      std: Standard deviation for normal perturbations around current
        optimization parameter vector.
      step_size: Step size for gradient ascent.
      top_percentage: Fraction of top performing perturbations to use for
        gradient estimation.
      orthogonal_suggestions: Whether to orthogonalize the perturbations.
      quasirandom_suggestions: Whether quasirandom perturbations should be used;
        valid only if orthogonal_suggestions = True.
      top_sort_type: How to sort evaluation results for selecting top
        directions. Valid options are: "max" and "diff".
      obs_norm_data_buffer: Buffer to sync statistics from all workers for
        online mean std observation normalizer.
      agent_keys: List of keys which uniquely identify the agents. The ordering
        needs to be consistent across the algorithm, policy, and worker.
      restore_state_from_single_agent: if True then when
        restore_state_from_checkpoint is called the state is duplicated
        self._num_agents times.
      **kwargs: Other keyword arguments for base class.
    """
    super().__init__(std=std,
                     step_size=step_size,
                     top_percentage=top_percentage,
                     orthogonal_suggestions=orthogonal_suggestions,
                     quasirandom_suggestions=quasirandom_suggestions,
                     top_sort_type=top_sort_type,
                     obs_norm_data_buffer=obs_norm_data_buffer,
                     **kwargs)
    if agent_keys is None:
      self._agent_keys = ["arm", "opp"]
    else:
      self._agent_keys = agent_keys
    self._num_agents = len(self._agent_keys)
    self._restore_state_from_single_agent = restore_state_from_single_agent

  def _split_params(self, params: np.ndarray) -> List[np.ndarray]:
    return np.array_split(params, self._num_agents)

  def _combine_params(self, params_per_agents: List[np.ndarray]) -> np.ndarray:
    return np.concatenate(params_per_agents, axis=0)

  def restore_state_from_checkpoint(self, new_state: Dict[str, Any]) -> None:
    logging.info("Restore: restore from 1 agent: %d",
                 self._restore_state_from_single_agent)
    logging.info("Restore: num_agents: %d", self._num_agents)
    logging.info("Restore: new state keys: %s", list(new_state.keys()))
    logging.info("Restore: new_state params shape: %s",
                 new_state["params_to_eval"].shape)

    # Initialize multiple agents from a single agent.
    if self._restore_state_from_single_agent:
      if new_state["params_to_eval"].ndim != 1:
        raise ValueError(
            f"Params to eval has {new_state['params_to_eval'].ndim} dims, "
            "should only have 1."
        )
      duplicated_state = {
          "params_to_eval":
              np.tile(new_state["params_to_eval"], self._num_agents)
      }
      if self._obs_norm_data_buffer is not None:
        duplicated_state["obs_norm_state"] = {}
        duplicated_state["obs_norm_state"]["mean"] = np.tile(
            new_state["obs_norm_state"]["mean"], self._num_agents)
        duplicated_state["obs_norm_state"]["std"] = np.tile(
            new_state["obs_norm_state"]["std"], self._num_agents)
        duplicated_state["obs_norm_state"]["n"] = (
            new_state["obs_norm_state"]["n"])

      self.state = duplicated_state
      logging.info("Restore: duplicated states params shape: %s",
                   duplicated_state["params_to_eval"].shape)

    # Initialize one agent from a single agent.
    else:
      self.state = new_state

    logging.info("Restored state: params shape: %s, opt params shape: %s, "
                 "obs norm state: %s",
                 self.state["params_to_eval"].shape,
                 self._opt_params.shape,
                 self.state.get("obs_norm_state", None))
    if self._obs_norm_data_buffer is not None:
      logging.info("Restored state: obs norm mean shape: %s, std shape: %s",
                   self.state["obs_norm_state"]["mean"].shape,
                   self.state["obs_norm_state"]["std"].shape)

  def maybe_save_custom_checkpoint(self,
                                   state: Dict[str, Any],
                                   checkpoint_path: Union[pathlib.Path, str]
                                  ) -> None:
    """Saves a checkpoint per agent with prefix checkpoint_path."""
    agent_params = self._split_params(state["params_to_eval"])
    for i in range(self._num_agents):
      per_agent_state = {}
      per_agent_state["params_to_eval"] = agent_params[i]
      if self._obs_norm_data_buffer is not None:
        obs_norm_state = state["obs_norm_state"]
        elems_per_agent = int(
            obs_norm_state["mean"].shape[-1] / self._num_agents)
        per_agent_state["obs_norm_state"] = {}
        start_idx = i * elems_per_agent
        end_idx = (i + 1) * elems_per_agent
        if obs_norm_state["mean"].ndim == 1:
          per_agent_state["obs_norm_state"]["mean"] = (
              obs_norm_state["mean"][start_idx: end_idx])
          per_agent_state["obs_norm_state"]["std"] = (
              obs_norm_state["std"][start_idx: end_idx])
        else:
          per_agent_state["obs_norm_state"]["mean"] = (
              obs_norm_state["mean"][:, start_idx: end_idx])
          per_agent_state["obs_norm_state"]["std"] = (
              obs_norm_state["std"][:, start_idx: end_idx])
        per_agent_state["obs_norm_state"]["n"] = obs_norm_state["n"]
      agent_checkpoint_path = f"{checkpoint_path}_agent_{i}"
      logging.info("Saving agent checkpoints to %s...", agent_checkpoint_path)
      self.save_checkpoint_internal(
          agent_checkpoint_path, per_agent_state
      self.save_checkpoint_oss(agent_checkpoint_path, per_agent_state)

  def save_checkpoint_oss(self, checkpoint_path: str, state: Any) -> None:
    with open(checkpoint_path, "wb") as f:
      pkl.dump(state, f)

  def split_and_save_checkpoint(self, checkpoint_path: str) -> None:
    state = checkpoint_util.load_checkpoint_state(checkpoint_path)
    self.maybe_save_custom_checkpoint(state=state,
                                      checkpoint_path=checkpoint_path)

  def _get_top_evaluation_results(
      self,
      agent_key: str,
      pos_eval_results: Sequence[worker_util.EvaluationResult],
      neg_eval_results: Sequence[worker_util.EvaluationResult]
      ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    pos_evals = np.array(
        [r.metrics[f"reward_{agent_key}"] for r in pos_eval_results])
    neg_evals = np.array(
        [r.metrics[f"reward_{agent_key}"] for r in neg_eval_results])
    if self._top_sort_type == "max":
      max_evals = np.max(np.vstack([pos_evals, neg_evals]), axis=0)
    elif self._top_sort_type == "diff":
      max_evals = np.abs(pos_evals - neg_evals)
    idx = (-max_evals).argsort()[:self._num_top]
    pos_evals = pos_evals[idx]
    neg_evals = neg_evals[idx]
    return pos_evals, neg_evals, idx

  def process_evaluations(
      self, eval_results: Sequence[worker_util.EvaluationResult]) -> None:
    """Processes the list of Blackbox function evaluations return from workers.

    Gradient is computed by taking a weighted sum of directions and
    difference of their value from the current value. The current parameter
    vector is then updated in the gradient direction with specified step size.

    Args:
      eval_results: List containing Blackbox function evaluations based on the
        order in which the suggestions were sent. ARS performs antithetic
        gradient estimation. The suggestions are sent for evaluation in pairs.
        The eval_results list should contain an even number of entries with the
        first half entries corresponding to evaluation result of positive
        perturbations and the last half corresponding to negative perturbations.
    """

    # Retrieve delta direction from the param suggestion sent for evaluation.
    pos_eval_results = eval_results[:self._num_suggestions]
    neg_eval_results = eval_results[self._num_suggestions:]
    filtered_pos_eval_results = []
    filtered_neg_eval_results = []
    for i in range(len(pos_eval_results)):
      if (pos_eval_results[i].params_evaluated.size) and (
          neg_eval_results[i].params_evaluated.size):
        filtered_pos_eval_results.append(pos_eval_results[i])
        filtered_neg_eval_results.append(neg_eval_results[i])

    params = np.array([r.params_evaluated for r in filtered_pos_eval_results])
    eval_results = filtered_pos_eval_results + filtered_neg_eval_results

    # This is length num pos results with splits per agent
    eval_params_per_agent = [self._split_params(p) for p in params]
    eval_params_per_agent = list(zip(*eval_params_per_agent))
    # This has length num agents with a 2d array with shape
    # (num_pos_results, agent_params_dim).
    eval_params_per_agent = [np.array(a) for a in eval_params_per_agent]

    current_params_per_agent = self._split_params(self._opt_params)
    updated_params_per_agent = []
    for (agent_eval_params, agent_params, agent_key) in zip(
        eval_params_per_agent, current_params_per_agent, self._agent_keys):
      pos_evals, neg_evals, idx = self._get_top_evaluation_results(
          agent_key=agent_key,
          pos_eval_results=filtered_pos_eval_results,
          neg_eval_results=filtered_neg_eval_results)
      all_top_evals = np.hstack([pos_evals, neg_evals])
      evals = pos_evals - neg_evals

      # Get delta directions corresponding to top evals
      directions = (agent_eval_params - agent_params) / self._std
      directions = directions[idx, :]

      # Estimate gradients
      gradient = np.dot(evals, directions) / evals.shape[0]
      if not np.isclose(np.std(all_top_evals), 0.0):
        gradient /= np.std(all_top_evals)

      # Apply gradients
      updated_agent_params = agent_params + self._step_size * gradient
      updated_params_per_agent.append(updated_agent_params)

    self._opt_params = self._combine_params(updated_params_per_agent)

    # Update the observation buffer
    if self._obs_norm_data_buffer is not None:
      for r in eval_results:
        self._obs_norm_data_buffer.merge(r.obs_norm_buffer_data)
