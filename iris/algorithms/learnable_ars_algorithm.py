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

"""Algorithm class for Learnable ARS."""

import collections
import math
from typing import Any, Callable, Dict, Optional, Sequence

from absl import logging
from flax import linen as nn
from iris import checkpoint_util
from iris import normalizer
from iris import worker_util
from iris.algorithms import ars_algorithm
from iris.algorithms import stateless_perturbation_generators
import jax
import jax.numpy as jnp
import numpy as np


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

  def initialize_carry(self, rng: jax.Array, params: jnp.ndarray) -> Any:
    del rng, params
    return None


class LearnableAugmentedRandomSearch(ars_algorithm.AugmentedRandomSearch):
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
    self._model_state = None
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
      self._model_state = self._restore_state_from_checkpoint(self._model_path)
      self._tree_weights = self._model.init(
          jax.random.PRNGKey(seed=self._seed), model_input, self._model_state
      )

    hyper_params, self._state = self._model.apply(
        self._tree_weights, model_input, self._model_state
    )
    step_size, std = hyper_params
    self._step_size = step_size
    self._std = std
    super().process_evaluations(eval_results)
