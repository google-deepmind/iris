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

from typing import Any, Dict, Optional, Sequence

from iris import normalizer
from iris.algorithms import algorithm
from iris.algorithms import stateless_perturbation_generators
from iris.workers import worker_util
import numpy as np


class PersistentES(algorithm.BlackboxAlgorithm):
  """Augmented random search algorithm for blackbox optimization."""

  def __init__(self,
               std: float,
               step_size: float,
               top_percentage: float = 1.0,
               orthogonal_suggestions: bool = False,
               quasirandom_suggestions: bool = False,
               top_sort_type: str = "max",
               obs_norm_data_buffer: Optional[normalizer.MeanStdBuffer] = None,
               partial_rollout_length: Optional[int] = 5,
               **kwargs) -> None:
    """Initializes the augmented random search algorithm.

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
      partial_rollout_length: Partial environment rollout length.
      **kwargs: Other keyword arguments for base class.
    """
    super().__init__(**kwargs)
    self._std = std
    self._step_size = step_size
    self._num_top = int(top_percentage * self._num_suggestions)
    self._num_top = max(1, self._num_top)
    self._orthogonal_suggestions = orthogonal_suggestions
    self._quasirandom_suggestions = quasirandom_suggestions
    self._top_sort_type = top_sort_type
    self._obs_norm_data_buffer = obs_norm_data_buffer
    self._partial_rollout_length = partial_rollout_length

  def initialize(self, state: Dict[str, Any]) -> None:
    """Initializes the algorithm from initial worker state."""
    self._opt_params = state["init_params"]
    self._positive_cumulative_perturbations = [0] * self._num_suggestions
    self._negative_cumulative_perturbations = [0] * self._num_suggestions

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
    pos_directions = []
    neg_directions = []
    for i in range(len(pos_eval_results)):
      if (pos_eval_results[i].params_evaluated.size) and (
          neg_eval_results[i].params_evaluated.size):
        filtered_pos_eval_results.append(pos_eval_results[i])
        filtered_neg_eval_results.append(neg_eval_results[i])

        params = pos_eval_results[i].params_evaluated
        pos_directions.append((params - self._opt_params) / self._std)
        pos_directions[-1] = self._positive_cumulative_perturbations[
            i] + pos_directions[-1]
        if pos_eval_results[i].metrics["current_step"] == 0:
          self._positive_cumulative_perturbations[i] = 0
        else:
          self._positive_cumulative_perturbations[i] = pos_directions[-1]

        params = neg_eval_results[i].params_evaluated
        neg_directions.append((params - self._opt_params) / self._std)
        neg_directions[-1] = self._negative_cumulative_perturbations[
            i] + neg_directions[-1]
        if neg_eval_results[i].metrics["current_step"] == 0:
          self._negative_cumulative_perturbations[i] = 0
        else:
          self._negative_cumulative_perturbations[i] = neg_directions[-1]

    pos_directions = np.array(pos_directions)
    neg_directions = np.array(neg_directions)
    eval_results = filtered_pos_eval_results + filtered_neg_eval_results

    # Get top evaluation results
    pos_evals = np.array([r.value for r in filtered_pos_eval_results])
    neg_evals = np.array([r.value for r in filtered_neg_eval_results])
    if self._top_sort_type == "max":
      max_evals = np.max(np.vstack([pos_evals, neg_evals]), axis=0)
    elif self._top_sort_type == "diff":
      max_evals = np.abs(pos_evals - neg_evals)
    idx = (-max_evals).argsort()[:self._num_top]
    pos_evals = pos_evals[idx]
    neg_evals = neg_evals[idx]
    all_top_evals = np.hstack([pos_evals, neg_evals])

    # Get delta directions corresponding to top evals
    pos_directions = pos_directions[idx, :]
    neg_directions = neg_directions[idx, :]

    # Estimate gradients
    gradient = (np.dot(pos_evals, pos_directions) +
                np.dot(neg_evals, neg_directions)) / pos_evals.shape[0]
    if not np.isclose(np.std(all_top_evals), 0.0):
      gradient /= np.std(all_top_evals)

    # Apply gradients
    self._opt_params += self._step_size * gradient

    # Update the observation buffer
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
      param_suggestions = self._np_random_state.normal(
          0, 1, (self._num_suggestions, dimensions))
      if self._orthogonal_suggestions:
        if self._quasirandom_suggestions:
          param_suggestions = stateless_perturbation_generators.RandomHadamardMatrixGenerator(
              self._num_suggestions, dimensions).generate_matrix()
        else:
          ortho_matrix, _ = np.linalg.qr(param_suggestions.T)
          param_suggestions = np.sqrt(dimensions) * ortho_matrix.T
      param_suggestions = np.vstack([
          self._opt_params + self._std * param_suggestions,
          self._opt_params - self._std * param_suggestions
      ])

    suggestions = []
    for params in param_suggestions:
      suggestion = {"params_to_eval": params}
      if evaluate:
        suggestion["partial_rollout_length"] = None
      else:
        suggestion["partial_rollout_length"] = self._partial_rollout_length
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
