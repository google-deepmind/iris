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

"""Algorithm class for Covariance Matrix Adaptation Evolutionary Strategy (CMA-ES) Blackbox algorithm."""

from typing import Any, Dict, Optional, Sequence

import cma
from iris import normalizer
from iris import worker_util
from iris.algorithms import algorithm
import numpy as np


class CMAES(algorithm.BlackboxAlgorithm):
  """CMA-ES for blackbox optimization.

     CMA-ES is a blackbox optimization algorithm that interleave between
     sampling new candidate solutions according to a multi-variate gaussian and
     updating the covariance matrix of the multi-variate gaussian based on the
     history data. More details regarding CMA-ES can be found here:
     https://arxiv.org/abs/1604.00772. In this code, we use the pycma package
     to implement the algorithm.
  """

  def __init__(self,
               std: float = 0.3,
               bounds: Sequence[float] = (-1, 1),
               obs_norm_data_buffer: Optional[normalizer.MeanStdBuffer] = None,
               **kwargs) -> None:
    """Initializes the augmented random search algorithm.

    Args:
      std: Initial standard deviation to be used in CMA-ES.
      bounds: Bounds of the search parameters.
      obs_norm_data_buffer: Buffer to sync statistics from all workers for
        online mean std observation normalizer.
      **kwargs: Other keyword arguments for base class.
    """
    super().__init__(**kwargs)
    self._std = std
    self._bounds = bounds
    self._cmaes = cma.CMAEvolutionStrategy(np.empty(5), self._std,
                                           {
                                               "popsize": self._num_suggestions,
                                               "bounds": list(self._bounds)
                                           })
    self._obs_norm_data_buffer = obs_norm_data_buffer

  def initialize(self, state: Dict[str, Any]) -> None:
    """Initializes the algorithm from initial worker state."""
    self._opt_params = state["init_params"]

    self._cmaes = cma.CMAEvolutionStrategy(self._opt_params, self._std,
                                           {
                                               "popsize": self._num_suggestions,
                                               "bounds": list(self._bounds)
                                           })
    # Initialize Observation normalization buffer with init data from the worker
    if self._obs_norm_data_buffer is not None:
      self._obs_norm_data_buffer.data = state["obs_norm_buffer_data"]
    self._best_value = None

  def process_evaluations(self,
                          eval_results: Sequence[worker_util.EvaluationResult]
                          ) -> None:
    """Processes the list of Blackbox function evaluations return from workers.

    Gradient is computed by taking a weighted sum of directions and
    difference of their value from the current value. The current parameter
    vector is then updated in the gradient direction with specified step size.

    Args:
      eval_results: List containing Blackbox function evaluations based on the
        order in which the suggestions were sent.
    """

    filtered_eval_results = [e for e in eval_results if e.params_evaluated.size]
    all_params = np.array([r.params_evaluated for r in filtered_eval_results])
    all_values = np.array([r.value for r in filtered_eval_results])

    if filtered_eval_results:
      if self._best_value is None or np.max(all_values) > self._best_value:
        self._best_value = np.max(all_values)
        self._opt_params = np.copy(all_params[np.argmax(all_values)])

      if len(all_params) == len(all_values) == self._num_suggestions:
        self._cmaes.tell(all_params, -all_values)

    # Update the observation buffer
    if self._obs_norm_data_buffer is not None:
      for r in filtered_eval_results:
        self._obs_norm_data_buffer.merge(r.obs_norm_buffer_data)

  def get_param_suggestions(self,
                            evaluate: bool = False
                            ) -> Sequence[Dict[str, Any]]:
    """Suggests a list of inputs to evaluate the Blackbox function on.

    Suggestions are sampled from a gaussian distribution around the current
    parameter vector. For each suggestion, a dict containing keyword arguments
    for the worker is sent.

    Args:
      evaluate: Whether to evaluate current optimization variables
        for reporting training progress.

    Returns:
      A list of suggested inputs for the workers to evaluate.
    """
    if evaluate:
      param_suggestions = [self._opt_params] * self._num_evals
    else:
      param_suggestions = self._cmaes.ask()
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
