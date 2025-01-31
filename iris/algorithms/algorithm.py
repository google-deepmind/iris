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

"""Algorithm class for distributed blackbox optimization library."""

import abc
import pathlib
from typing import Any, Dict, Sequence, Union
from iris.workers import worker_util
import numpy as np


PARAMS_TO_EVAL = "params_to_eval"
OBS_NORM_BUFFER_STATE = "obs_norm_buffer_state"
UPDATE_OBS_NORM_BUFFER = "update_obs_norm_buffer"


class BlackboxAlgorithm(abc.ABC):
  """Base class for Blackbox optimization algorithms."""

  def __init__(self,
               num_suggestions: int,
               random_seed: int,
               num_evals: int = 50) -> None:
    """Initializes the blackbox algorithm.

    Args:
      num_suggestions: Number of suggestions to sample for blackbox function
        evaluation.
      random_seed: Seed for numpy random state.
      num_evals: Number of times to evaluate blackbox function while reporting
        performance of current parameters.
    """
    self._num_suggestions = num_suggestions
    self._num_evals = num_evals
    self._np_random_state = np.random.RandomState(random_seed)
    self._opt_params = np.empty(0)

  @property
  def opt_params(self):
    """Returns the optimizer parameters."""
    return self._opt_params

  @abc.abstractmethod
  def initialize(self, state: Dict[str, Any]) -> None:
    """Initializes the algorithm from initial worker state."""
    raise NotImplementedError(
        "Should be implemented in derived classes for specific algorithms.")

  @abc.abstractmethod
  def get_param_suggestions(self,
                            evaluate: bool = False) -> Sequence[Dict[str, Any]]:
    """Suggests a list of inputs to evaluate the Blackbox function on."""
    raise NotImplementedError(
        "Should be implemented in derived classes for specific algorithms.")

  @abc.abstractmethod
  def process_evaluations(self,
                          eval_results: Sequence[worker_util.EvaluationResult]):
    """Processes the list of Blackbox function evaluations return from workers.

    Args:
      eval_results: List containing Blackbox function evaluations based on the
        order in which the suggestion were sent. The value is a tuple of
        suggestion evaluated and the result after evaluation.
    """
    del eval_results
    raise NotImplementedError(
        "Should be implemented in derived classes for specific algorithms.")

  @property
  def state(self):
    return {PARAMS_TO_EVAL: self._opt_params}

  @state.setter
  def state(self, new_state: Dict[str, Any]) -> None:
    self._opt_params = new_state[PARAMS_TO_EVAL]

  def restore_state_from_checkpoint(self, new_state: Dict[str, Any]) -> None:
    self.state = new_state[PARAMS_TO_EVAL]

  def maybe_save_custom_checkpoint(self,
                                   state: Dict[str, Any],
                                   checkpoint_path: Union[pathlib.Path, str]
                                   ) -> None:
    """If implemented, saves a custom checkpoint to checkpoint_path."""
    del state, checkpoint_path
    return None
