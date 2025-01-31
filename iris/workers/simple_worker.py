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

"""Simple worker for evaluating any given blackbox function."""

from typing import Any, Callable

from absl import logging
from iris.workers import worker
from iris.workers import worker_util
import numpy as np


class SimpleWorker(worker.Worker):
  """Class for evaluating a given blackbox function."""

  def __init__(
      self,
      blackbox_function: Callable[[np.ndarray], worker.FloatLike],
      initial_params: np.ndarray | None = None,
      init_function: Callable[..., Any] | None = None,
      **kwargs
  ) -> None:
    super().__init__(**kwargs)
    self._extra_args = {}
    if init_function is not None:
      initial_params, self._extra_args = init_function()
    self._init_state["init_params"] = initial_params
    self._blackbox_function = blackbox_function

  def work(
      self,
      params_to_eval: np.ndarray,
      enable_logging: bool = False,
  ) -> worker_util.EvaluationResult:
    """Runs the blackbox function on input suggestion."""
    value = self._blackbox_function(params_to_eval, **self._extra_args)
    if enable_logging:
      logging.info("Value: %f", value)
    evaluation_result = worker_util.EvaluationResult(params_to_eval, value)
    return evaluation_result
