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

"""Worker class for evaluating a blackbox function."""

import abc
from typing import Any

from iris.workers import worker_util
import numpy as np


FloatLike = float | np.float32 | np.float64


class Worker(abc.ABC):
  """Class for evaluating a blackbox function."""

  def __init__(self, worker_id: int, worker_type: str = "main") -> None:
    self._worker_id = worker_id
    self._worker_type = worker_type
    self._init_state = {}

  @abc.abstractmethod
  def work(
      self,
      params_to_eval: Any,
      enable_logging: bool = False,
  ) -> worker_util.EvaluationResult:
    """Runs the blackbox function on input vars."""

  def get_init_state(self):
    return self._init_state
