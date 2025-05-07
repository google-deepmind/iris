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

"""Buffer class for storing streaming observations and actions in RL."""

import abc
import copy
from typing import Any, Dict, Sequence
from absl import logging
import numpy as np

MEAN = "mean"
STD = "std"
N = "n"
UNNORM_VAR = "unnorm_var"


class Buffer(abc.ABC):
  """Buffer class for collecting online statistics from data."""

  @abc.abstractmethod
  def reset(self) -> None:
    """Reset buffer."""

  @abc.abstractmethod
  def push(self, x: np.ndarray) -> None:
    """Push new data point."""

  @abc.abstractmethod
  def merge(self, data: Dict[str, Any]) -> None:
    """Merge data from another buffer."""

  @property
  @abc.abstractmethod
  def data(self) -> Dict[str, Any]:
    """Returns copy of current data in buffer."""

  @data.setter
  @abc.abstractmethod
  def data(self, new_data: Dict[str, Any]) -> None:
    """Sets data of buffer."""

  @property
  @abc.abstractmethod
  def shape(self) -> Sequence[int]:
    """Shape of data point."""

  @property
  @abc.abstractmethod
  def state(self) -> Dict[str, Any]:
    """Returns copy of current state of buffer."""

  @state.setter
  @abc.abstractmethod
  def state(self, new_state: Dict[str, Any]) -> None:
    """Sets state of buffer."""


class NoOpBuffer(Buffer):
  """No-op buffer."""

  def reset(self) -> None:
    pass

  def push(self, x: np.ndarray) -> None:
    pass

  def merge(self, data: Dict[str, Any]) -> None:
    pass

  @property
  def data(self) -> Dict[str, Any]:
    return {}

  @data.setter
  def data(self, new_data: Dict[str, Any]) -> None:
    pass

  @property
  def shape(self) -> Sequence[int]:
    return ()

  @property
  def state(self) -> Dict[str, Any]:
    return {}

  @state.setter
  def state(self, new_state: Dict[str, Any]) -> None:
    pass


class MeanStdBuffer(Buffer):
  """Collect stats for calculating mean and std online."""

  def __init__(self, shape: Sequence[int] = (0,)) -> None:
    self._shape = shape
    self._data = {
        N: 0,
        MEAN: np.zeros(self._shape, dtype=np.float64),
        UNNORM_VAR: np.zeros(self._shape, dtype=np.float64),
    }
    self.reset()

  def reset(self) -> None:
    self._data[N] = 0
    self._data[MEAN] = np.zeros(self._shape, dtype=np.float64)
    self._data[UNNORM_VAR] = np.zeros(self._shape, dtype=np.float64)

  def push(self, x: np.ndarray) -> None:
    n1 = self._data[N]
    self._data[N] += 1
    if self._data[N] == 1:
      self._data[MEAN] = x.copy()
    else:
      delta = x - self._data[MEAN]
      self._data[MEAN] += delta / self._data[N]
      self._data[UNNORM_VAR] += delta * delta * n1 / self._data[N]

  def merge(self, data: Dict[str, Any]) -> None:
    """Merge data from another buffer."""
    n1 = self._data[N]
    n2 = data[N]
    n = n1 + n2
    if n <= 0:
      logging.warning(
          "Cannot merge data from another buffer due to "
          "both buffers are empty: n1: %i, n2: %i",
          n1,
          n2,
      )
      return

    if (
        not np.isfinite(data[MEAN]).all()
        or not np.isfinite(data[UNNORM_VAR]).all()
    ):
      logging.info(
          "Infinite value found when merging obs_norm_buffer_data,"
          " skipping: %s",
          data,
      )
      return

    m2 = data[MEAN]
    delta = self._data[MEAN] - m2
    delta_sq = delta * delta
    mean = (n1 * self._data[MEAN] + n2 * m2) / n
    s2 = data[UNNORM_VAR]
    unnorm_var = self._data[UNNORM_VAR] + s2 + delta_sq * n1 * n2 / n
    self._data[N] = n
    self._data[MEAN] = mean
    self._data[UNNORM_VAR] = unnorm_var

  @property
  def data(self) -> Dict[str, Any]:
    return copy.deepcopy(self._data)

  @data.setter
  def data(self, new_data: Dict[str, Any]) -> None:
    self._data = copy.deepcopy(new_data)

  @property
  def shape(self) -> Sequence[int]:
    return self._shape

  @property
  def state(self) -> Dict[str, Any]:
    return {MEAN: self._data[MEAN], STD: self._std, N: self._data[N]}

  @state.setter
  def state(self, new_state: Dict[str, Any]) -> None:
    new_state = copy.deepcopy(new_state)
    self._data[MEAN] = new_state[MEAN]
    self._data[N] = new_state[N]

    std = new_state[STD]
    std[std == float("inf")] = 0
    var = np.square(std)
    unnorm_var = (
        var * (self._data[N] - 1)
        if self._data[N] > 1
        else np.zeros_like(self._data[MEAN])
    )
    self._data[UNNORM_VAR] = unnorm_var

  @property
  def _var(self) -> np.ndarray:
    return (
        self._data[UNNORM_VAR] / (self._data[N] - 1)
        if self._data[N] > 1
        else np.ones_like(self._data[MEAN])
    )

  @property
  def _std(self) -> np.ndarray:
    # asarray is needed for boolean indexing to work when shape = (1)
    std = np.asarray(np.sqrt(self._var))
    std[std < 1e-7] = float("inf")
    return std
