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

"""Normalizer class for observation and action normalization in RL."""

import abc
import copy
from typing import Any, Dict, Optional, Sequence, Union
from absl import logging
import gym
from gym import spaces
from gym.spaces import utils
import numpy as np

_EPSILON = 1e-8
OBS_NORM_STATE = "obs_norm_state"
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


class Normalizer(abc.ABC):
  """Base Normalizer class."""

  def __init__(
      self, space: gym.Space, ignored_keys: Optional[Sequence[str]] = None
  ) -> None:
    """Initializes Normalizer.

    Args:
      space: Gym environment observation or action space whose samples will be
        normalized.
      ignored_keys: For dictionary input, certain keys can be ignored during
        normalization.
    """
    self._space = space
    self._space_ignored = None
    self._ignored_keys = ignored_keys or []
    if self._ignored_keys and isinstance(space, spaces.Dict):
      self._space = {}
      self._space_ignored = {}
      for key in space.spaces.keys():
        if key not in self._ignored_keys:
          self._space[key] = space[key]
        else:
          self._space_ignored[key] = space[key]
      self._space = spaces.Dict(self._space)
      self._space_ignored = spaces.Dict(self._space_ignored)
    self._flat_space = utils.flatten_space(self._space)
    self._state = {}

  @property
  @abc.abstractmethod
  def buffer(self) -> Buffer:
    """Buffer for collecting normalization statistics."""

  @abc.abstractmethod
  def __call__(
      self,
      value: Union[np.ndarray, Dict[str, np.ndarray]],
      update_buffer: bool = True,
  ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
    """Apply normalization."""

  def _filter_ignored_input(
      self, input_dict: Dict[str, np.ndarray]
  ) -> Dict[str, np.ndarray]:
    ignored_dict = {}
    for key in self._ignored_keys:
      ignored_dict[key] = input_dict[key]
      del input_dict[key]
    return ignored_dict

  def _add_ignored_input(
      self,
      input_dict: Dict[str, np.ndarray],
      ignored_input: Dict[str, np.ndarray],
  ) -> Dict[str, np.ndarray]:
    if isinstance(input_dict, dict):
      input_dict.update(ignored_input)
    return input_dict

  @property
  def state(self) -> Dict[str, np.ndarray]:
    return self._state.copy()

  @state.setter
  def state(self, state: Dict[str, np.ndarray]) -> None:
    self._state = state.copy()


class NoNormalizer(Normalizer):
  """No Normalization applied to input."""

  @property
  def buffer(self) -> Buffer:
    return NoOpBuffer()

  def __call__(
      self,
      value: Union[np.ndarray, Dict[str, np.ndarray]],
      update_buffer: bool = True,
  ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
    del update_buffer
    return value


class ActionRangeDenormalizer(Normalizer):
  """Actions mapped to given range from [-1, 1]."""

  def __init__(
      self, space: gym.Space, ignored_keys: Optional[Sequence[str]] = None
  ) -> None:
    super().__init__(space, ignored_keys)
    low = self._flat_space.low
    high = self._flat_space.high
    self._state["mid"] = (low + high) / 2.0
    self._state["half_range"] = (high - low) / 2.0

  @property
  def buffer(self) -> Buffer:
    return NoOpBuffer()

  def __call__(
      self,
      action: Union[np.ndarray, Dict[str, np.ndarray]],
      update_buffer: bool = True,
  ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
    """Maps actions from range [-1, 1] to the range in given action space.

    Args:
      action: Normalized action in the range [-1, 1].
      update_buffer: Whether to update buffer based on this action.

    Returns:
      De-normalized action in the action space range.
    """
    del update_buffer  # No buffer to update
    action = action.copy()
    ignored_action = self._filter_ignored_input(action)
    action = utils.flatten(self._space, action)
    action = (action * self._state["half_range"]) + self._state["mid"]
    action = utils.unflatten(self._space, action)
    action = self._add_ignored_input(action, ignored_action)
    return action


class ObservationRangeNormalizer(Normalizer):
  """Observations mapped from given range to [-1, 1]."""

  def __init__(
      self, space: gym.Space, ignored_keys: Optional[Sequence[str]] = None
  ) -> None:
    super().__init__(space, ignored_keys)
    low = self._flat_space.low
    high = self._flat_space.high
    self._state["mid"] = (low + high) / 2.0
    self._state["half_range"] = (high - low) / 2.0

  @property
  def buffer(self) -> Buffer:
    return NoOpBuffer()

  def __call__(
      self,
      observation: Union[np.ndarray, Dict[str, np.ndarray]],
      update_buffer: bool = True,
  ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
    """Maps observations from range in given observation space to [-1, 1].

    Args:
      observation: Unnormalized observation in the observation space range.
      update_buffer: Whether to update buffer based on this action.

    Returns:
      Normalized observation in the range [-1, 1].
    """
    del update_buffer  # No buffer to update
    observation = observation.copy()
    ignored_observation = self._filter_ignored_input(observation)

    observation = utils.flatten(self._space, observation)
    observation = (observation - self._state["mid"]) / self._state["half_range"]
    observation = utils.unflatten(self._space, observation)
    observation = self._add_ignored_input(observation, ignored_observation)
    return observation


class RunningMeanStdNormalizer(Normalizer):
  """Standardize observations with mean and std calculated online."""

  def __init__(
      self, space: gym.Space, ignored_keys: Optional[Sequence[str]] = None
  ) -> None:
    super().__init__(space, ignored_keys)
    shape = self._flat_space.shape
    self._state[MEAN] = np.zeros(shape, dtype=np.float64)
    self._state[STD] = np.ones(shape, dtype=np.float64)
    self._buffer = MeanStdBuffer(shape)

  @property
  def buffer(self) -> MeanStdBuffer:
    return self._buffer

  def __call__(
      self,
      observation: Union[np.ndarray, Dict[str, np.ndarray]],
      update_buffer: bool = True,
  ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
    observation = observation.copy()
    ignored_observation = self._filter_ignored_input(observation)
    observation = utils.flatten(self._space, observation)
    if update_buffer:
      self._buffer.push(observation)
    observation -= self._state[MEAN]
    observation /= self._state[STD] + _EPSILON
    observation = utils.unflatten(self._space, observation)
    observation = self._add_ignored_input(observation, ignored_observation)
    return observation


class RunningMeanStdAgentVsAgentNormalizer(RunningMeanStdNormalizer):
  """Standardize observations with mean and std calculated online."""

  def __init__(self, space: gym.Space) -> None:
    # We use the "ignored_keys" to split the agent obs to process individually.
    super().__init__(space, ignored_keys=["opp"])
    self._buffer = MeanStdBuffer(shape=self._flat_space.shape)

  @property
  def buffer(self) -> MeanStdBuffer:
    return self._buffer

  def __call__(
      self,
      observation: Union[np.ndarray, Dict[str, np.ndarray]],
      update_buffer: bool = True,
  ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
    observation = observation.copy()

    opp_observation = self._filter_ignored_input(observation)
    opp_observation = utils.flatten(self._space_ignored, opp_observation)

    arm_observation = utils.flatten(self._space, observation)
    if update_buffer:
      self._buffer.push(arm_observation)
      # Adding the opponent observations to the buffer greatly speeds learning
      # in self-play scenarios.
      self._buffer.push(opp_observation)

    def _normalized(obs, unflatten_space):
      obs -= self._state[MEAN]
      obs /= self._state[STD] + _EPSILON
      obs = utils.unflatten(unflatten_space, obs)
      return obs

    opp_observation = _normalized(opp_observation, self._space_ignored)
    arm_observation = _normalized(arm_observation, self._space)
    return self._add_ignored_input(arm_observation, opp_observation)
