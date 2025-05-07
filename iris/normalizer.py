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
from typing import Dict, Optional, Sequence, Union
import gym
from gym import spaces
from gym.spaces import utils
from iris import buffer
import numpy as np

_EPSILON = 1e-8
OBS_NORM_STATE = "obs_norm_state"


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
  def buffer(self) -> buffer.Buffer:
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
  def buffer(self) -> buffer.Buffer:
    return buffer.NoOpBuffer()

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
  def buffer(self) -> buffer.Buffer:
    return buffer.NoOpBuffer()

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
  def buffer(self) -> buffer.Buffer:
    return buffer.NoOpBuffer()

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
    self._state[buffer.MEAN] = np.zeros(shape, dtype=np.float64)
    self._state[buffer.STD] = np.ones(shape, dtype=np.float64)
    self._buffer = buffer.MeanStdBuffer(shape)

  @property
  def buffer(self) -> buffer.MeanStdBuffer:
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
    observation -= self._state[buffer.MEAN]
    observation /= self._state[buffer.STD] + _EPSILON
    observation = utils.unflatten(self._space, observation)
    observation = self._add_ignored_input(observation, ignored_observation)
    return observation


class RunningMeanStdAgentVsAgentNormalizer(RunningMeanStdNormalizer):
  """Standardize observations with mean and std calculated online."""

  def __init__(self, space: gym.Space) -> None:
    # We use the "ignored_keys" to split the agent obs to process individually.
    super().__init__(space, ignored_keys=["opp"])
    self._buffer = buffer.MeanStdBuffer(shape=self._flat_space.shape)

  @property
  def buffer(self) -> buffer.MeanStdBuffer:
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
      obs -= self._state[buffer.MEAN]
      obs /= self._state[buffer.STD] + _EPSILON
      obs = utils.unflatten(unflatten_space, obs)
      return obs

    opp_observation = _normalized(opp_observation, self._space_ignored)
    arm_observation = _normalized(arm_observation, self._space)
    return self._add_ignored_input(arm_observation, opp_observation)
