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

"""Utility functions for manipulating gym spaces."""

from typing import Dict, Sequence, Union
import gym
import numpy as np


def filter_space(space: gym.Space,
                 selected: Sequence[Union[str, int]]) -> gym.Space:
  """Filters gym space based on list of selected keys or dimensions."""
  if isinstance(space, gym.spaces.Box):
    low = space.low.take(selected)
    high = space.high.take(selected)
    return gym.spaces.Box(low=low, high=high)
  elif isinstance(space, gym.spaces.Dict):
    filtered_space = {}
    for sensor in selected:
      filtered_space[sensor] = space[sensor]
    return gym.spaces.Dict(filtered_space)
  else:
    raise NotImplementedError


def extend_space(space: gym.Space, key: str, value: gym.Space):
  """Adds new keys or dimensions to the space."""
  if isinstance(space, gym.spaces.Box):
    low = np.concatenate((space.low, value.low))
    high = np.concatenate((space.high, value.high))
    return gym.spaces.Box(low=low, high=high)
  elif isinstance(space, gym.spaces.Dict):
    extended_space = dict(space.spaces)
    if key not in extended_space:
      extended_space[key] = value
    else:
      raise Exception(
          "The key to be extened, '{}' is already present in the space.".format(
              key))
    return gym.spaces.Dict(extended_space)
  else:
    raise NotImplementedError


def filter_sample(
    x: Union[np.ndarray, Dict[str, np.ndarray]],
    selected: Sequence[Union[str, int]]
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
  """Filters input data based on list of selected keys or dimensions."""
  if isinstance(x, dict):
    filtered_x = {}
    for sensor in selected:
      filtered_x[sensor] = x[sensor]
  else:
    filtered_x = np.array(x).take(selected)
  return filtered_x


def extend_sample(
    x: Union[np.ndarray, Dict[str, np.ndarray]], key: str,
    value: np.ndarray) -> Union[np.ndarray, Dict[str, np.ndarray]]:
  """Adds new keys or dimensions to the input data."""
  x = x.copy()
  if isinstance(x, dict):
    if key not in x:
      x[key] = value
    else:
      raise Exception(
          "The key to be extened, '{}' is already present in the dict.".format(
              key))
  else:
    x = np.concatenate((value, x))
  return x
