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

"""Policy class for computing action from weights and observation vector."""

from typing import Dict, Union

import gym
from gym.spaces import utils
import numpy as np


class BasePolicy(object):
  """Base policy class for reinforcement learning."""

  def __init__(self, ob_space: gym.Space, ac_space: gym.Space) -> None:
    """Initializes the policy.

    Args:
      ob_space: Input observation space.
      ac_space: Output action space.
    """
    self._ob_space = ob_space
    self._ac_space = ac_space
    self._ob_dim = utils.flatdim(self._ob_space)
    self._ac_dim = utils.flatdim(self._ac_space)
    self._weights = np.empty(0)
    self._representation_weights = np.empty(0)
    self._iteration = 0

  def get_iteration(self) -> int:
    """Returns the iteration."""
    return self._iteration

  def set_iteration(self, value: int | None):
    """Sets the coordinator iteration number.

    Args:
      value: Optional integer value. If the value is None, the internal
        iteration counter is not updated.
    """
    if value is None:
      return
    self._iteration = value

  def update_weights(self, new_weights: np.ndarray) -> None:
    self._weights[:] = new_weights[:]

  def get_weights(self) -> np.ndarray:
    return self._weights

  def get_representation_weights(self):
    return self._representation_weights

  def update_representation_weights(
      self, new_representation_weights: np.ndarray) -> None:
    self._representation_weights[:] = new_representation_weights[:]

  def reset(self):
    pass

  def act(self, ob: Union[np.ndarray, Dict[str, np.ndarray]]
          ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
    """Maps the observation to action."""
    raise NotImplementedError(
        "Should be implemented in derived classes for specific policies.")
