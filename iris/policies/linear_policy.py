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

"""Linear policy class that computes action as <w, ob>."""

from typing import Dict, Union

import gym
from gym.spaces import utils
from iris.policies import base_policy
import numpy as np


class LinearPolicy(base_policy.BasePolicy):
  """Linear policy class that computes action as <w, ob>."""

  def __init__(self,
               ob_space: gym.Space,
               ac_space: gym.Space,
               activation: str = "clip") -> None:
    """Initializes the linear policy. See the base class for more details."""

    super().__init__(ob_space, ac_space)
    self._weights = np.zeros(self._ac_dim * self._ob_dim, dtype=np.float64)
    self._activation = activation
    if self._activation == "tanh":
      self._activation = np.tanh
    elif self._activation == "clip":
      self._activation = lambda x: np.clip(x, -1.0, 1.0)

  def act(self, ob: Union[np.ndarray, Dict[str, np.ndarray]]
          ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
    """Maps the observation to action.

    Args:
      ob: The observations in reinforcement learning.

    Returns:
      The actions in reinforcement learning.
    """
    ob = utils.flatten(self._ob_space, ob)
    matrix_weights = np.reshape(self._weights, (self._ac_dim, self._ob_dim))
    actions = self._activation(np.dot(matrix_weights, ob))
    actions = utils.unflatten(self._ac_space, actions)
    return actions
