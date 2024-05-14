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

"""Feed-forward fully connected neural network policy."""

from typing import Dict, Sequence, Union
import gym
from gym.spaces import utils
from iris.policies import base_policy
import numpy as np


class FullyConnectedNeuralNetworkPolicy(base_policy.BasePolicy):
  """Feed-forward fully connected neural network policy."""

  def __init__(self,
               ob_space: gym.Space,
               ac_space: gym.Space,
               hidden_layer_sizes: Sequence[int],
               activation: str = "tanh") -> None:
    """Initializes the linear policy. See the base class for more details."""

    super().__init__(ob_space, ac_space)
    self._hidden_layer_sizes = hidden_layer_sizes
    self._activation = activation
    if self._activation == "tanh":
      self._activation = np.tanh
    elif self._activation == "clip":
      self._activation = lambda x: np.clip(x, -1.0, 1.0)
    self._layer_sizes = [self._ob_dim]
    self._layer_sizes.extend(self._hidden_layer_sizes)
    self._layer_sizes.append(self._ac_dim)
    self._layer_weight_start_idx = []
    self._layer_weight_end_idx = []
    num_weights = 0
    num_layers = len(self._layer_sizes)
    for ith_layer in range(num_layers - 1):
      self._layer_weight_start_idx.append(num_weights)
      num_weights += (
          self._layer_sizes[ith_layer] * self._layer_sizes[ith_layer + 1])
      self._layer_weight_end_idx.append(num_weights)
    self._weights = np.zeros(num_weights, dtype=np.float64)

  def act(self, ob: Union[np.ndarray, Dict[str, np.ndarray]]
          ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
    """Maps the observation to action.

    Args:
      ob: The observations in reinforcement learning.
    Returns:
      The actions in reinforcement learning.
    """
    ob = utils.flatten(self._ob_space, ob)
    ith_layer_result = ob
    num_layers = len(self._layer_sizes)
    for ith_layer in range(num_layers - 1):
      start = self._layer_weight_start_idx[ith_layer]
      end = self._layer_weight_end_idx[ith_layer]
      mat_weight = np.reshape(
          self._weights[start:end],
          (self._layer_sizes[ith_layer + 1], self._layer_sizes[ith_layer]))
      ith_layer_result = np.dot(mat_weight, ith_layer_result)
      ith_layer_result = self._activation(ith_layer_result)
    actions = ith_layer_result
    actions = utils.unflatten(self._ac_space, actions)
    return actions
