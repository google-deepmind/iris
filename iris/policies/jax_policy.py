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

"""Policy class that computes action by running jax neural networks."""

from typing import Any, Dict, Union

import gym
from gym.spaces import utils
from iris.policies import base_policy
import jax
import numpy as np


class JaxPolicy(base_policy.BasePolicy):
  """Policy class that computes action by running jax models."""

  def __init__(self,
               ob_space: gym.Space,
               ac_space: gym.Space,
               model: Any,
               init_x: Any,
               seed: int = 42) -> None:
    """Initializes a jax policy. See the base class for more details."""
    super().__init__(ob_space=ob_space, ac_space=ac_space)
    self.model = model()
    self._tree_weights = self.model.init(
        jax.random.PRNGKey(seed=seed),
        init_x)
    self._layer_weights, self._tree_def = jax.tree.flatten(self._tree_weights)
    self._weights = self.get_weights()

  def update_tree_weights(self, new_tree_weights: Dict[str, Any]) -> None:
    self._tree_weights = new_tree_weights
    self._layer_weights, self._tree_def = jax.tree.flatten(self._tree_weights)
    self._weights = self.get_weights()

  def update_weights(self, new_weights: np.ndarray) -> None:
    """Sets the weights of the neural network.

    Args:
      new_weights: An 1D list of all weights of the neural network.
    """
    new_layer_weights = []
    current_id = 0
    for w in self._layer_weights:
      w = np.array(w)
      weight_shape = w.shape
      weight_len = len(w.flatten().tolist())
      new_w = np.reshape(new_weights[current_id:current_id + weight_len],
                         weight_shape)
      current_id += weight_len
      new_layer_weights.append(new_w)
    self._layer_weights = new_layer_weights.copy()
    self._tree_weights = jax.tree.unflatten(self._tree_def, self._layer_weights)
    self._weights = new_weights[:]

  def get_weights(self) -> np.ndarray:
    """Gets the weights of the neural network policy.

    Returns:
      An 1D list of all weights in the neural network.
    """
    weights = []
    for w in self._layer_weights:
      weights.extend(np.array(w).flatten().tolist())
    return np.array(weights)

  def act(self, ob: Union[np.ndarray, Dict[str, np.ndarray]]
          ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
    """Maps the observation to action.

    Args:
      ob: The observations in reinforcement learning.

    Returns:
      The action in reinforcement learning.
    """
    ob = utils.flatten(self._ob_space, ob)
    action = self.model.apply(
        self._tree_weights,
        [ob],
        mutable=['batch_stats'])[0][0]
    action = utils.unflatten(self._ac_space, action)
    return action
