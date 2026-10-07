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

"""Policy class that computes action by running keras neural networks."""

from typing import Dict, List, Union

import gym
from gym.spaces import utils
from iris.policies import base_policy
import numpy as np
import tensorflow as tf


class KerasPolicy(base_policy.BasePolicy):
  """Policy class that computes action by running keras models."""

  def __init__(self, ob_space: gym.Space, ac_space: gym.Space,
               **kwargs) -> None:
    """Initializes a keras policy. See the base class for more details."""
    super().__init__(ob_space=ob_space, ac_space=ac_space)
    tf.compat.v1.enable_eager_execution()
    self._build_model(**kwargs)
    self._weights = self.get_weights()

  def _build_model(self) -> None:
    self.model = tf.keras.models.Model()

  def get_layers(self) -> List[tf.keras.layers.Layer]:
    return self.model.layers

  def get_representation_layers(self) -> List[tf.keras.layers.Layer]:
    return []

  def update_weights(self, new_weights: np.ndarray) -> None:
    """Sets the weights of the neural network.

    Args:
      new_weights: An 1D list of all weights of the neural network.
    """
    layers = self.get_layers()
    current_id = 0
    for index, layer in enumerate(layers):
      layer_weights = layer.get_weights()
      new_layer_weights = []
      for ith_component in range(len(layer_weights)):  # weights and biases
        w = layer_weights[ith_component]
        weight_shape = w.shape
        weight_len = len(w.flatten().tolist())
        new_layer_weights.append(
            np.reshape(new_weights[current_id:current_id + weight_len],
                       weight_shape))
        current_id += weight_len
      layers[index].set_weights(new_layer_weights)
    self._weights = new_weights[:]

  def get_weights(self):
    """Gets the weights of the neural network policy.

    Returns:
      An 1D list of all weights in the neural network.
    """
    weights = []
    for layer in self.get_layers():
      for w in layer.get_weights():
        weights.extend(w.flatten().tolist())
    return np.array(weights)

  def get_representation_weights(self):
    """Gets all the representation weights.

    Returns:
      List of 1D list of weights in the neural network.
    """
    weights = []
    for layer in self.get_representation_layers():
      for w in layer.get_weights():
        weights.extend(w.flatten().tolist())
    return np.array(weights)

  def update_representation_weights(self, new_representation_weights):
    """Sets all the representation weights."""
    layers = self.get_representation_layers()
    current_id = 0
    for index, layer in enumerate(layers):
      layer_weights = layer.get_weights()
      new_layer_weights = []
      for ith_component in range(len(layer_weights)):  # weights and biases
        w = layer_weights[ith_component]
        weight_shape = w.shape
        weight_len = len(w.flatten().tolist())
        new_layer_weights.append(
            np.reshape(
                new_representation_weights[current_id:current_id + weight_len],
                weight_shape))
        current_id += weight_len
      layers[index].set_weights(new_layer_weights)
    self._representation_weights = new_representation_weights[:]

  def act(self, ob: Union[np.ndarray, Dict[str, np.ndarray]]
          ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
    """Maps the observation to action.

    Args:
      ob: The observations in reinforcement learning.

    Returns:
      The actions in reinforcement learning.
    """
    ob = utils.flatten(self._ob_space, ob)
    actions = self.model(np.array([ob])).numpy()[0]
    actions = utils.unflatten(self._ac_space, actions)
    return actions
