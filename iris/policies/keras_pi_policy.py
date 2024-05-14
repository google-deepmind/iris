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

"""Policy network on learned predictive information representations."""
from typing import Dict, List, Optional, Sequence, Union

import gym
from gym import spaces
from gym.spaces import utils
from iris.policies import keras_policy
from iris.policies import spatial_softmax
import numpy as np
import tensorflow as tf


class KerasPIPolicy(keras_policy.KerasPolicy):
  """Policy network on learned predictive information representations."""

  def __init__(self, ob_space: gym.Space, ac_space: gym.Space,
               **kwargs) -> None:
    """Initializes a keras CNN policy. See the base class for more details."""
    super().__init__(ob_space=ob_space, ac_space=ac_space, **kwargs)
    # Build the encoder h that outputs [hidden_state, hidden_state_vision_only]
    self.build_h(**kwargs)
    # Build the decoder f that outputs imitation policy action and value
    self.build_f(**kwargs)
    # Build latent transition and reward prediction function g
    self.build_g(**kwargs)
    # Build projection head px for input states
    self.build_px(**kwargs)
    # Build projection head py for target states
    self.build_py(**kwargs)
    self._representation_weights = self.get_representation_weights()

  def get_representation_layers(self) -> List[tf.keras.layers.Layer]:
    representation_layers = self.h_model.layers
    representation_layers += self.f_model.layers + self.g_model.layers
    representation_layers += self.px_model.layers + self.py_model.layers
    return representation_layers

  def _create_vision_input_layers(self):
    vision_input_layers = []
    for image_label in self._image_input_labels:
      image_size = self._ob_space[image_label].shape
      vision_input_layers.append(tf.keras.layers.Input(
          batch_input_shape=(1, image_size[0], image_size[1], image_size[2]),
          dtype="float",
          name="vision_input" + image_label))
    return vision_input_layers

  def _create_other_input_layer(self):
    if isinstance(self._ob_space, gym.spaces.Box):
      self._other_ob_space = self._ob_space
    else:
      self._other_ob_space = self._ob_space.spaces.copy()
      for input_label in self._image_input_labels:
        del self._other_ob_space[input_label]
      self._other_ob_space = spaces.Dict(self._other_ob_space)
    self._other_ob_dim = utils.flatdim(self._other_ob_space)
    if self._other_ob_dim > 0:
      return tf.keras.layers.Input(
          batch_input_shape=(1, self._other_ob_dim),
          dtype="float",
          name="other_input")
    return None

  def _create_vision_processing_layers(
      self,
      x: tf.keras.layers.Layer,
      conv_filter_sizes: Sequence[int],
      conv_kernel_sizes: Sequence[int],
      image_feature_length: int,
      pool_sizes: Optional[Sequence[int]] = None,
      pool_strides: Optional[Sequence[int]] = None,
      final_vision_activation: str = "relu",
      use_spatial_softmax: bool = False,
      **kwargs) -> tf.keras.layers.Layer:
    """Create keras layers for CNN image processing.

    Args:
      x: Input keras layer.
      conv_filter_sizes: A list of filter sizes (number of output channels) of
        the conv layers.
      conv_kernel_sizes: A list of kernel sizes for the conv layers. An element
        n means a nxn kernel.
      image_feature_length: The length of the image feature vector.
      pool_sizes: A list of pool sizes after the conv layers. If an element is
        None, then there will be no pooling for the corresponding conv layer.
      pool_strides: The strides of pooling.
      final_vision_activation: Activation for final vision output.
      use_spatial_softmax: Whether to use spatial softmax to aggregate the CNN
        output. If false, CNN output is ismple flattened.
      **kwargs: Arguments for creating encoders.

    Returns:
     Tf keras layer after vision processing.
    """
    # Convolution and pooling layers.
    if pool_sizes is None:
      pool_sizes = [None] * len(conv_filter_sizes)
    if pool_strides is None:
      pool_strides = [None] * len(conv_filter_sizes)

    for filter_size, kernel_size, pool_size, pool_stride in zip(
        conv_filter_sizes, conv_kernel_sizes, pool_sizes, pool_strides):
      x = tf.keras.layers.Conv2D(
          filter_size,
          kernel_size=kernel_size,
          padding="valid",
          activation=final_vision_activation)(
              x)
      if pool_size is not None:
        x = tf.keras.layers.MaxPool2D(
            pool_size=pool_size, strides=pool_stride)(
                x)

    # Flattening or spatial softmax on image feature map.
    if use_spatial_softmax:
      x = spatial_softmax.SpatialSoftmax(data_format="channels_last")(x)
    else:
      x = tf.keras.layers.Flatten()(x)

    # Encoding image into a feature vector.
    return tf.keras.layers.Dense(image_feature_length,
                                 activation=final_vision_activation)(x)

  def _build_model(self,  # pytype: disable=signature-mismatch  # overriding-parameter-count-checks
                   state_dim: int,
                   fc_layer_sizes: Sequence[int],
                   **kwargs) -> None:
    # hidden state input
    state_input = tf.keras.layers.Input(
        batch_input_shape=(1, state_dim),
        dtype="float",
        name="s_input")

    # policy
    x = state_input
    for fc_layer_size in fc_layer_sizes:
      x = tf.keras.layers.Dense(fc_layer_size, activation="tanh")(x)
    action_output = tf.keras.layers.Dense(self._ac_dim, activation="tanh")(x)
    self.model = tf.keras.models.Model(inputs=state_input,
                                       outputs=[action_output])

  def build_h(
      self,
      h_fc_layer_sizes: Sequence[int],
      image_input_label: Optional[Union[Sequence[str], str]] = None,
      **kwargs):
    # image_input_label: Label of image input in observation dictionary.
    if image_input_label is None:
      self._image_input_labels = []
    elif isinstance(image_input_label, str):
      self._image_input_labels = [image_input_label]
    else:
      self._image_input_labels = image_input_label

    inputs = self._create_vision_input_layers()

    vision_outputs = []
    for vision_input in inputs:
      vision_outputs.append(
          self._create_vision_processing_layers(x=vision_input, **kwargs))
    vision_output = tf.keras.layers.concatenate(
        vision_outputs) if vision_outputs else None

    # Add other sensor observations.
    other_input = self._create_other_input_layer()
    if other_input is None:
      x = vision_output
    else:
      inputs.append(other_input)
      x = other_input
      if vision_output is not None:
        x = tf.keras.layers.concatenate([vision_output, other_input])
    if x is None:
      raise ValueError("No input provided")

    # state: fully connected layers.
    for h_fc_layer_size in h_fc_layer_sizes:
      x = tf.keras.layers.Dense(h_fc_layer_size, activation="tanh")(x)
    outputs = [x, vision_output] if vision_output is not None else x
    self.h_model = tf.keras.models.Model(inputs=inputs, outputs=outputs)

  def build_f(self,
              state_dim: int,
              f_fc_layer_sizes: Sequence[int],
              num_supports: int = 51,
              **kwargs):
    state_input = tf.keras.layers.Input(
        batch_input_shape=(1, state_dim),
        dtype="float",
        name="s_input")
    x = state_input
    for f_fc_layer_size in f_fc_layer_sizes:
      x = tf.keras.layers.Dense(f_fc_layer_size, activation="tanh")(x)
    p = tf.keras.layers.Dense(self._ac_dim, activation="tanh")(x)
    v = tf.keras.layers.Dense(num_supports)(x)
    self.f_model = tf.keras.models.Model(inputs=state_input, outputs=[p, v])

  def build_g(self,
              state_dim: int,
              g_fc_layer_sizes: Sequence[int],
              **kwargs):
    state_input = tf.keras.layers.Input(
        batch_input_shape=(1, state_dim),
        dtype="float",
        name="s_input")
    action_input = tf.keras.layers.Input(
        batch_input_shape=(1, self._ac_dim),
        dtype="float",
        name="action_input")

    x = tf.keras.layers.concatenate([state_input, action_input])
    for g_fc_layer_size in g_fc_layer_sizes:
      x = tf.keras.layers.Dense(g_fc_layer_size, activation="tanh")(x)
    u_next = tf.keras.layers.Dense(1)(x)
    s_next = tf.keras.layers.Dense(state_dim, activation="tanh")(x)
    self.g_model = tf.keras.models.Model(inputs=[state_input, action_input],
                                         outputs=[u_next, s_next])

  def build_px(self,
               state_dim: int,
               **kwargs):
    state_input = tf.keras.layers.Input(
        batch_input_shape=(1, state_dim),
        dtype="float",
        name="s_input")

    x = state_input
    x = tf.keras.layers.Dense(64, activation="tanh")(x)
    z = tf.keras.layers.Dense(state_dim)(x)
    self.px_model = tf.keras.models.Model(inputs=state_input, outputs=z)

  def build_py(self,
               state_dim: int,
               image_feature_length: int,
               **kwargs):
    state_input = tf.keras.layers.Input(
        batch_input_shape=(
            1, image_feature_length * len(self._image_input_labels)),
        dtype="float",
        name="s_input")

    x = state_input
    x = tf.keras.layers.Dense(64, activation="tanh")(x)
    z = tf.keras.layers.Dense(state_dim)(x)
    self.py_model = tf.keras.models.Model(inputs=state_input, outputs=z)

  def act(self, ob: Union[np.ndarray, Dict[str, np.ndarray]]
          ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
    """Maps the observation to action.

    Args:
      ob: The observations in reinforcement learning.

    Returns:
      The actions in reinforcement learning.
    """
    # Gather model inputs.
    inputs = []
    for image_label in self._image_input_labels:
      vision_input = np.array([ob[image_label]])
      inputs.append(vision_input)

    if self._other_ob_dim > 0:
      other_ob = ob.copy()
      for image_label in self._image_input_labels:
        del other_ob[image_label]

      # Flatten other observations.
      other_input = utils.flatten(self._other_ob_space, other_ob)
      inputs.append(np.array([other_input]))

    # Run model.
    s, _ = self.h_model(inputs)
    output = self.model(s)

    # Parse model output.
    actions = output.numpy()

    # Convert computed actions into desired action space dimensions.
    actions = utils.unflatten(self._ac_space, actions)
    return actions

  def rollout(self, ob: Union[np.ndarray, Dict[str, np.ndarray]],
              rollout_length: int) -> np.ndarray:
    # Separate vision input and other observations.
    inputs = []
    for image_label in self._image_input_labels:
      vision_input = np.array(ob[image_label])
      vision_input = np.squeeze(vision_input, axis=0)
      inputs.append(vision_input)

    other_ob = ob.copy()
    for image_label in self._image_input_labels:
      del other_ob[image_label]

    # Flatten other observations.
    other_input = utils.flatten(self._other_ob_space, other_ob)

    # Gather model inputs.
    inputs.append(np.array([other_input]))

    # Run model.
    s, _ = self.h_model(inputs)
    reward = 0.
    for _ in range(rollout_length):
      action = self.model(s)
      u_next, s = self.g_model([s, action])
      reward += u_next
    _, z = self.f_model(s)
    vd = tf.nn.softmax(z)
    supports = tf.linspace(-10.0, 10.0, 51)
    v = tf.reduce_sum(vd * supports[None, ...], axis=-1)
    reward += v
    reward = np.squeeze(reward.numpy(), axis=0)
    return reward
