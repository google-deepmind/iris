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

"""Policy class that computes action by running convolutional neural network."""
from typing import Dict, Optional, Sequence, Union

import gym
from gym import spaces
from gym.spaces import utils
from iris.policies import keras_policy
from iris.policies import spatial_softmax
import numpy as np
import tensorflow as tf


class KerasCNNPolicy(keras_policy.KerasPolicy):
  """Policy class, computes action by running convolutional neural network."""

  def __init__(self, ob_space: gym.Space, ac_space: gym.Space,
               **kwargs) -> None:
    """Initializes a keras CNN policy. See the base class for more details."""
    self._rnn_state = None
    super().__init__(ob_space=ob_space, ac_space=ac_space, **kwargs)

  def _create_vision_input_layers(self):
    vision_input_layers = []
    for image_label in self._image_input_labels:
      image_size = self._ob_space[image_label].shape
      vision_input_layers.append(
          tf.keras.layers.Input(
              batch_input_shape=(1, image_size[0], image_size[1],
                                 image_size[2]),
              dtype="float",
              name="vision_input" + image_label))
    return vision_input_layers

  def _create_other_input_layer(self):
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
      use_spatial_softmax: bool = False) -> tf.keras.layers.Layer:
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
    return tf.keras.layers.Dense(
        image_feature_length, activation=final_vision_activation)(
            x)

  def _create_rnn_layers(self, x, inputs):
    """By default, creates an LSTM."""
    lstm_h_state_input = tf.keras.layers.Input(
        batch_input_shape=(1, self._rnn_units),
        dtype="float",
        name="lstm_h_state_input")
    lstm_c_state_input = tf.keras.layers.Input(
        batch_input_shape=(1, self._rnn_units),
        dtype="float",
        name="lstm_c_state_input")
    inputs.append(lstm_h_state_input)
    inputs.append(lstm_c_state_input)
    h_state = lstm_h_state_input
    c_state = lstm_c_state_input
    x = tf.keras.layers.Reshape((1, -1))(x)
    x, h_state, c_state = tf.keras.layers.LSTM(
        units=self._rnn_units, return_state=True, stateful=True)(
            x, initial_state=[lstm_h_state_input, lstm_c_state_input])
    return x, [h_state, c_state]

  def _build_model(self,  # pytype: disable=signature-mismatch  # overriding-parameter-count-checks
                   fc_layer_sizes: Sequence[int],
                   use_rnn: bool = False,
                   rnn_units: int = 32,
                   image_input_label: Union[Sequence[str], str] = "vision",
                   final_layer_init: str = "glorot_uniform",
                   **kwargs) -> None:
    """Constructs a keras CNN to process vision and other sensor data.

    Args:
      fc_layer_sizes: A list of number of neurons for all the hidden layers.
      use_rnn: Whether to use RNN (default LSTM) layer right after the conv
        layers for memory.
      rnn_units: The dimensionality of the rnn states if an RNN layer (default
        LSTM) is used.
      image_input_label: Label of image input in observation dictionary.
      final_layer_init: Final layer kernel initialization method.
      **kwargs: Arguments for creating image processing layers.
    """
    # Set some internal class variables.
    self._use_rnn = use_rnn
    self._rnn_units = rnn_units
    if isinstance(image_input_label, str):
      self._image_input_labels = [image_input_label]
    else:
      self._image_input_labels = image_input_label
    self.reset()

    inputs = self._create_vision_input_layers()
    outputs = []

    vision_outputs = []
    for vision_input in inputs:
      vision_outputs.append(
          self._create_vision_processing_layers(x=vision_input, **kwargs))
    vision_output = tf.keras.layers.concatenate(vision_outputs)

    if self._use_rnn:
      vision_output, states = self._create_rnn_layers(vision_output, inputs)
      outputs.extend(states)

    # Add other sensor observations.
    other_input = self._create_other_input_layer()
    if other_input is None:
      x = vision_output
    else:
      inputs.append(other_input)
      x = tf.keras.layers.concatenate([vision_output, other_input])

    # Final fully connected layers.
    for fc_layer_size in fc_layer_sizes:
      x = tf.keras.layers.Dense(fc_layer_size, activation="tanh")(x)
    action_output = tf.keras.layers.Dense(
        self._ac_dim, activation="tanh", kernel_initializer=final_layer_init)(
            x)
    outputs.append(action_output)

    self.model = tf.keras.models.Model(inputs=inputs, outputs=outputs)

  def reset(self) -> None:
    """Resets the policy's internal state (default LSTM)."""
    lstm_h_state = np.zeros(shape=(1, self._rnn_units), dtype="float")
    lstm_c_state = np.zeros(shape=(1, self._rnn_units), dtype="float")
    self._rnn_state = [lstm_h_state, lstm_c_state]

  def act(
      self, ob: Union[np.ndarray, Dict[str, np.ndarray]]
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

    if self._use_rnn:
      inputs.extend(self._rnn_state)

    if self._other_ob_dim > 0:
      other_ob = ob.copy()
      for image_label in self._image_input_labels:
        del other_ob[image_label]

      # Flatten other observations.
      other_input = utils.flatten(self._other_ob_space, other_ob)
      inputs.append(np.array([other_input]))

    # Run model.
    output = self.model(inputs)

    # Parse model output.
    if self._use_rnn:
      num_state_objects = len(self._rnn_state)
      self._rnn_state = [output[i].numpy() for i in range(num_state_objects)]
      output = output[num_state_objects:]
    actions = output[0].numpy()

    # Convert computed actions into desired action space dimensions.
    actions = utils.unflatten(self._ac_space, actions)
    return actions
