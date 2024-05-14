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

"""Keras implementation of spatial softmax layer."""

from typing import Any, Optional, Text
import tensorflow.compat.v1 as tf


@tf.keras.utils.register_keras_serializable(package='Custom')
class SpatialSoftmax(tf.keras.layers.Layer):
  """Computes the spatial softmax of a convolutional feature map.

  This is a Keras reimplementation of tf.contrib.layers.spatial_softmax.
  First computes the softmax over the spatial extent of each channel of a
  convolutional feature map. Then computes the expected 2D position of the
  points of maximal activation for each channel, resulting in a set of
  feature keypoints [i1, j1, ... iN, jN] for all N channels.

  Read more here:
  "Learning visual feature spaces for robotic manipulation with
  deep spatial autoencoders." Finn et al., http://arxiv.org/abs/1509.06113.

  Input shape:
    N-H-W-C or N-C-H-W tensor with shape: (batch_size, height, width, channel)`
    if data_format is 'channels_last', otherwise (batch_size, channel, height,
    width)

  Output shape:
    N-(2 * C) tensor: (batch_size, 2 * channel)
  """

  def __init__(self,
               temperature: Optional[float] = None,
               trainable: bool = True,
               data_format: Text = 'channels_last',
               **kwargs):
    """Initializes the layer.

    Args:
      temperature: The initial temperature. If None, the internal temperature
        variable will be one-initialized.
      trainable: Whether the temperature parameter can be trained.
      data_format: The input format of the feature map.
      **kwargs: Other arguments.

    Raises:
      ValueError: if the temperature parameter is zero or negative. Or if
      the data format is unrecogniazed.

    """
    super(SpatialSoftmax, self).__init__(**kwargs)
    if temperature is not None and temperature <= 0:
      raise ValueError('Temperature cannot be zero or negative.')
    self.temperature = self.add_weight(
        name='temperature',
        shape=(),
        initializer=tf.keras.initializers.Constant(temperature)
        if temperature is not None else 'ones',
        trainable=trainable)
    if data_format not in ('channels_first', 'channels_last'):
      raise ValueError('Data format {} is not recorgnized.'.format(data_format))
    self.data_format = data_format

  def get_config(self):
    base_config = super(SpatialSoftmax, self).get_config()
    base_config['data_format'] = self.data_format
    base_config['temperature'] = tf.keras.backend.eval(self.temperature)
    return base_config

  def compute_output_shape(self, input_shape):
    input_shape = tf.TensorShape(input_shape).as_list()
    if self.data_format == 'channels_last':
      num_channels = input_shape[3]
    else:
      num_channels = input_shape[1]
    return tf.TensorShape([input_shape[0], 2 * num_channels])

  def call(self, inputs: Any):
    """Computes the output from the feature map.

    Args:
      inputs: A TensorFlow tensor or equivalents. The batched input feature
        map(s).

    Returns:
      The activation cooridnates for all channels.
    """
    shape_tensor = tf.shape(inputs)
    static_shape = inputs.shape
    if self.data_format == 'channels_last':
      height, width, num_channels = shape_tensor[1], shape_tensor[
          2], static_shape[3]
    else:
      num_channels, height, width = static_shape[1], shape_tensor[
          2], shape_tensor[3]
    pos_x, pos_y = tf.meshgrid(
        tf.lin_space(-1., 1., num=height),
        tf.lin_space(-1., 1., num=width),
        indexing='ij')
    pos_x = tf.reshape(pos_x, [height * width])
    pos_y = tf.reshape(pos_y, [height * width])

    if self.data_format == 'channels_last':
      inputs = tf.transpose(inputs, [0, 3, 1, 2])
    inputs = tf.reshape(inputs, [-1, height * width])

    softmax_attention = tf.nn.softmax(inputs / self.temperature)
    expected_x = tf.reduce_sum(pos_x * softmax_attention, [1], keepdims=True)
    expected_y = tf.reduce_sum(pos_y * softmax_attention, [1], keepdims=True)
    expected_xy = tf.concat([expected_x, expected_y], 1)
    feature_keypoints = tf.reshape(
        expected_xy, [-1, tf.dimension_value(num_channels) * 2])
    feature_keypoints.set_shape([None, tf.dimension_value(num_channels) * 2])
    return feature_keypoints
