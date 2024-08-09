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

"""A keras layer for encoding image into patches."""

from typing import Tuple

import tensorflow as tf


class ImageEncoder(tf.keras.layers.Layer):
  """Keras layer for encoding image into patches."""

  def __init__(self,
               patch_height: int,
               patch_width: int,
               stride_height: int,
               stride_width: int,
               normalize_positions: bool = True) -> None:
    """Initializes Keras layer for encoding image into patches.

    Args:
      patch_height: Height of image patch for encoding.
      patch_width: Width of image patch for encoding.
      stride_height: Stride (shift) height for consecutive image patches.
      stride_width: Stride (shift) width for consecutive image patches.
      normalize_positions: True to normalize patch center positions.
    """
    super().__init__()
    self._patch_height = patch_height
    self._patch_width = patch_width
    self._stride_height = stride_height
    self._stride_width = stride_width
    self._normalize_positions = normalize_positions

  def call(self, images: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    batch_shape, image_height, image_width, channels = images.shape
    if batch_shape is None:
      batch_shape = tf.shape(images)[0]
    patches = tf.image.extract_patches(
        images,
        sizes=[1, self._patch_height, self._patch_width, 1],
        strides=[1, self._stride_height, self._stride_width, 1],
        rates=[1, 1, 1, 1],
        padding='VALID')
    encoding = tf.reshape(
        patches,
        [batch_shape, -1, self._patch_height * self._patch_width * channels])
    pos_x = tf.range(self._patch_height // 2, image_height, self._stride_height)
    pos_y = tf.range(self._patch_width // 2, image_width, self._stride_width)
    if self._normalize_positions:
      pos_x /= image_height
      pos_y /= image_width
    x, y = tf.meshgrid(pos_x, pos_y)
    x = tf.transpose(x)
    y = tf.transpose(y)
    centers = tf.stack([x, y], axis=-1)
    centers = tf.reshape(centers, (-1, 2))
    centers = tf.tile(centers, (batch_shape, 1))
    centers = tf.reshape(centers, (batch_shape, -1, 2))
    centers = tf.cast(centers, 'float32')
    return encoding, centers
