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

"""A keras layer for positional encoding."""

from typing import Tuple

import tensorflow as tf


class PositionalEncoding(tf.keras.layers.Layer):
  """Keras layer for positional encoding."""

  def call(self,
           seq_len: int,
           encoding_dimension: int) -> Tuple[tf.Tensor, tf.Tensor]:
    num_freq = encoding_dimension // 2
    indices = tf.expand_dims(tf.range(seq_len), 0)
    indices = tf.tile(indices, [num_freq, 1])
    freq_fn = lambda k: 1.0/(10000 ** (2*k/encoding_dimension))
    freq = tf.keras.layers.Lambda(freq_fn)(tf.range(num_freq))
    freq = tf.expand_dims(freq, 1)
    freq = tf.tile(freq, [1, seq_len])
    args = tf.multiply(freq, tf.cast(indices, dtype=tf.float64))
    sin_enc = tf.math.sin(args)
    cos_enc = tf.math.sin(args)
    encoding = tf.keras.layers.Concatenate(axis=0)([sin_enc, cos_enc])
    encoding = tf.expand_dims(tf.transpose(encoding), 0)
    return encoding
