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

"""A keras layer for masking based attention."""

from typing import Callable
import tensorflow as tf


class FavorMaskingAttention(tf.keras.layers.Layer):
  """A keras layer for masking based attention.

  A layer that creates a representation of the RGB(D)-image using attention
  mechanism from https://arxiv.org/abs/2009.14794. It leverages Performer-ReLU
  (go/performer) attention module in order to bypass explicit materialization of
  the L x L attention tensor, where L is the number of patches (potentially even
  individual pixels). This reduces time complexity of the attention module from
  quadratic to linear in L and provides a gateway to processing high-resolution
  images, where explicitly calculating attention tensor is not feasible. The
  ranking procedure is adopted from https://arxiv.org/abs/2003.08165, where
  scores of patches are defined as sums of the entries of the corresponding
  column in the attention tensor. After ranking, top K tokens are preserved and
  the rest of them are masked by 0.
  """

  def __init__(
      self,
      kernel_transformation: Callable[..., tf.Tensor],
      top_k: int = 5) -> None:  # pytype: disable=annotation-type-mismatch
    """Initializes FavorMaskingAttention layer.

    Args:
      kernel_transformation: Transformation used to get finite kernel features.
      top_k: Number of top patches that will be chosen to "summarize" entire
        image.
    """
    super().__init__()
    self._kernel_transformation = kernel_transformation
    self._top_k = top_k

  def call(self,
           queries: tf.Tensor,
           keys: tf.Tensor,
           values: tf.Tensor) -> tf.Tensor:
    queries_prime = self._kernel_transformation(
        data=tf.expand_dims(queries, axis=2),
        is_query=True)
    queries_prime = tf.squeeze(queries_prime, axis=2)
    keys_prime = self._kernel_transformation(
        data=tf.expand_dims(keys, axis=2),
        is_query=False)
    keys_prime = tf.squeeze(keys_prime, axis=2)
    _, length, _ = queries_prime.shape
    all_ones = tf.ones([1, length])
    reduced_queries_prime = tf.matmul(all_ones, queries_prime)
    scores = tf.matmul(reduced_queries_prime, keys_prime, transpose_b=True)
    scores = tf.reshape(scores, (-1, length))
    sorted_idxs = tf.argsort(scores, axis=-1, direction='DESCENDING')
    cutoff = tf.gather(
        scores, sorted_idxs[:, self._top_k], axis=1, batch_dims=1)
    cond = scores > tf.expand_dims(cutoff, -1)
    return tf.where(tf.expand_dims(cond, -1),
                    values,
                    tf.zeros_like(values))
