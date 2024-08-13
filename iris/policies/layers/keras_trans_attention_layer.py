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

"""A keras layer for performer attention."""

from typing import Callable
import tensorflow as tf


class FavorTransAttention(tf.keras.layers.Layer):
  """A keras layer for FAVOR trans attention.

  A layer that leverages Performer-ReLU (go/performer) attention module in order
  to bypass explicit materialization of the L x L attention tensor, where L is
  the number of patches (potentially even individual pixels). This reduces time
  complexity of the attention module from quadratic to linear in L and provides
  a gateway to processing high-resolution images, where explicitly calculating
  attention tensor is not feasible. Performer attention is applied to the input
  sequence of tokens to transform it into an encoded sequence.
  """

  def __init__(
      self,
      kernel_transformation: Callable[..., tf.Tensor],) -> None:  # pytype: disable=annotation-type-mismatch
    """Initializes FavorTransAttention layer.

    Args:
      kernel_transformation: Transformation used to get finite kernel features.
    """
    super().__init__()
    self._kernel_transformation = kernel_transformation

  def call(self,
           queries: tf.Tensor,
           keys: tf.Tensor,
           values: tf.Tensor) -> tf.Tensor:

    # Pass queries and keys through a non-linear kernel transformation to get
    # Q' and K'
    queries_prime = self._kernel_transformation(
        data=tf.expand_dims(queries, axis=1),
        is_query=True)
    queries_prime = tf.squeeze(queries_prime, axis=1)
    keys_prime = self._kernel_transformation(
        data=tf.expand_dims(keys, axis=1),
        is_query=False)
    keys_prime = tf.squeeze(keys_prime, axis=1)
    b, l, _ = queries_prime.shape
    if b is None:
      b = tf.shape(queries_prime)[0]
    if l is None:
      l = tf.shape(queries_prime)[1]

    # For applying FAVOR attention, product of K' and value vector is multiplied
    # by Q' prime without having to materialize the attention matrix
    # A = Q'(K')^T
    # Multiply K' and value vector
    kvs = tf.einsum("blm,bld->bmd", keys_prime, values)  # bmd
    # Multiply Q' with previous result to get attention output, x
    x = tf.einsum("blm,bmd->bld", queries_prime, kvs)  # bld

    # For normalization, attention output, x is divided by x_norm. x_norm is
    # obtained similarly to x by replacing value vector with all ones.
    kvs_norm = tf.einsum("blm,bld->bmd", keys_prime, tf.ones(
        (b, l, 1)))  # bmd (d=1)
    x_norm = tf.einsum("blm,bmd->bld", queries_prime, kvs_norm)  # bld

    return x/x_norm
