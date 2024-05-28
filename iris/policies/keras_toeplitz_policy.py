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

"""Policy class that computes action by running toeplitz network."""

from typing import Sequence
from iris.policies import keras_policy
import tensorflow as tf


class Toeplitz(tf.keras.layers.Layer):
  """Keras layer with weights structured as a Toeplitz matrix.

  This layer implements a structured weight matrix as described in the paper,
  "Structured Transforms for Small-Footprint Deep Learning"
  (link: https://arxiv.org/pdf/1510.01722v1.pdf). Thus, this layer effectively
  implements Toeplitz-based policy  proposed in "Structured Evolution with
  Compact Architectures for Scalable Policy Optimization"
  (link: https://arxiv.org/abs/1804.02395).

  A square n*n Toeplitz weight matrix is constructed from a trainable row vector
  (t_0, t_-1, ..., t_-(n-1)) and a trainable column vector
  (t_0, t_1, ..., t_(n-1)) as follows:

    A = |t_0      t_-1    ...   t_-(n-1) |
        |t_1      t_0     ...        .   |
        | .        .       .         .   |
        | .        .       .         .   |
        | .        .       .        t_-1 |
        |t_(n-1)  ...     t_1        t_0 |

  First entry in the row and column is shared (t_0).

  When layer input and output dimensions are different, then a rectangular
  weight matrix is required. To implement a rectangular weight matrix, the lower
  dimension vector is padded with zeros to match the other vector size.

  Below is an example with row size 3 and column size 5:

    row = [a b c]
    zero padded row = [a b c 0 0]
    col = [a d e f g]
    A = |a b c 0 0|
        |d a b c 0|
        |e d a b c|
        |f e d a b|
        |g f e d a|

  When 5D input vector is multiplied with this weight matrix, it produces 5D
  output vector which is cropped to the desired size of 3D.

  When input dimension is smaller than output, the input is paddedd with zeros
  before multiplication.
  """

  def __init__(
      self,
      units: int = 32,
      activation: str = "tanh",
      use_bias: bool = True,
      kernel_initializer: str = "random_normal",
  ) -> None:
    super().__init__()
    self._units = units
    self._activation = tf.keras.activations.get(activation)
    self._use_bias = use_bias
    self._kernel_initializer = kernel_initializer

  def build(self, input_shape: Sequence[int]) -> None:
    self._cross_weight = self.add_weight(
        shape=(1,), initializer=self._kernel_initializer, trainable=True
    )
    self._col = self.add_weight(
        shape=(input_shape[-1] - 1,),
        initializer=self._kernel_initializer,
        trainable=True,
    )
    self._row = self.add_weight(
        shape=(self._units - 1,),
        initializer=self._kernel_initializer,
        trainable=True,
    )

    if self._use_bias:
      self._b = self.add_weight(
          shape=(self._units,), initializer="random_normal", trainable=True
      )
    else:
      self._b = 0

  def call(self, inputs: tf.Tensor) -> tf.Tensor:
    input_shape = inputs.shape
    toeplitz_row_size = max(input_shape[-1], self._units)
    extended_col = tf.concat(
        [
            self._cross_weight,
            self._col,
            tf.zeros(toeplitz_row_size - input_shape[-1]),
        ],
        0,
    )
    extended_row = tf.concat(
        [
            self._cross_weight,
            self._row,
            tf.zeros(toeplitz_row_size - self._units),
        ],
        0,
    )
    weight_matrix = tf.linalg.LinearOperatorToeplitz(extended_col, extended_row)
    # TODO: Move Toeplitz weight matrix creation without zero
    # padding to the build function when TF supports rectangular Toeplitz
    # matrices
    zero_padding_shape = input_shape.as_list()
    zero_padding_shape[-1] = max(0, self._units - input_shape[-1])
    extended_inputs = tf.concat([inputs, tf.zeros(zero_padding_shape)], -1)
    extended_outputs = tf.matmul(extended_inputs, weight_matrix)
    output_shape = input_shape.as_list()
    output_shape[-1] = self._units
    outputs = tf.slice(extended_outputs, [0] * len(output_shape), output_shape)
    return self._activation(outputs + self._b)


class KerasToeplitzPolicy(keras_policy.KerasPolicy):
  """Policy class that computes action by running toeplitz network."""

  # pytype: disable=signature-mismatch  # overriding-parameter-count-checks
  def _build_model(
      self,
      hidden_layer_sizes: Sequence[int],
      activation: str = "tanh",
      use_bias: bool = False,
      kernel_initializer: str = "zeros",
  ) -> None:
    """Constructs a keras feed forward neural network model.

    Args:
      hidden_layer_sizes: List of hiden layer sizes.
      activation: Activation function for layers.
      use_bias: Whether to use bias in layers.
      kernel_initializer: Initializer for the weights matrix.
    """
    # Creates model.
    input_layer = tf.keras.layers.Input(
        shape=(self._ob_dim,), batch_size=1, dtype="float32", name="input"
    )
    x = input_layer
    for layer_size in hidden_layer_sizes:
      x = Toeplitz(
          layer_size,
          activation=activation,
          use_bias=use_bias,
          kernel_initializer=kernel_initializer,
      )(x)
    output_layer = Toeplitz(
        self._ac_dim,
        activation=activation,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
    )(x)
    self.model = tf.keras.models.Model(
        inputs=[input_layer], outputs=[output_layer]
    )

  # pytype: enable=signature-mismatch  # overriding-parameter-count-checks
