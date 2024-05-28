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

"""Policy class that computes action by running keras neural network."""

from typing import Sequence
from iris.policies import keras_policy
import tensorflow as tf


class KerasNNPolicy(keras_policy.KerasPolicy):
  """Policy class that computes action by running feed fwd neural network."""

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
      x = tf.keras.layers.Dense(
          layer_size,
          activation=activation,
          use_bias=use_bias,
          kernel_initializer=kernel_initializer,
      )(x)
    output_layer = tf.keras.layers.Dense(
        self._ac_dim,
        activation=activation,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
    )(x)
    self.model = tf.keras.models.Model(
        inputs=[input_layer], outputs=[output_layer]
    )

  # pytype: enable=signature-mismatch  # overriding-parameter-count-checks
