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

from iris.policies.layers import keras_ranking_attention_layer
from lingvo.core import favor_attention as favor
import numpy as np
import tensorflow as tf
from absl.testing import absltest


class FavorRankingAttentionTest(absltest.TestCase):

  def test_layer_output(self):
    """Tests the output of RankingAttention layer."""
    query_layer = tf.keras.layers.Input(
        batch_input_shape=(2, 3, 4), dtype="float", name="query")
    key_layer = tf.keras.layers.Input(
        batch_input_shape=(2, 3, 4), dtype="float", name="keys")
    value_layer = tf.keras.layers.Input(
        batch_input_shape=(2, 3, 4), dtype="float", name="values")
    output_layer = keras_ranking_attention_layer.FavorRankingAttention(
        kernel_transformation=favor.relu_kernel_transformation,
        top_k=2)(query_layer, key_layer, value_layer)
    model = tf.keras.models.Model(
        inputs=[query_layer, key_layer, value_layer], outputs=[output_layer])
    queries = np.arange(2 * 3 * 4).reshape((2, 3, 4))
    top_values = model.predict((queries, queries, queries))
    self.assertEqual(top_values.shape, (2, 2, 4))
    true_values = np.arange(2 * 3 * 4).reshape((2, 3, 4))
    true_values = np.flip(true_values[:, 1:, :], 1)
    np.testing.assert_array_almost_equal(top_values, true_values, 1)


if __name__ == "__main__":
  absltest.main()
