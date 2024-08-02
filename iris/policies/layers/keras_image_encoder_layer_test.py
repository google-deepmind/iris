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

from iris.policies.layers import keras_image_encoder_layer
import numpy as np
import tensorflow as tf
from absl.testing import absltest


class ImageEncoderTest(absltest.TestCase):

  def test_layer_output(self):
    """Tests the output of ImageEncoder layer."""
    input_layer = tf.keras.layers.Input(
        batch_input_shape=(2, 5, 6, 2), dtype="float", name="input")
    output_layer = keras_image_encoder_layer.ImageEncoder(
        patch_height=2,
        patch_width=2,
        stride_height=1,
        stride_width=1)(input_layer)
    model = tf.keras.models.Model(inputs=[input_layer], outputs=[output_layer])
    images = np.arange(2*5*6*2).reshape((2, 5, 6, 2))
    encoding, centers = model.predict(images)[0]
    self.assertEqual(encoding.shape, (2, 20, 8))
    self.assertEqual(centers.shape, (2, 20, 2))


if __name__ == "__main__":
  absltest.main()
