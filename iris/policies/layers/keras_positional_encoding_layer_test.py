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

from iris.policies.layers import keras_positional_encoding_layer
from absl.testing import absltest


class PositionalEncodingTest(absltest.TestCase):

  def test_layer_output(self):
    """Tests the output of PositionalEncoding layer."""
    encoding = keras_positional_encoding_layer.PositionalEncoding()(7, 4)
    self.assertEqual(encoding.shape, (1, 7, 4))

if __name__ == "__main__":
  absltest.main()
