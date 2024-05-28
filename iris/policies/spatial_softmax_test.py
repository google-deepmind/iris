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

# pytype: disable=attribute-error
from iris.policies import spatial_softmax
import tensorflow as tf2
import tensorflow.compat.v1 as tf
from absl.testing import absltest

# TODO: Remove this try/except once import is fixed.
try:
  test_utils = tf2._keras_internal.testing_infra.test_utils  # pylint:disable=protected-access
except AttributeError:
  test_utils = None

_INPUT_SHAPE = (16, 32, 32, 128)
_TEMPERATURE = 2.5


@absltest.skipIf(test_utils is None, 'test_utils not available')
class SpatialSoftmaxTest(tf.test.TestCase):

  def test_with_default(self):
    out = test_utils.layer_test(
        spatial_softmax.SpatialSoftmax, input_shape=_INPUT_SHAPE
    )
    self.assertAllEqual(out.shape, (_INPUT_SHAPE[0], _INPUT_SHAPE[3] * 2))

  def test_with_preset_temperature(self):
    test_utils.layer_test(
        spatial_softmax.SpatialSoftmax,
        kwargs={'temperature': _TEMPERATURE},
        input_shape=_INPUT_SHAPE,
    )

  def test_get_weights(self):
    layer = spatial_softmax.SpatialSoftmax(temperature=_TEMPERATURE)
    self.assertEqual(layer.get_weights(), [_TEMPERATURE])

  def test_with_channels_first(self):
    input_shape = (16, 128, 32, 32)
    out = test_utils.layer_test(
        spatial_softmax.SpatialSoftmax,
        kwargs={'data_format': 'channels_first'},
        input_shape=input_shape,
    )
    self.assertAllEqual(out.shape, (input_shape[0], input_shape[1] * 2))


if __name__ == '__main__':
  tf.test.main()
