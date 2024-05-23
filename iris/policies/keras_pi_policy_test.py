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

from gym import spaces
from iris.policies import keras_pi_policy
import numpy as np
from absl.testing import absltest


class KerasPIPolicyTest(absltest.TestCase):

  def test_policy_act(self):
    """Tests the act function for keras PI policy."""
    policy = keras_pi_policy.KerasPIPolicy(
        ob_space=spaces.Dict({
            'vision': spaces.Box(low=-10, high=10, shape=(2, 2, 1)),
            'sensor1': spaces.Box(low=-10, high=10, shape=(2,)),
            'sensor2': spaces.Box(low=-10, high=10, shape=(2,)),
        }),
        ac_space=spaces.Box(low=-10, high=10, shape=(5,)),
        state_dim=2,
        conv_filter_sizes=[2],
        conv_kernel_sizes=[2],
        image_feature_length=2,
        fc_layer_sizes=[2],
        h_fc_layer_sizes=[2],
        f_fc_layer_sizes=[2],
        g_fc_layer_sizes=[2],
        image_input_label='vision')
    policy.update_weights(np.ones(21))
    policy.update_representation_weights(np.ones(1001))
    image = np.ones((2, 2, 1))
    act = policy.act({
        'vision': image,
        'sensor1': [-3, -3],
        'sensor2': [-3, -3],
    })
    np.testing.assert_array_almost_equal(act, np.ones((5)), 1)
    policy.update_weights(np.zeros(21))
    policy.update_representation_weights(np.zeros(1001))
    act = policy.act({
        'vision': image,
        'sensor1': [-3, -3],
        'sensor2': [-3, -3],
    })
    np.testing.assert_array_almost_equal(act, np.zeros((5)), 1)


if __name__ == '__main__':
  absltest.main()
