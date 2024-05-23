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
from iris.policies import keras_cnn_policy
import numpy as np
from absl.testing import absltest


class KerasCNNPolicyTest(absltest.TestCase):

  def test_policy_act(self):
    """Tests the act function for keras CNN policy."""
    policy = keras_cnn_policy.KerasCNNPolicy(
        ob_space=spaces.Dict({
            'vision': spaces.Box(low=-10, high=10, shape=(2, 2, 1)),
            'sensor1': spaces.Box(low=-10, high=10, shape=(2,)),
            'sensor2': spaces.Box(low=-10, high=10, shape=(2,)),
        }),
        ac_space=spaces.Box(low=-10, high=10, shape=(5,)),
        conv_filter_sizes=[1],
        conv_kernel_sizes=[2],
        image_feature_length=2,
        fc_layer_sizes=[2],
        use_rnn=False)
    policy.reset()
    policy.update_weights(new_weights=np.ones(38))
    image = np.ones((2, 2, 1))
    act = policy.act({
        'vision': image,
        'sensor1': [-3, -3],
        'sensor2': [-3, -3],
    })
    np.testing.assert_array_almost_equal(act, np.ones((5)), 1)
    policy.update_weights(new_weights=np.zeros(38))
    act = policy.act({
        'vision': image,
        'sensor1': [-3, -3],
        'sensor2': [-3, -3],
    })
    np.testing.assert_array_almost_equal(act, np.zeros((5)), 1)

  def test_lstm_state(self):
    policy = keras_cnn_policy.KerasCNNPolicy(
        ob_space=spaces.Dict({
            'vision': spaces.Box(low=-10, high=10, shape=(2, 2, 1)),
            'sensor1': spaces.Box(low=-10, high=10, shape=(2,)),
            'sensor2': spaces.Box(low=-10, high=10, shape=(2,)),
        }),
        ac_space=spaces.Box(low=-10, high=10, shape=(5,)),
        conv_filter_sizes=[1],
        conv_kernel_sizes=[2],
        image_feature_length=2,
        fc_layer_sizes=[2],
        use_rnn=True,
        rnn_units=2)
    np.random.seed(seed=13)
    weights = np.random.uniform(
        low=-0.01, high=0.01, size=policy.get_weights().size)
    policy.reset()
    policy.update_weights(weights)
    observation = {
        'vision': np.ones((2, 2, 1)),
        'sensor1': [-3, -3],
        'sensor2': [-3, -3],
    }
    prev_h_state = np.zeros(shape=(1, 2), dtype='float')
    prev_c_state = np.zeros(shape=(1, 2), dtype='float')

    # Checks that the LSTM state changes although the observations are the same.
    for _ in range(5):
      policy.act(observation)
      rnn_state = policy._rnn_state
      np.testing.assert_raises(AssertionError,
                               np.testing.assert_array_almost_equal,
                               prev_h_state, rnn_state[0])  #  pytype: disable=unsupported-operands
      np.testing.assert_raises(AssertionError,
                               np.testing.assert_array_almost_equal,
                               prev_c_state, rnn_state[1])  #  pytype: disable=unsupported-operands
      prev_h_state = rnn_state[0]  #  pytype: disable=unsupported-operands
      prev_c_state = rnn_state[1]  #  pytype: disable=unsupported-operands

    # Checks that the LSTM state is reset to zero.
    policy.reset()
    new_rnn_state = policy._rnn_state
    np.testing.assert_array_almost_equal(new_rnn_state[0],  #  pytype: disable=unsupported-operands
                                         np.zeros(shape=(1, 2)))
    np.testing.assert_array_almost_equal(new_rnn_state[0],  #  pytype: disable=unsupported-operands
                                         np.zeros(shape=(1, 2)))


if __name__ == '__main__':
  absltest.main()
