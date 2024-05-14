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

"""Tests for keras_policy."""

from gym import spaces
from iris.policies import keras_nn_policy
from iris.policies import nn_policy
import numpy as np
from absl.testing import absltest


class KerasNNPolicyTest(absltest.TestCase):

  def test_policy_act(self):
    """Tests the act function for keras neural network policy."""
    policy = keras_nn_policy.KerasNNPolicy(
        ob_space=spaces.Box(low=-10, high=10, shape=(2,)),
        ac_space=spaces.Box(low=-10, high=10, shape=(1,)),
        hidden_layer_sizes=[2])
    policy.update_weights(new_weights=np.ones(6))
    keras_act = policy.act(np.array([2, -1]))
    np.testing.assert_array_almost_equal(keras_act, [0.9], 1)

    # Comparing keras action output with Numpy NN policy output
    numpy_policy = nn_policy.FullyConnectedNeuralNetworkPolicy(
        ob_space=spaces.Box(low=-10, high=10, shape=(2,)),
        ac_space=spaces.Box(low=-10, high=10, shape=(1,)),
        hidden_layer_sizes=[2])
    numpy_policy.update_weights(new_weights=np.ones(6))
    numpy_act = numpy_policy.act(np.array([2, -1]))
    np.testing.assert_array_almost_equal(keras_act, numpy_act)

  def test_policy_act_dict(self):
    """Tests act for keras NN policy with dict observation."""
    policy = keras_nn_policy.KerasNNPolicy(
        ob_space=spaces.Dict({
            'sensor1': spaces.Box(low=-10, high=10, shape=(2,)),
            'sensor2': spaces.Box(low=-10, high=10, shape=(2,))
        }),
        ac_space=spaces.Box(low=-10, high=10, shape=(1,)),
        hidden_layer_sizes=[2])
    policy.update_weights(new_weights=np.ones(10))
    act = policy.act({
        'sensor1': np.array([2, 2]),
        'sensor2': np.array([-1, -1])
    })
    np.testing.assert_array_almost_equal(act, [0.9], 1)


if __name__ == '__main__':
  absltest.main()
