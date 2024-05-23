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

from flax import linen as nn
from gym import spaces
from iris.policies import jax_policy
from iris.policies import nn_policy
import numpy as np
from absl.testing import absltest


class JaxNet(nn.Module):
  """ResNetV1."""

  @nn.compact
  def __call__(self, x, train: bool = True):
    x = nn.tanh(nn.Dense(2, use_bias=False)(x))
    x = nn.tanh(nn.Dense(1, use_bias=False)(x))
    return x


class JaxPolicyTest(absltest.TestCase):

  def test_policy_act(self):
    """Tests the act function for jax neural network policy."""
    init_x = np.ones((1, 2), np.float32)
    policy = jax_policy.JaxPolicy(
        ob_space=spaces.Dict(
            {'a': spaces.Box(low=-10, high=10, shape=(2,))}),
        ac_space=spaces.Box(low=-10, high=10, shape=(1,)),
        model=JaxNet,
        init_x=init_x)
    policy.update_weights(new_weights=np.ones(6))
    act = policy.act({'a': np.array([[2, -1]])})
    np.testing.assert_array_almost_equal(act, [0.9], 1)

    # Comparing keras action output with Numpy NN policy output
    numpy_policy = nn_policy.FullyConnectedNeuralNetworkPolicy(
        ob_space=spaces.Box(low=-10, high=10, shape=(2,)),
        ac_space=spaces.Box(low=-10, high=10, shape=(1,)),
        hidden_layer_sizes=[2])
    numpy_policy.update_weights(new_weights=np.ones(6))
    numpy_act = numpy_policy.act(np.array([2, -1]))
    np.testing.assert_array_almost_equal(act, numpy_act)


if __name__ == '__main__':
  absltest.main()
