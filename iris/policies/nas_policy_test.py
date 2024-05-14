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

"""Tests for nas_policy."""
from absl import logging
from gym import spaces
from iris.policies import nas_policy
import numpy as np
import tensorflow as tf
from absl.testing import absltest
from absl.testing import parameterized


class NasPoliciesTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.named_parameters(('aggregate', 'aggregate'),
                                  ('independent', 'independent'),
                                  ('residual', 'residual'))
  def test_edge_sparsity_policy_act(self, edge_sample_mode):
    """Tests the act function for edge pruning policy."""
    obs_dim = 5
    ac_dim = 3
    policy = nas_policy.NumpyEdgeSparsityPolicy(
        ob_space=spaces.Box(low=-10, high=10, shape=(obs_dim,)),
        ac_space=spaces.Box(low=-10, high=10, shape=(ac_dim,)),
        hidden_layer_sizes=[8],
        hidden_layer_edge_num=[16, 16],
        edge_sample_mode=edge_sample_mode)
    logging.info(policy._edge_dict)
    example_obs = np.random.normal(size=obs_dim)
    act1 = policy.act(example_obs)

    old_weights = policy.get_weights()
    new_weights = 0.1 * old_weights
    policy.update_weights(new_weights)
    act2 = policy.act(example_obs)

    self.assertNotAllEqual(act1, act2)

    policy.init_topology()
    act3 = policy.act(example_obs)
    self.assertNotAllEqual(act2, act3)


if __name__ == '__main__':
  absltest.main()
