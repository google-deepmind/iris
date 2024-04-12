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

"""Tests for linear_policy."""

from gym import spaces
from iris.policies import linear_policy
import numpy as np
from absl.testing import absltest


class LinearPolicyTest(absltest.TestCase):

  def test_policy_act(self):
    """Tests the act function for linear policy."""
    policy = linear_policy.LinearPolicy(
        ob_space=spaces.Box(low=-10, high=10, shape=(2,)),
        ac_space=spaces.Box(low=-10, high=10, shape=(1,)))
    policy.update_weights(new_weights=np.ones(2))
    act = policy.act(np.array([2, -1]))[0]
    self.assertEqual(act, 1)

  def test_policy_act_dict(self):
    """Tests the act function for linear policy with dict observation."""
    policy = linear_policy.LinearPolicy(
        ob_space=spaces.Dict({
            'sensor1': spaces.Box(low=-10, high=10, shape=(2,)),
            'sensor2': spaces.Box(low=-10, high=10, shape=(2,)),
        }),
        ac_space=spaces.Box(low=-10, high=10, shape=(1,)))
    policy.update_weights(new_weights=np.ones(4))
    act = policy.act({
        'sensor1': np.array([2, 2]),
        'sensor2': np.array([-1, -1])
    })[0]
    self.assertEqual(act, 1)


if __name__ == '__main__':
  absltest.main()
