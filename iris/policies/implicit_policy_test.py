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
from gym.spaces import utils
from iris.policies import implicit_policy
import numpy as np
from absl.testing import absltest


class ImplicitPolicyTest(absltest.TestCase):

  def test_policy_act(self):
    """Tests the act method for the implicit policy."""

    ob_space = spaces.Dict({
        'sensor1': spaces.Box(low=-10, high=10, shape=(2,)),
        'sensor2': spaces.Box(low=-10, high=10, shape=(2,)),
    })
    ac_space = spaces.Box(low=-10, high=10, shape=(3,))
    action_low = utils.flatten_space(ac_space).low
    action_high = utils.flatten_space(ac_space).high
    state_dim = len(utils.flatten_space(ob_space).low)
    action_dim = len(action_low)
    latent_dim = 30
    st_hidden_layer_sizes = [100, 100]
    ac_hidden_layer_sizes = [50, 50]
    state_tower = implicit_policy.NeuralNetworkTower(state_dim, latent_dim,
                                                     st_hidden_layer_sizes)
    action_tower = implicit_policy.NeuralNetworkTower(action_dim, latent_dim,
                                                      ac_hidden_layer_sizes)
    num_samples = 100
    energy = implicit_policy.NegatedDotProductEnergy(state_tower, action_tower)
    action_calculator = implicit_policy.MinEnergyActionCalculator(
        energy, action_low, action_high, num_samples)
    tested_implicit_policy = implicit_policy.ImplicitEBMPolicy(
        ob_space, ac_space, action_calculator)
    action = tested_implicit_policy.act({
        'sensor1': np.array([2, 2]),
        'sensor2': np.array([-1, -1])
    })
    action = utils.flatten(ac_space, action)
    first_bound = action < action_high
    second_bound = action > action_low

    for x in first_bound:
      self.assertEqual(x, True)
    for x in second_bound:
      self.assertEqual(x, True)

  def test_policy_update_get_weights(self):
    """Tests the update_weights / get_weights method for the implicit policy."""
    ob_space = spaces.Dict({
        'sensor1': spaces.Box(low=-3, high=2, shape=(3,)),
        'sensor2': spaces.Box(low=-7, high=5, shape=(4, 5)),
    })
    ac_space = spaces.Box(low=15, high=16, shape=(10,))

    action_low = utils.flatten_space(ac_space).low
    action_high = utils.flatten_space(ac_space).high
    state_dim = len(utils.flatten_space(ob_space).low)
    action_dim = len(action_low)
    latent_dim = 54
    st_hidden_layer_sizes = [45, 55]
    ac_hidden_layer_sizes = [89, 24]

    state_tower = implicit_policy.NeuralNetworkTower(state_dim, latent_dim,
                                                     st_hidden_layer_sizes)
    action_tower = implicit_policy.NeuralNetworkTower(action_dim, latent_dim,
                                                      ac_hidden_layer_sizes)
    num_samples = 100
    energy = implicit_policy.NegatedDotProductEnergy(state_tower, action_tower)
    action_calculator = implicit_policy.MinEnergyActionCalculator(
        energy, action_low, action_high, num_samples)
    tested_implicit_policy = implicit_policy.ImplicitEBMPolicy(
        ob_space, ac_space, action_calculator)

    new_weights = np.random.normal(
        size=(len(tested_implicit_policy.get_weights())))
    tested_implicit_policy.update_weights(new_weights)
    policy_weights = tested_implicit_policy.get_weights()
    self.assertEqual(len(new_weights), len(policy_weights))
    for i in range(len(new_weights)):
      self.assertEqual(new_weights[i], policy_weights[i])


if __name__ == '__main__':
  absltest.main()
