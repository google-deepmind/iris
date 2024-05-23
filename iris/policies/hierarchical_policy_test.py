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

import functools

import gym
from iris.policies import hierarchical_policy
from iris.policies import keras_cnn_policy
from iris.policies import nn_policy
import numpy as np

from absl.testing import absltest
from absl.testing import parameterized


def merge_two_dicts(x, y):
  """Given two dicts, merge them into a new dict as a shallow copy."""
  z = dict(x)
  z.update(y)
  return z

DEFAULT_POLICY_PARAMS = {
    "ob_space": gym.spaces.Box(-1, 1, (2,)),
    "ac_space": gym.spaces.Box(-1, 1, (1,)),
}

DEFAULT_POLICY_PARAMS_DICT = {
    "ob_space": gym.spaces.Dict({
        "sensor_1": gym.spaces.Box(-1, 1, (1,)),
        "sensor_2": gym.spaces.Box(-1, 1, (1,))
    }),
    "ac_space": gym.spaces.Box(-1, 1, (1,)),
}

DEFAULT_POLICY_PARAMS_VISION = {
    "ob_space": gym.spaces.Dict({
        "vision": gym.spaces.Box(-1, 1, (3, 3, 1)),
        "sensor_1": gym.spaces.Box(-1, 1, (1,)),
        "sensor_2": gym.spaces.Box(-1, 1, (1,))
    }),
    "ac_space": gym.spaces.Box(-1, 1, (1,)),
}

HRL_POLICY_PARAMS = merge_two_dicts(DEFAULT_POLICY_PARAMS, {
    "level_params": [
        {"in_command_dim": 0,
         "out_command_dim": 5,
         "selected_observations": [0, 1],
         "fixed_timescale": None,
         "timescale_range": (2, 5),
         "policy": functools.partial(
             nn_policy.FullyConnectedNeuralNetworkPolicy,
             hidden_layer_sizes=(2,),
             activation="clip")},
        {"in_command_dim": 4,
         "selected_observations": [1],
         "fixed_timescale": 1,
         "timescale_range": (0, 0),
         "policy": functools.partial(
             nn_policy.FullyConnectedNeuralNetworkPolicy,
             hidden_layer_sizes=(2,),
             activation="clip")}]
})

HRL_POLICY_PARAMS_DICT = merge_two_dicts(
    DEFAULT_POLICY_PARAMS_DICT, {
        "level_params": [{
            "in_command_dim": 0,
            "out_command_dim": 5,
            "selected_observations": ["sensor_1", "sensor_2"],
            "fixed_timescale": None,
            "timescale_range": (2, 5),
            "policy":
                functools.partial(
                    nn_policy.FullyConnectedNeuralNetworkPolicy,
                    hidden_layer_sizes=(2,),
                    activation="clip")
        }, {
            "in_command_dim": 4,
            "selected_observations": ["sensor_2"],
            "fixed_timescale": 1,
            "timescale_range": (0, 0),
            "policy":
                functools.partial(
                    nn_policy.FullyConnectedNeuralNetworkPolicy,
                    hidden_layer_sizes=(2,),
                    activation="clip")
        }]
    })

HRL_POLICY_PARAMS_VISION = merge_two_dicts(
    DEFAULT_POLICY_PARAMS_VISION, {
        "level_params": [{
            "in_command_dim": 0,
            "out_command_dim": 5,
            "selected_observations": ["vision", "sensor_1", "sensor_2"],
            "fixed_timescale": None,
            "timescale_range": (2, 5),
            "policy": functools.partial(
                keras_cnn_policy.KerasCNNPolicy,
                conv_filter_sizes=(2, 2),
                conv_kernel_sizes=(2, 2),
                image_feature_length=3,
                fc_layer_sizes=(),)
        }, {
            "in_command_dim": 4,
            "out_command_dim": 1,
            "selected_observations": ["sensor_1", "sensor_2"],
            "fixed_timescale": 1,
            "timescale_range": (0, 0),
            "policy":
                functools.partial(
                    nn_policy.FullyConnectedNeuralNetworkPolicy,
                    hidden_layer_sizes=(2,),
                    activation="clip")
        }]
    })


class HierarchicalPolicyTest(parameterized.TestCase):

  @parameterized.parameters((HRL_POLICY_PARAMS, False),
                            (HRL_POLICY_PARAMS_DICT, True))
  def test_hierarchical_policy_act(self, policy_params, is_ob_dict):
    """Tests the act function for hierarchical policies."""
    policy = hierarchical_policy.HierarchicalPolicy(**policy_params)
    policy.update_weights(new_weights=np.ones(26))
    ob = np.array([5, -4])
    if is_ob_dict:
      ob = {"sensor_1": np.array([5]), "sensor_2": np.array([-4])}
    act = policy.act(ob)
    # Latent command is [1, 1, 1, 1] and low level output is
    # 2 * sum([1, 1, 1, 1, -4]) = 0
    self.assertAlmostEqual(act, 0, places=2)
    # Check that latent command remains constant until high level activates
    interval = policy.levels[0]._act_after_steps
    ob = np.array([5, -4.5])
    if is_ob_dict:
      ob = {"sensor_1": np.array([5]), "sensor_2": np.array([-4.5])}
    for _ in range(interval):
      act = policy.act(ob)
      # Latent command is still [1, 1, 1, 1] and low level output is
      # 2 * sum([1, 1, 1, 1, -4.5]) = -1
      self.assertAlmostEqual(act, -1, places=2)
    ob = np.array([-0.5, 0.5])
    if is_ob_dict:
      ob = {"sensor_1": np.array([-0.5]), "sensor_2": np.array([0.5])}
    act = policy.act(ob)
    # Latent command has now changed to [0, 0, 0, 0] and low level output is
    # 2 * sum([0, 0, 0, 0, 0.5]) = 1
    self.assertAlmostEqual(act, 1, places=2)

  def test_vision_hierarchical_policy_act(self):
    """Tests the act function for hierarchical policy with vision input."""
    policy = hierarchical_policy.HierarchicalPolicy(**HRL_POLICY_PARAMS_VISION)
    policy.update_weights(new_weights=np.ones(81))
    ob = {
        "vision": np.ones((3, 3, 1)),
        "sensor_1": np.array([5]),
        "sensor_2": np.array([-4])
    }
    act = policy.act(ob)[0]
    self.assertAlmostEqual(act, 1, places=2)

if __name__ == "__main__":
  absltest.main()
