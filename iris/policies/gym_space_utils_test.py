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

import gym
from iris.policies import gym_space_utils
import numpy as np
from absl.testing import absltest


SPACE_DICT = gym.spaces.Dict({"sensor_1": gym.spaces.Box(-1, 1, (2,)),
                              "sensor_2": gym.spaces.Box(-1, 1, (2,))})
SPACE_DICT_IC = gym.spaces.Dict({"sensor_1": gym.spaces.Box(-1, 1, (2,)),
                                 "sensor_2": gym.spaces.Box(-1, 1, (2,)),
                                 "in_command": gym.spaces.Box(-1, 1, (3,))})
SPACE_BOX = gym.spaces.Box(low=np.array([-1]*4), high=np.array([1]*4))
SPACE_BOX_IC = gym.spaces.Box(low=np.array([-1]*7), high=np.array([1]*7))
OB_DICT = {"sensor_1": np.array([1, 2]), "sensor_2": np.array([3, 4])}
OB_DICT_IC = {"sensor_1": np.array([1, 2]),
              "sensor_2": np.array([3, 4]),
              "in_command": np.array([5, 6, 7])}
OB_BOX = np.array([1, 2, 3, 4])
OB_BOX_IC = np.array([5, 6, 7, 1, 2, 3, 4])


class GymSpaceUtilsTest(absltest.TestCase):

  def test_filter_space(self):
    space = gym_space_utils.filter_space(SPACE_BOX, [2, 3])
    self.assertEqual(space.shape, (2,))
    space = gym_space_utils.filter_space(
        SPACE_DICT, ["sensor_1"])
    filtered_space = gym.spaces.Dict({"sensor_1": gym.spaces.Box(-1, 1, (2,))})
    self.assertEqual(space, filtered_space)

  def test_extend_space(self):
    space = gym_space_utils.extend_space(SPACE_BOX,
                                         "in_command",
                                         gym.spaces.Box(-1, 1, (3,)))
    self.assertEqual(space, SPACE_BOX_IC)
    space = gym_space_utils.extend_space(SPACE_DICT,
                                         "in_command",
                                         gym.spaces.Box(-1, 1, (3,)))
    self.assertEqual(space, SPACE_DICT_IC)

  def test_filter_sample(self):
    ob = gym_space_utils.filter_sample(np.array([1, 2]), [1])
    np.testing.assert_array_equal(ob, np.array([2]))
    ob = gym_space_utils.filter_sample(OB_DICT, ["sensor_2"])
    np.testing.assert_equal(ob, {"sensor_2": np.array([3, 4])})

  def test_extend_sample(self):
    ob = gym_space_utils.extend_sample(np.array([1, 2]),
                                       "in_command",
                                       np.array([3, 4]))
    np.testing.assert_array_equal(ob, np.array([3, 4, 1, 2]))
    ob = gym_space_utils.extend_sample(OB_DICT, "in_command",
                                       np.array([5, 6, 7]))
    np.testing.assert_equal(ob, OB_DICT_IC)

if __name__ == "__main__":
  absltest.main()
