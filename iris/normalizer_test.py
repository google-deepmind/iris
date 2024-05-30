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
from iris import normalizer
import numpy as np
from absl.testing import absltest


class BufferTest(absltest.TestCase):

  def test_meanstdbuffer(self):
    buffer = normalizer.MeanStdBuffer((1))
    buffer.push(np.asarray(10.0))
    buffer.push(np.asarray(11.0))

    new_buffer = normalizer.MeanStdBuffer((1))
    new_buffer.data = buffer.data

    self.assertEqual(new_buffer._std, buffer._std)
    self.assertEqual(new_buffer._data['n'], buffer._data['n'])
    self.assertEqual(new_buffer.mean, buffer.mean)


class NormalizerTest(absltest.TestCase):

  def test_no_normalizer(self):
    norm = normalizer.NoNormalizer(
        gym.spaces.Box(low=np.zeros(5), high=np.ones(5))
    )
    value = np.ones(5)
    norm_value = norm(value)
    np.testing.assert_array_equal(norm_value, value)

  def test_action_range_denormalizer(self):
    space = gym.spaces.Box(low=np.zeros(5), high=5 * np.ones(5))
    norm = normalizer.ActionRangeDenormalizer(space)
    value = np.array([1, 1, -1, -1, 1])
    norm_value = norm(value)
    np.testing.assert_array_equal(norm_value, np.array([5, 5, 0, 0, 5]))

    space = gym.spaces.Dict({
        'sensor1': gym.spaces.Box(low=np.zeros(5), high=5 * np.ones(5)),
        'sensor2': gym.spaces.Box(low=np.zeros(3), high=5 * np.ones(3)),
    })
    norm = normalizer.ActionRangeDenormalizer(space, ignored_keys=['sensor2'])
    value = {
        'sensor1': np.array([1, 1, -1, -1, 1]),
        'sensor2': np.array([1, 1, -1]),
    }
    norm_value = norm(value)
    np.testing.assert_array_equal(
        norm_value['sensor1'], np.array([5, 5, 0, 0, 5])
    )
    np.testing.assert_array_equal(norm_value['sensor2'], np.array([1, 1, -1]))

  def test_observation_range_normalizer(self):
    space = gym.spaces.Box(low=np.zeros(5), high=5 * np.ones(5))
    norm = normalizer.ObservationRangeNormalizer(space)
    value = np.array([5, 5, 0, 0, 5])
    norm_value = norm(value)
    np.testing.assert_array_equal(norm_value, np.array([1, 1, -1, -1, 1]))

    space = gym.spaces.Dict({
        'sensor1': gym.spaces.Box(low=np.zeros(5), high=5 * np.ones(5)),
        'sensor2': gym.spaces.Box(low=np.zeros(3), high=5 * np.ones(3)),
    })
    norm = normalizer.ObservationRangeNormalizer(
        space, ignored_keys=['sensor2']
    )
    value = {
        'sensor1': np.array([5, 5, 0, 0, 5]),
        'sensor2': np.array([5, 5, 0]),
    }
    norm_value = norm(value)
    np.testing.assert_array_equal(
        norm_value['sensor1'], np.array([1, 1, -1, -1, 1])
    )
    np.testing.assert_array_equal(norm_value['sensor2'], np.array([5, 5, 0]))

  def test_running_mean_std_normalizer(self):
    space = gym.spaces.Box(low=np.zeros(5), high=5 * np.ones(5))
    norm = normalizer.RunningMeanStdNormalizer(space)
    value = np.array([5, 5, 0, 0, 5])
    norm_value = norm(value)
    np.testing.assert_array_equal(norm_value, value)

    space = gym.spaces.Dict({
        'sensor1': gym.spaces.Box(low=np.zeros(5), high=5 * np.ones(5)),
        'sensor2': gym.spaces.Box(low=np.zeros(3), high=5 * np.ones(3)),
    })
    norm = normalizer.RunningMeanStdNormalizer(space, ignored_keys=['sensor2'])
    value = {
        'sensor1': np.array([5, 5, 0, 0, 5]),
        'sensor2': np.array([5, 5, 0]),
    }
    norm_value = norm(value)
    np.testing.assert_array_equal(norm_value['sensor1'], value['sensor1'])
    np.testing.assert_array_equal(norm_value['sensor2'], np.array([5, 5, 0]))

    norm.state = norm.buffer.state
    norm_value = norm(value)
    np.testing.assert_array_equal(
        norm_value['sensor1'], np.zeros_like(value['sensor1'])
    )
    np.testing.assert_array_equal(norm_value['sensor2'], np.array([5, 5, 0]))

    state = {
        'mean': value['sensor1'] / 2.0,
        'std': np.ones_like(value['sensor1']),
    }
    norm.state = state
    norm_value = norm(value)
    np.testing.assert_array_equal(norm_value['sensor1'], value['sensor1'] / 2.0)
    np.testing.assert_array_equal(norm_value['sensor2'], np.array([5, 5, 0]))

    np.testing.assert_array_equal(norm.buffer._data['mean'], value['sensor1'])
    self.assertEqual(norm.buffer._data['n'], 3)

    data = {
        'n': 1,
        'mean': value['sensor1'] / 2.0,
        'unnorm_var': np.zeros_like(value['sensor1']),
    }
    norm.buffer.merge(data)
    np.testing.assert_array_equal(
        norm.buffer._data['mean'], value['sensor1'] * (7 / 8)
    )
    self.assertEqual(norm.buffer._data['n'], 4)

  def test_mean_std_buffer_empty_merge(self):
    mean_std_buffer = normalizer.MeanStdBuffer()
    self.assertEqual(mean_std_buffer._data['n'], 0)
    mean_std_buffer.merge({'n': 0})
    self.assertEqual(mean_std_buffer._data['n'], 0)

  def test_mean_std_buffer_scalar(self):
    mean_std_buffer = normalizer.MeanStdBuffer((1))
    mean_std_buffer.push(np.asarray(10.0))
    self.assertEqual(mean_std_buffer._std, 1.0)  # First value is always 1.0.

    mean_std_buffer.push(np.asarray(11.0))
    # sqrt(11.0-10.0 / 2.0)
    np.testing.assert_almost_equal(mean_std_buffer._std, np.sqrt(0.5))

  def test_mean_std_buffer_reject_infinity_on_merge(self):
    mean_std_buffer = normalizer.MeanStdBuffer((1))
    mean_std_buffer.push(np.asarray(10.0))

    infinty_buffer = normalizer.MeanStdBuffer((1))
    infinty_buffer.push(np.asarray(np.inf))

    mean_std_buffer.merge(infinty_buffer.data)
    self.assertEqual(mean_std_buffer._data['n'], 1)  # Still only 1 value.


if __name__ == '__main__':
  absltest.main()
