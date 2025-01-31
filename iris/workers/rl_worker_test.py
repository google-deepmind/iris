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
from iris.policies import base_policy
from iris.policies import linear_policy
from iris.workers import rl_worker
import numpy as np

from absl.testing import absltest
from absl.testing import parameterized


class RlWorkerTest(parameterized.TestCase):

  def test_rl_worker(self):
    env = gym.make(id='Pendulum-v0')
    policy = linear_policy.LinearPolicy(
        ob_space=env.observation_space, ac_space=env.action_space
    )
    observation_normalizer = normalizer.NoNormalizer(env.observation_space)
    action_denormalizer = normalizer.NoNormalizer(env.action_space)
    metrics_fn = lambda env, step_output: {'extra_metric': 1.0}

    def stats_fn(name, values):
      summed_values = sum(values)
      max_value = max(values)
      top_10_values = sorted(values)[-10:]
      last_10_values = values[-10:]
      return max_value, {
          'summed_' + name: summed_values,
          'max_' + name: max_value,
          'top_10_' + name: top_10_values,
          'last_10_' + name: last_10_values,
      }

    worker_obj = rl_worker.RLWorker(
        worker_id=0,
        env=env,
        policy=policy,
        observation_normalizer=observation_normalizer,
        action_denormalizer=action_denormalizer,
        rollout_length=200,
        metrics_fn=metrics_fn,
        stats_fn=stats_fn,
    )
    with self.assertLogs(level='INFO') as logs:
      result1 = worker_obj.work(
          params_to_eval=np.ones(3),
          obs_norm_state={},
          update_obs_norm_buffer=False,
          enable_logging=True,
      )
    self.assertIn('INFO:absl:Step:', logs.output[0])
    self.assertIn('INFO:absl:Total Reward:', logs.output[-1])
    self.assertIn('Num invalid rollouts: 0', logs.output[-1])
    self.assertLen(logs.output, 401)
    self.assertLessEqual(result1.value, 0)
    self.assertEqual(result1.metrics['extra_metric'], 1.0)
    self.assertLessEqual(result1.metrics['summed_extra_metric'], 200.0)

    result2 = worker_obj.work(
        params_to_eval=np.ones(3),
        obs_norm_state={},
        update_obs_norm_buffer=False,
    )
    self.assertLessEqual(result2.value, 0)

  @parameterized.parameters(
      (False, False, 1, 1, False),
      (False, True, 3, 1, False),
      (True, False, 3, 1, False),
      (True, True, 3, 3, False),
      (True, True, 5, 5, False),
      (True, True, 7, 5, True),
  )
  def test_rl_worker_using_retry_rollout(
      self,
      retry_rollout,
      retry_rollout_info_flag,
      max_rollouts,
      expected_rollouts,
      catch_rollout_error,
  ):

    class _RolloutEnv(gym.Wrapper):

      def __init__(self, env: gym.Env):
        super().__init__(env)
        self._retry_rollout_info_flag = retry_rollout_info_flag
        self._max_rollouts = max_rollouts
        self._num_rollouts = 0

      def step(self, action):
        obs, reward, done, info = super().step(action)
        if done:
          self._num_rollouts += 1
        if done and self._retry_rollout_info_flag:
          info[rl_worker._VALID_ROLLOUT] = (
              False if self._num_rollouts < self._max_rollouts else True
          )
        return obs, reward, done, info

    env = gym.make(id='Pendulum-v0')
    env = _RolloutEnv(env)
    policy = linear_policy.LinearPolicy(
        ob_space=env.observation_space, ac_space=env.action_space
    )
    observation_normalizer = normalizer.NoNormalizer(env.observation_space)
    action_denormalizer = normalizer.NoNormalizer(env.action_space)
    metrics_fn = lambda env, step_output: {'extra_metric': 1.0}

    worker_obj = rl_worker.RLWorker(
        worker_id=0,
        env=env,
        policy=policy,
        observation_normalizer=observation_normalizer,
        action_denormalizer=action_denormalizer,
        rollout_length=500,
        metrics_fn=metrics_fn,
        retry_rollout=retry_rollout,
        max_rollout_retries=5,
    )
    if catch_rollout_error:
      with self.assertRaises(rl_worker.RolloutRetryError):
        worker_obj.work(
            params_to_eval=np.ones(3),
            obs_norm_state={},
            update_obs_norm_buffer=False,
            enable_logging=True,
        )
    else:
      with self.assertLogs(level='INFO') as logs:
        result1 = worker_obj.work(
            params_to_eval=np.ones(3),
            obs_norm_state={},
            update_obs_norm_buffer=False,
            enable_logging=True,
        )
      self.assertIn('INFO:absl:Step:', logs.output[0])
      self.assertIn('INFO:absl:Total Reward:', logs.output[-1])
      self.assertIn(
          f'Num invalid rollouts: {expected_rollouts-1}', logs.output[-1]
      )
      self.assertLen(logs.output, 400 * expected_rollouts + 1)
      self.assertLessEqual(result1.value, 0)
      self.assertLessEqual(result1.metrics['extra_metric'], 500.0)
      self.assertEqual(env._num_rollouts, expected_rollouts)

      result2 = worker_obj.work(
          params_to_eval=np.ones(3),
          obs_norm_state={},
          update_obs_norm_buffer=False,
      )
      self.assertLessEqual(result2.value, 0)

  @parameterized.parameters((None, 0), (0, 0), (1, 1), (2, 2), (3, 3))
  def test_rl_worker_with_iteration(self, iteration, expected_iteration):
    env = gym.make(id='Pendulum-v0')
    policy = linear_policy.LinearPolicy(
        ob_space=env.observation_space, ac_space=env.action_space
    )
    observation_normalizer = normalizer.NoNormalizer(env.observation_space)
    action_denormalizer = normalizer.NoNormalizer(env.action_space)

    worker_obj = rl_worker.RLWorker(
        worker_id=0,
        env=env,
        policy=policy,
        observation_normalizer=observation_normalizer,
        action_denormalizer=action_denormalizer,
        rollout_length=2,
    )
    with self.assertLogs(level='INFO') as logs:
      result = worker_obj.work(
          params_to_eval=np.ones(3),
          obs_norm_state={},
          update_obs_norm_buffer=False,
          enable_logging=True,
          iteration=iteration,
      )
      self.assertIn('INFO:absl:Step:', logs.output[0])
      self.assertIn('INFO:absl:Total Reward:', logs.output[-1])
      self.assertLessEqual(result.value, 0)

    self.assertEqual(policy.get_iteration(), expected_iteration)

  @parameterized.parameters((None, 0), (0, 1), (1, 2), (2, 3), (3, 4))
  def test_rl_worker_with_iteration_with_dummy_policy(
      self, iteration, expected_iteration
  ):

    class _DummyPolicy(base_policy.BasePolicy):

      def __init__(self, ob_space: gym.Space, ac_space: gym.Space):
        super().__init__(ob_space, ac_space)
        self._weights = np.zeros(self._ac_dim * self._ob_dim, dtype=np.float64)

      def set_iteration(self, value: int | None):
        if value is None:
          return
        value += 1
        super().set_iteration(value)

      def act(
          self, ob: np.ndarray | dict[str, np.ndarray]
      ) -> np.ndarray | dict[str, np.ndarray]:
        return self._ac_space.sample()

    env = gym.make(id='Pendulum-v0')
    policy = _DummyPolicy(
        ob_space=env.observation_space, ac_space=env.action_space
    )
    observation_normalizer = normalizer.NoNormalizer(env.observation_space)
    action_denormalizer = normalizer.NoNormalizer(env.action_space)

    worker_obj = rl_worker.RLWorker(
        worker_id=0,
        env=env,
        policy=policy,
        observation_normalizer=observation_normalizer,
        action_denormalizer=action_denormalizer,
        rollout_length=2,
    )
    worker_obj.work(
        params_to_eval=np.ones(3),
        obs_norm_state={},
        update_obs_norm_buffer=False,
        enable_logging=True,
        iteration=iteration,
    )

    self.assertEqual(policy.get_iteration(), expected_iteration)


if __name__ == '__main__':
  absltest.main()
