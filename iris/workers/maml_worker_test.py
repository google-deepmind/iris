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
from iris import normalizer
from iris.policies import linear_policy
from iris.workers import maml_worker
from iris.workers import rl_worker
import numpy as np

from absl.testing import absltest
from absl.testing import parameterized


class MamlWorkerTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('_hillclimb', maml_worker.AdaptationType.HILLCLIMB),
      ('_gradient', maml_worker.AdaptationType.GRADIENT),
  )
  def test_maml_rl_worker(self, adaptation_type):
    env = gym.make(id='Pendulum-v0')
    policy = linear_policy.LinearPolicy(
        ob_space=env.observation_space, ac_space=env.action_space
    )
    observation_normalizer = normalizer.NoNormalizer(env.observation_space)
    action_denormalizer = normalizer.NoNormalizer(env.action_space)

    worker_constructor = functools.partial(
        rl_worker.RLWorker,
        env=env,
        policy=policy,
        observation_normalizer=observation_normalizer,
        action_denormalizer=action_denormalizer,
        rollout_length=500,
    )

    worker_obj = maml_worker.MAMLWorker(
        worker_constructor=worker_constructor,
        adaptation_type=adaptation_type,
        adaptation_kwargs={},
        worker_id=0,
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
    self.assertLessEqual(result1.value, 0)

    result2 = worker_obj.work(
        params_to_eval=np.ones(3),
        obs_norm_state={},
        update_obs_norm_buffer=False,
    )
    self.assertLessEqual(result2.value, 0)


if __name__ == '__main__':
  absltest.main()
