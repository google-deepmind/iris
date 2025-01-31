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
from iris.policies import nas_policy
from iris.workers import pyglove_rl_worker
import pyglove as pg

from absl.testing import absltest


class PygloveRlWorkerTest(absltest.TestCase):

  def test_pyglove_rl_worker(self):
    env = gym.make(id='Pendulum-v0')
    policy = nas_policy.NumpyEdgeSparsityPolicy(
        ob_space=env.observation_space,
        ac_space=env.action_space,
        hidden_layer_sizes=[16],
        hidden_layer_edge_num=[3, 3],
    )

    weights = policy.get_weights()
    dna_1 = pg.random_dna(policy.dna_spec)
    serialized_dna_1 = pg.to_json_str(dna_1)

    observation_normalizer = normalizer.NoNormalizer(env.observation_space)
    action_denormalizer = normalizer.NoNormalizer(env.action_space)
    worker_obj = pyglove_rl_worker.PyGloveRLWorker(
        policy=policy,
        worker_id=0,
        env=env,
        observation_normalizer=observation_normalizer,
        action_denormalizer=action_denormalizer,
        rollout_length=500,
    )
    with self.assertLogs(level='INFO') as logs:
      result1 = worker_obj.work(
          metadata=serialized_dna_1,
          params_to_eval=2.0 * weights,
          obs_norm_state={},
          update_obs_norm_buffer=False,
          enable_logging=True,
      )
    self.assertIn('INFO:absl:Step:', logs.output[0])
    self.assertIn('INFO:absl:Total Reward:', logs.output[-1])
    self.assertLen(logs.output, 201)
    self.assertLessEqual(result1.value, 0)

    dna_2 = pg.random_dna(policy.dna_spec)
    serialized_dna_2 = pg.to_json_str(dna_2)
    result2 = worker_obj.work(
        metadata=serialized_dna_2,
        params_to_eval=-1.0 * weights,
        obs_norm_state={},
        update_obs_norm_buffer=False,
    )
    self.assertLessEqual(result2.value, 0)


if __name__ == '__main__':
  absltest.main()
