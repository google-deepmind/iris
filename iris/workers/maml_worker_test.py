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
from iris.workers import simple_worker
import numpy as np

from absl.testing import absltest
from absl.testing import parameterized


class AdaptationOptimizersTest(parameterized.TestCase):
  """Tests adaptation optimizers."""

  def setUp(self):
    self.worker_obj = simple_worker.SimpleWorker(
        worker_id=0,
        initial_params=5.0 * np.ones(2),
        blackbox_function=lambda x: -1 * np.sum(x**2),
    )
    self.init_params = self.worker_obj._init_state['init_params']
    super().setUp()

  def test_multiple_eval(self):
    mean_val, results = maml_worker._multiple_eval(
        params_to_eval=self.init_params,
        num_evals=5,
        work_fn=self.worker_obj.work,
    )
    self.assertLen(results, 5)
    self.assertEqual(mean_val, np.mean([result.value for result in results]))

  def test_gradient_adaptation(self):
    num_iterations = 2
    num_iteration_suggestions = 10
    num_adapted_evals = 5
    adaptation = maml_worker.GradientAdaptation(
        num_iterations=num_iterations,
        num_iteration_suggestions=num_iteration_suggestions,
        num_adapted_evals=num_adapted_evals,
    )
    val, results = adaptation.run_adaptation(
        params_to_eval=self.init_params, work_fn=self.worker_obj.work
    )

    self.assertLen(
        results,
        2 * num_iterations * num_iteration_suggestions + num_adapted_evals,
    )

    self.assertEqual(
        val, np.mean([result.value for result in results[-num_adapted_evals:]])
    )

    meta_value = self.worker_obj.work(self.init_params).value
    self.assertGreaterEqual(val, meta_value)

  @parameterized.parameters(('batch',), ('average',))
  def test_hillclimb_adaptation(self, parallel_alg: str):
    num_iterations = 20
    num_iteration_suggestions = 2
    num_adapted_evals = 4
    num_meta_evals = 4
    adaptation = maml_worker.HillClimbAdaptation(
        parallel_alg=parallel_alg,  # pytype: disable=wrong-arg-types
        num_iterations=num_iterations,
        num_iteration_suggestions=num_iteration_suggestions,
        num_adapted_evals=num_adapted_evals,
        num_meta_evals=num_meta_evals,
    )

    meta_value = self.worker_obj.work(self.init_params).value

    val, results = adaptation.run_adaptation(
        params_to_eval=self.init_params, work_fn=self.worker_obj.work
    )
    self.assertLen(
        results,
        num_meta_evals
        + num_iterations * num_iteration_suggestions
        + num_adapted_evals,
    )
    self.assertEqual(
        val, np.mean([result.value for result in results[-num_adapted_evals:]])
    )
    self.assertGreaterEqual(val, meta_value)

    val, new_results = adaptation.run_adaptation(
        params_to_eval=self.init_params,
        work_fn=self.worker_obj.work,
        meta_value=meta_value,
    )
    self.assertEqual(
        val,
        np.mean([result.value for result in new_results[-num_adapted_evals:]]),
    )
    self.assertLen(
        new_results,
        num_iterations * num_iteration_suggestions + num_adapted_evals,
    )
    self.assertGreaterEqual(val, meta_value)

  @parameterized.parameters(('batch',), ('average',))
  def test_hillclimb_iteration_step(self, parallel_alg: str):
    num_iterations = 20
    num_iteration_suggestions = 2
    num_adapted_evals = 4
    num_meta_evals = 4
    adaptation = maml_worker.HillClimbAdaptation(
        parallel_alg=parallel_alg,  # pytype: disable=wrong-arg-types
        num_iterations=num_iterations,
        num_iteration_suggestions=num_iteration_suggestions,
        num_adapted_evals=num_adapted_evals,
        num_meta_evals=num_meta_evals,
    )

    if parallel_alg == 'batch':
      potential_best_params, potential_pivot_value, eval_results = (
          adaptation._batch_iteration_step(
              params_to_eval=self.init_params, work_fn=self.worker_obj.work
          )
      )
    elif parallel_alg == 'average':
      potential_best_params, potential_pivot_value, eval_results = (
          adaptation._average_iteration_step(
              params_to_eval=self.init_params, work_fn=self.worker_obj.work
          )
      )
    else:
      raise ValueError(f'Unknown parallel algorithm: {parallel_alg}')

    self.assertLen(eval_results, num_iteration_suggestions)
    self.assertEqual(
        potential_pivot_value, self.worker_obj.work(potential_best_params).value
    )


class MamlWorkerTest(parameterized.TestCase):

  @parameterized.parameters(
      (maml_worker.HillClimbAdaptation,), (maml_worker.GradientAdaptation,)
  )
  def test_maml_rl_worker(self, adaptation_cls):
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
        adaptation_constructor=adaptation_cls,
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
