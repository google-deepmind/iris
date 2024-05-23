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

"""Tests for adaptation_optimizers."""
from iris import worker
from iris.maml import adaptation_optimizers
import numpy as np
from absl.testing import absltest
from absl.testing import parameterized


class AdaptationOptimizersTest(parameterized.TestCase):
  """Tests adaptation optimizers."""

  def setUp(self):
    self.worker_obj = worker.SimpleWorker(
        worker_id=0,
        initial_params=5.0 * np.ones(2),
        blackbox_function=lambda x: -1 * np.sum(x**2))
    self.init_params = self.worker_obj._init_state['init_params']
    super().setUp()

  def test_multiple_eval(self):
    mean_val, results = adaptation_optimizers.multiple_eval(
        params_to_eval=self.init_params,
        num_evals=5,
        work_fn=self.worker_obj.work)
    self.assertLen(results, 5)
    self.assertEqual(mean_val, np.mean([result.value for result in results]))

  def test_gradient_adaptation(self):
    num_iterations = 2
    num_iteration_suggestions = 10
    num_adapted_evals = 5
    adaptation = adaptation_optimizers.GradientAdaptation(
        num_iterations=num_iterations,
        num_iteration_suggestions=num_iteration_suggestions,
        num_adapted_evals=num_adapted_evals)
    val, results = adaptation.run_adaptation(
        params_to_eval=self.init_params, work_fn=self.worker_obj.work)

    self.assertLen(
        results,
        2 * num_iterations * num_iteration_suggestions + num_adapted_evals)

    self.assertEqual(
        val, np.mean([result.value for result in results[-num_adapted_evals:]]))

    meta_value = self.worker_obj.work(self.init_params).value
    self.assertGreaterEqual(val, meta_value)

  @parameterized.named_parameters(
      ('_batch', adaptation_optimizers.HillClimbAdaptationType.BATCH),
      ('_average', adaptation_optimizers.HillClimbAdaptationType.AVERAGE))
  def test_hillclimb_adaptation(self, parallel_alg):
    num_iterations = 20
    num_iteration_suggestions = 2
    num_adapted_evals = 4
    num_meta_evals = 4
    adaptation = adaptation_optimizers.HillClimbAdaptation(
        parallel_alg=parallel_alg,
        num_iterations=num_iterations,
        num_iteration_suggestions=num_iteration_suggestions,
        num_adapted_evals=num_adapted_evals,
        num_meta_evals=num_meta_evals)

    meta_value = self.worker_obj.work(self.init_params).value

    val, results = adaptation.run_adaptation(
        params_to_eval=self.init_params, work_fn=self.worker_obj.work)
    self.assertLen(
        results, num_meta_evals + num_iterations * num_iteration_suggestions +
        num_adapted_evals)
    self.assertEqual(
        val, np.mean([result.value for result in results[-num_adapted_evals:]]))
    self.assertGreaterEqual(val, meta_value)

    val, new_results = adaptation.run_adaptation(
        params_to_eval=self.init_params,
        work_fn=self.worker_obj.work,
        meta_value=meta_value)
    self.assertEqual(
        val,
        np.mean([result.value for result in new_results[-num_adapted_evals:]]))
    self.assertLen(
        new_results,
        num_iterations * num_iteration_suggestions + num_adapted_evals)
    self.assertGreaterEqual(val, meta_value)

  @parameterized.named_parameters(
      ('_batch', adaptation_optimizers.HillClimbAdaptationType.BATCH),
      ('_average', adaptation_optimizers.HillClimbAdaptationType.AVERAGE))
  def test_hillclimb_iteration_step(self, parallel_alg):
    num_iterations = 20
    num_iteration_suggestions = 2
    num_adapted_evals = 4
    num_meta_evals = 4
    adaptation = adaptation_optimizers.HillClimbAdaptation(
        parallel_alg=parallel_alg,
        num_iterations=num_iterations,
        num_iteration_suggestions=num_iteration_suggestions,
        num_adapted_evals=num_adapted_evals,
        num_meta_evals=num_meta_evals)

    if parallel_alg is adaptation_optimizers.HillClimbAdaptationType.BATCH:
      potential_best_params, potential_pivot_value, eval_results = (
          adaptation._batch_iteration_step(
              params_to_eval=self.init_params, work_fn=self.worker_obj.work))
    elif parallel_alg is adaptation_optimizers.HillClimbAdaptationType.AVERAGE:
      potential_best_params, potential_pivot_value, eval_results = (
          adaptation._average_iteration_step(
              params_to_eval=self.init_params, work_fn=self.worker_obj.work))

    self.assertLen(eval_results, num_iteration_suggestions)
    self.assertEqual(potential_pivot_value,
                     self.worker_obj.work(potential_best_params).value)


if __name__ == '__main__':
  absltest.main()
