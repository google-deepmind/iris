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

from iris.workers import worker_util
import numpy as np

from absl.testing import absltest


class WorkerUtilTest(absltest.TestCase):

  def test_merge(self):
    result1 = worker_util.EvaluationResult(
        params_evaluated=np.zeros(6),
        value=np.float64(5.0),
        obs_norm_buffer_data={
            'n': 5,
            'mean': np.zeros(7),
            'unnorm_var': np.ones(7),
        },
        metadata='',
        metrics={'extra_metric': np.float64(1.0)},
    )
    result2 = worker_util.EvaluationResult(
        params_evaluated=np.zeros(6),
        value=np.float64(10.0),
        obs_norm_buffer_data={
            'n': 10,
            'mean': np.ones(7),
            'unnorm_var': 2 * np.ones(7),
        },
        metadata='',
        metrics={'extra_metric': np.float64(3.0)},
    )
    merged_result = worker_util.merge_eval_results([result1, result2])
    mean_value = np.mean([result1.value, result2.value])
    buffer_data_mean = 10 * np.ones(7) / 15.0
    self.assertEqual(merged_result.value, mean_value)
    self.assertIsNotNone(merged_result.obs_norm_buffer_data)
    self.assertEqual(merged_result.obs_norm_buffer_data['n'], 15)
    np.testing.assert_array_equal(
        merged_result.obs_norm_buffer_data['mean'], buffer_data_mean
    )
    self.assertEqual(merged_result.metrics['extra_metric'], 2.0)

  def test_merge_empty(self):
    with self.assertRaisesRegex(ValueError, '(?=.*empty)(?=.*merge)'):
      worker_util.merge_eval_results([])


if __name__ == '__main__':
  absltest.main()
