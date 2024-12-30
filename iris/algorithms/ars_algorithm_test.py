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

from iris import normalizer
from iris import worker_util
from iris.algorithms import ars_algorithm
import numpy as np
import tensorflow as tf
from absl.testing import absltest
from absl.testing import parameterized


class AlgorithmTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.parameters(
      (True, False),
      (False, False),
  )
  def test_ars_gradient(self, orthogonal_suggestions, quasirandom_suggestions):
    algo = ars_algorithm.AugmentedRandomSearch(
        num_suggestions=3,
        step_size=0.5,
        std=1.,
        top_percentage=1,
        orthogonal_suggestions=orthogonal_suggestions,
        quasirandom_suggestions=quasirandom_suggestions,
        random_seed=7)
    init_state = {'init_params': np.array([10., 10.])}
    algo.initialize(init_state)
    suggestions = algo.get_param_suggestions()
    self.assertLen(suggestions, 6)
    eval_results = [
        worker_util.EvaluationResult(np.array([10., 11.]), 10),  # pytype: disable=wrong-arg-types  # numpy-scalars
        worker_util.EvaluationResult(np.empty(0), 0),  # pytype: disable=wrong-arg-types  # numpy-scalars
        worker_util.EvaluationResult(np.array([10., 11.]), 10),  # pytype: disable=wrong-arg-types  # numpy-scalars
        worker_util.EvaluationResult(np.array([10., 9.]), -10),  # pytype: disable=wrong-arg-types  # numpy-scalars
        worker_util.EvaluationResult(np.array([10., 9.]), -10),  # pytype: disable=wrong-arg-types  # numpy-scalars
        worker_util.EvaluationResult(np.empty(0), 0),  # pytype: disable=wrong-arg-types  # numpy-scalars
    ]
    algo.process_evaluations(eval_results)
    np.testing.assert_array_equal(algo._opt_params, np.array([10, 11]))

  def test_ars_gradient_with_schedule(self):
    algo = ars_algorithm.AugmentedRandomSearch(
        num_suggestions=3,
        step_size=lambda x: x + 0.5,
        std=lambda x: x + 1.,
        top_percentage=1,
        random_seed=7)
    init_state = {'init_params': np.array([10., 10.])}
    algo.initialize(init_state)
    suggestions = algo.get_param_suggestions()
    self.assertLen(suggestions, 6)
    eval_results = [
        worker_util.EvaluationResult(np.array([10., 11.]), 10),  # pytype: disable=wrong-arg-types  # numpy-scalars
        worker_util.EvaluationResult(np.empty(0), 0),  # pytype: disable=wrong-arg-types  # numpy-scalars
        worker_util.EvaluationResult(np.array([10., 11.]), 10),  # pytype: disable=wrong-arg-types  # numpy-scalars
        worker_util.EvaluationResult(np.array([10., 9.]), -10),  # pytype: disable=wrong-arg-types  # numpy-scalars
        worker_util.EvaluationResult(np.array([10., 9.]), -10),  # pytype: disable=wrong-arg-types  # numpy-scalars
        worker_util.EvaluationResult(np.empty(0), 0),  # pytype: disable=wrong-arg-types  # numpy-scalars
    ]
    algo.process_evaluations(eval_results)
    np.testing.assert_array_equal(algo._opt_params, np.array([10, 11]))

  @parameterized.parameters(
      ({'mean': np.asarray([1., 2.]), 'std': np.asarray([3., 4.]), 'n': 5},),
      (None,),
  )
  def test_restore_state_from_checkpoint(self, expected_obs_norm_state):
    algo = ars_algorithm.AugmentedRandomSearch(
        num_suggestions=3,
        step_size=0.5,
        std=1.0,
        top_percentage=1,
        orthogonal_suggestions=True,
        quasirandom_suggestions=True,
        obs_norm_data_buffer=normalizer.MeanStdBuffer()
        if expected_obs_norm_state is not None
        else None,
        random_seed=7,
    )
    init_state = {'init_params': np.array([10.0, 10.0])}
    if expected_obs_norm_state:
      init_state['obs_norm_buffer_data'] = {
          'mean': np.asarray([0.0, 0.0]),
          'std': np.asarray([1.0, 1.0]),
          'n': 0,
      }
    algo.initialize(init_state)
    # self.assertIsNotNone(algo._obs_norm_data_buffer)
    with self.subTest('init-mean'):
      self.assertAllClose(np.array(algo._opt_params), init_state['init_params'])
    if (
        expected_obs_norm_state is not None
        and algo._obs_norm_data_buffer is not None
    ):
      with self.subTest('init-obs-mean'):
        self.assertAllClose(
            np.asarray(algo._obs_norm_data_buffer.data['mean']),
            np.asarray(init_state['obs_norm_buffer_data']['mean']),
        )
      with self.subTest('init-obs-n'):
        self.assertAllClose(
            np.asarray(algo._obs_norm_data_buffer.data['n']),
            np.asarray(init_state['obs_norm_buffer_data']['n']),
        )
      with self.subTest('init-obs-std'):
        self.assertAllClose(
            np.asarray(algo._obs_norm_data_buffer.data['std']),
            init_state['obs_norm_buffer_data']['std'],
        )

    expected_restore_state = {'params_to_eval': np.array([5.0, 6.0])}
    if expected_obs_norm_state is not None:
      expected_restore_state['obs_norm_state'] = expected_obs_norm_state
    algo.restore_state_from_checkpoint(expected_restore_state)

    self.assertAllClose(
        algo._opt_params, expected_restore_state['params_to_eval']
    )
    if (
        expected_obs_norm_state is not None
        and algo._obs_norm_data_buffer is not None
    ):
      std = expected_restore_state['obs_norm_state']['std']
      var = np.square(std)
      expected_unnorm_var = var * 4
      with self.subTest('restore-obs-mean'):
        self.assertAllClose(
            np.asarray(algo._obs_norm_data_buffer.data['mean']),
            np.asarray(expected_restore_state['obs_norm_state']['mean']),
        )
      with self.subTest('restore-obs-n'):
        self.assertAllClose(
            np.asarray(algo._obs_norm_data_buffer.data['n']),
            np.asarray(expected_restore_state['obs_norm_state']['n']),
        )
      with self.subTest('restore-obs-std'):
        self.assertAllClose(
            np.asarray(algo._obs_norm_data_buffer.data['unnorm_var']),
            expected_unnorm_var,
        )


if __name__ == '__main__':
  absltest.main()
