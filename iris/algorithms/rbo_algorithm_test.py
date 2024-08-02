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

from iris import worker_util
from iris.algorithms import rbo_algorithm
import numpy as np
from absl.testing import absltest
from absl.testing import parameterized


class AlgorithmTest(parameterized.TestCase):

  def test_rbo_gradient(self):
    algo = rbo_algorithm.RBO(
        num_suggestions=4,
        step_size=0.5,
        std=1.,
        regularizer=0.01,
        regression_method='lasso',
        random_seed=7)
    init_state = {'init_params': np.array([10., 10.])}
    algo.initialize(init_state)
    eval_results = [
        worker_util.EvaluationResult(np.array([10., 11.]), 10),  # pytype: disable=wrong-arg-types  # numpy-scalars
        worker_util.EvaluationResult(np.array([10., 9.]), -10),  # pytype: disable=wrong-arg-types  # numpy-scalars
        worker_util.EvaluationResult(np.empty(0), 0),  # pytype: disable=wrong-arg-types  # numpy-scalars
        worker_util.EvaluationResult(np.array([10., 11.]), 10),  # pytype: disable=wrong-arg-types  # numpy-scalars
        worker_util.EvaluationResult(np.array([10., 11.]), 10),  # pytype: disable=wrong-arg-types  # numpy-scalars
        worker_util.EvaluationResult(np.array([10., 9.]), -10),  # pytype: disable=wrong-arg-types  # numpy-scalars
        worker_util.EvaluationResult(np.array([10., 9.]), -10),  # pytype: disable=wrong-arg-types  # numpy-scalars
        worker_util.EvaluationResult(np.empty(0), 0),  # pytype: disable=wrong-arg-types  # numpy-scalars
    ]
    algo.process_evaluations(eval_results)
    np.testing.assert_array_almost_equal(
        algo._opt_params, np.array([10., 10.]), decimal=3)

  @parameterized.parameters(
      ('lasso', False, False),
      ('ridge', False, False),
      ('lp', False, False),
      ('lasso', True, False),
      ('lasso', False, True),
      ('lasso', True, True),
  )
  def test_rbo_gradient_2(self, regression_method, orthogonal_suggestions,
                          quasirandom_suggestions):
    algo = rbo_algorithm.RBO(
        num_suggestions=4,
        step_size=0.5,
        std=1.,
        regularizer=0.01,
        orthogonal_suggestions=orthogonal_suggestions,
        quasirandom_suggestions=quasirandom_suggestions,
        regression_method=regression_method,
        random_seed=7)
    init_state = {'init_params': np.array([10., 10.])}
    algo.initialize(init_state)
    eval_results = [
        worker_util.EvaluationResult(np.array([10., 11.]), 10),  # pytype: disable=wrong-arg-types  # numpy-scalars
        worker_util.EvaluationResult(np.array([10., 9.]), -10),  # pytype: disable=wrong-arg-types  # numpy-scalars
        worker_util.EvaluationResult(np.empty(0), 0),  # pytype: disable=wrong-arg-types  # numpy-scalars
        worker_util.EvaluationResult(np.array([10., 11.]), 10),  # pytype: disable=wrong-arg-types  # numpy-scalars
        worker_util.EvaluationResult(np.array([10., 11.]), 10),  # pytype: disable=wrong-arg-types  # numpy-scalars
        worker_util.EvaluationResult(np.array([10., 9.]), -10),  # pytype: disable=wrong-arg-types  # numpy-scalars
        worker_util.EvaluationResult(np.array([10., 9.]), -10),  # pytype: disable=wrong-arg-types  # numpy-scalars
        worker_util.EvaluationResult(np.empty(0), 0),  # pytype: disable=wrong-arg-types  # numpy-scalars
    ]
    algo.process_evaluations(eval_results)
    np.testing.assert_equal(len(algo._opt_params), 2)


if __name__ == '__main__':
  absltest.main()
