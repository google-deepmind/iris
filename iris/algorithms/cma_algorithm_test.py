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

from iris.algorithms import cma_algorithm
from iris.workers import worker_util
import numpy as np
from absl.testing import absltest

_TRUE_OPTIMAL = (-1, -1)


def test_fn(x):
  """A simple quadrtic function to be maximized."""
  return -np.linalg.norm(x - np.array(_TRUE_OPTIMAL))


class CMAAlgorithmTest(absltest.TestCase):

  def setUp(self):
    super(CMAAlgorithmTest, self).setUp()
    self.algo = cma_algorithm.CMAES(
        num_suggestions=10,
        std=0.5,
        bounds=(-3, 3),
        random_seed=7)
    init_state = {'init_params': np.array([0., 0.])}
    self.algo.initialize(init_state)

  def test_get_param_suggestions(self):
    eval_suggestion_list = self.algo.get_param_suggestions(evaluate=True)
    for eval_suggestion in eval_suggestion_list:
      np.testing.assert_almost_equal(eval_suggestion['params_to_eval'],
                                     np.array([0., 0.]))

  def test_cma_optimization(self):
    for i in range(100):
      suggestion_list = self.algo.get_param_suggestions(evaluate=False)

      eval_results = []
      for suggestion in suggestion_list:
        eval_results.append(
            worker_util.EvaluationResult(
                np.array(suggestion['params_to_eval']),
                test_fn(np.array(suggestion['params_to_eval']))))
      if i%10 == 0:
        eval_results[0] = worker_util.EvaluationResult(np.empty(0), 0)  # pytype: disable=wrong-arg-types  # numpy-scalars
      self.algo.process_evaluations(eval_results)
    np.testing.assert_almost_equal(self.algo._opt_params, _TRUE_OPTIMAL)


if __name__ == '__main__':
  absltest.main()
