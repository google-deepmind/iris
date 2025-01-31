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

from iris.algorithms import pes_algorithm
from iris.workers import worker_util
import numpy as np
from absl.testing import absltest
from absl.testing import parameterized


class AlgorithmTest(parameterized.TestCase):

  @parameterized.parameters(
      (True, False),
      (False, False),
  )
  def test_pes_gradient(self, orthogonal_suggestions, quasirandom_suggestions):
    algo = pes_algorithm.PersistentES(
        num_suggestions=3,
        step_size=0.5,
        std=1.,
        top_percentage=1,
        orthogonal_suggestions=orthogonal_suggestions,
        quasirandom_suggestions=quasirandom_suggestions,
        random_seed=7)
    init_state = {'init_params': np.array([10., 10.])}
    algo.initialize(init_state)
    eval_results = [
        worker_util.EvaluationResult(  # pytype: disable=wrong-arg-types  # numpy-scalars
            np.array([10., 11.]), 10, metrics={'current_step': 5}),
        worker_util.EvaluationResult(  # pytype: disable=wrong-arg-types  # numpy-scalars
            np.empty(0), 0, metrics={'current_step': 5}),
        worker_util.EvaluationResult(  # pytype: disable=wrong-arg-types  # numpy-scalars
            np.array([10., 11.]), 10, metrics={'current_step': 5}),
        worker_util.EvaluationResult(  # pytype: disable=wrong-arg-types  # numpy-scalars
            np.array([10., 9.]), -10, metrics={'current_step': 5}),
        worker_util.EvaluationResult(  # pytype: disable=wrong-arg-types  # numpy-scalars
            np.array([10., 9.]), -10, metrics={'current_step': 5}),
        worker_util.EvaluationResult(  # pytype: disable=wrong-arg-types  # numpy-scalars
            np.empty(0), 0, metrics={'current_step': 5}),
    ]
    algo.process_evaluations(eval_results)
    np.testing.assert_array_equal(algo._opt_params, np.array([10, 11]))


if __name__ == '__main__':
  absltest.main()
