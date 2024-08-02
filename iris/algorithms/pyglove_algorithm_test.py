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
from iris.algorithms import pyglove_algorithm
import numpy as np
import pyglove as pg
from absl.testing import absltest
from absl.testing import parameterized


def make_init_state():
  dna_spec = pg.template(pg.one_of(['a', 'b'])).dna_spec()
  return {'serialized_dna_spec': pg.to_json_str(dna_spec)}


def make_evaluation_results(suggestion_list):
  eval_results = []
  for suggestion in suggestion_list[:-1]:
    evaluation_result = worker_util.EvaluationResult(
        params_evaluated=suggestion['params_to_eval'],  # FYI Empty array.
        value=np.random.uniform(),
        metadata=suggestion['metadata'])
    eval_results.append(evaluation_result)
  eval_results.append(worker_util.EvaluationResult(np.empty(0), 0))
  return eval_results


class PygloveAlgorithmTest(parameterized.TestCase):

  def setUp(self):
    self.num_suggestions = 100
    self.random_seed = 7
    super().setUp()

  @parameterized.named_parameters(
      ('hill_climb', 'hill_climb'), ('neat', 'neat'),
      ('policy_gradient', 'policy_gradient'),
      ('random_search', 'random_search'),
      ('regularized_evolution', 'regularized_evolution'))
  def test_pyglove_algo_step(self, controller_str):
    algo = pyglove_algorithm.PyGloveAlgorithm(
        controller_str=controller_str,
        num_suggestions=self.num_suggestions,
        random_seed=self.random_seed)

    init_state = make_init_state()
    algo.initialize(init_state)

    suggestion_list = algo.get_param_suggestions(evaluate=False)
    eval_results = make_evaluation_results(suggestion_list)
    algo.process_evaluations(eval_results)

  @parameterized.named_parameters(('False', False), ('True', True))
  def test_multithreading(self, multithreading):
    algo = pyglove_algorithm.PyGloveAlgorithm(
        multithreading=multithreading,
        num_suggestions=self.num_suggestions,
        random_seed=self.random_seed)

    init_state = make_init_state()
    algo.initialize(init_state)

    suggestion_list = algo.get_param_suggestions(evaluate=False)
    eval_results = make_evaluation_results(suggestion_list)
    algo.process_evaluations(eval_results)


if __name__ == '__main__':
  absltest.main()
