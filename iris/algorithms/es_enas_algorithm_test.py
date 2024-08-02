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

# pytype: disable=attribute-error
from gym import spaces
from iris import worker_util
from iris.algorithms import es_enas_algorithm
from iris.policies import nas_policy
import numpy as np
import pyglove as pg
from absl.testing import absltest
from absl.testing import parameterized


def make_init_state():
  policy = nas_policy.NumpyEdgeSparsityPolicy(
      ob_space=spaces.Box(low=-10, high=10, shape=(5,)),
      ac_space=spaces.Box(low=-10, high=10, shape=(3,)),
      hidden_layer_sizes=[16],
      hidden_layer_edge_num=[3, 3])
  weights = policy.get_weights()
  return {
      'init_params': weights,
      'serialized_dna_spec': pg.to_json_str(policy.dna_spec)
  }


def make_evaluation_results(suggestion_list):
  eval_results = []
  for suggestion in suggestion_list[:-1]:
    evaluation_result = worker_util.EvaluationResult(
        params_evaluated=suggestion['params_to_eval'],
        value=np.random.uniform(),
        metadata=suggestion['metadata'])
    eval_results.append(evaluation_result)
  eval_results.append(worker_util.EvaluationResult(np.empty(0), 0))  # pytype: disable=wrong-arg-types  # numpy-scalars
  return eval_results


class EsEnasAlgorithmTest(parameterized.TestCase):

  def setUp(self):
    self.dna_proposal_interval = 1
    self.num_suggestions = 40
    self.step_size = 0.5
    self.std = 1.0
    self.random_seed = 7
    super().setUp()

  @parameterized.named_parameters(
      ('hill_climb', 'hill_climb'), ('neat', 'neat'),
      ('policy_gradient', 'policy_gradient'),
      ('random_search', 'random_search'),
      ('regularized_evolution', 'regularized_evolution'))
  def test_es_enas_step(self, controller_str):
    algo = es_enas_algorithm.ES_ENAS(
        controller_str=controller_str,
        dna_proposal_interval=self.dna_proposal_interval,
        num_suggestions=self.num_suggestions,
        step_size=self.step_size,
        std=self.std,
        random_seed=self.random_seed)

    init_state = make_init_state()
    algo.initialize(init_state)

    suggestion_list = algo.get_param_suggestions(evaluate=False)
    self.assertEqual(algo._interval_counter, 1)

    eval_results = make_evaluation_results(suggestion_list)
    algo.process_evaluations(eval_results)

    # Verifies that algo can keep track of previous evaluations.
    current_full_state = algo.state
    controller_state = algo._controller.get_state()

    algo.initialize(init_state)
    algo.state = current_full_state
    self.assertEqual(algo._controller.get_state(), controller_state)

  @parameterized.named_parameters(('False', False), ('True', True))
  def test_multithreading(self, multithreading):
    algo = es_enas_algorithm.ES_ENAS(
        dna_proposal_interval=self.dna_proposal_interval,
        multithreading=multithreading,
        num_suggestions=self.num_suggestions,
        step_size=self.step_size,
        std=self.std,
        random_seed=self.random_seed)

    init_state = make_init_state()
    algo.initialize(init_state)

    suggestion_list = algo.get_param_suggestions(evaluate=False)
    eval_results = make_evaluation_results(suggestion_list)
    algo.process_evaluations(eval_results)


if __name__ == '__main__':
  absltest.main()
