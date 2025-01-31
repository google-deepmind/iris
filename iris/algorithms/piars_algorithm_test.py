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
from iris.algorithms import piars_algorithm
from iris.policies import keras_pi_policy
from iris.workers import worker_util
import numpy as np
from absl.testing import absltest
from absl.testing import parameterized


class AlgorithmTest(parameterized.TestCase):

  @parameterized.parameters(
      (True, False),
      (False, False),
  )
  def test_ars_gradient(self, orthogonal_suggestions, quasirandom_suggestions):
    env = gym.make(id='Pendulum-v0')
    policy = keras_pi_policy.KerasPIPolicy(
        ob_space=env.observation_space,
        ac_space=env.action_space,
        state_dim=2,
        conv_filter_sizes=(2,),
        conv_kernel_sizes=(2,),
        image_feature_length=2,
        fc_layer_sizes=(2,),
        h_fc_layer_sizes=(2,),
        f_fc_layer_sizes=(2,),
        g_fc_layer_sizes=(2,),
    )
    algo = piars_algorithm.PIARS(
        num_suggestions=3,
        step_size=0.5,
        std=1.0,
        top_percentage=1,
        orthogonal_suggestions=orthogonal_suggestions,
        quasirandom_suggestions=quasirandom_suggestions,
        env=env,
        policy=policy,
        learn_representation=False,
        random_seed=7,
    )
    init_state = {
        'init_params': np.array([10.0, 10.0]),
        'init_representation_params': np.array([10.0, 10.0]),
    }
    algo.initialize(init_state)
    eval_results = [
        worker_util.EvaluationResult(np.array([10.0, 11.0]), 10),  # pytype: disable=wrong-arg-types  # numpy-scalars
        worker_util.EvaluationResult(np.empty(0), 0),  # pytype: disable=wrong-arg-types  # numpy-scalars
        worker_util.EvaluationResult(np.array([10.0, 11.0]), 10),  # pytype: disable=wrong-arg-types  # numpy-scalars
        worker_util.EvaluationResult(np.array([10.0, 9.0]), -10),  # pytype: disable=wrong-arg-types  # numpy-scalars
        worker_util.EvaluationResult(np.array([10.0, 9.0]), -10),  # pytype: disable=wrong-arg-types  # numpy-scalars
        worker_util.EvaluationResult(np.empty(0), 0),  # pytype: disable=wrong-arg-types  # numpy-scalars
    ]
    algo.process_evaluations(eval_results)
    np.testing.assert_array_equal(algo._opt_params, np.array([10, 11]))


if __name__ == '__main__':
  absltest.main()
