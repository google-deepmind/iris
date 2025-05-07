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

import os
from iris import buffer
from iris import checkpoint_util
from iris.algorithms import multi_agent_ars_algorithm
from iris.workers import worker_util
import numpy as np
import tensorflow as tf
from absl.testing import absltest
from absl.testing import parameterized


class AlgorithmTest(tf.test.TestCase, parameterized.TestCase):

  def _init_algo(self, agent_keys=None):
    return multi_agent_ars_algorithm.MultiAgentAugmentedRandomSearch(
        num_suggestions=4,
        step_size=0.5,
        std=1.0,
        top_percentage=1,
        orthogonal_suggestions=True,
        quasirandom_suggestions=False,
        top_sort_type='diff',
        random_seed=7,
        agent_keys=agent_keys,
    )

  @parameterized.parameters(
      (None, ['arm', 'opp'], 2),
      (['agent_a', 'agent_b', 'agent_c'], ['agent_a', 'agent_b', 'agent_c'], 3),
  )
  def test_init(self, agent_keys, expected_agent_keys, expected_num_agents):
    algo = self._init_algo(agent_keys)
    self.assertListEqual(algo._agent_keys, expected_agent_keys)
    self.assertEqual(algo._num_agents, expected_num_agents)

  def _build_evaluation_results(self) -> list[worker_util.EvaluationResult]:
    eval_results = [
        worker_util.EvaluationResult(  # pytype: disable=wrong-arg-types  # numpy-scalars
            params_evaluated=np.array([10.0, 11.0, 12.0, 13.0]),
            value=10,
            metrics={'reward_arm': 10, 'reward_opp': -5},
        ),
        worker_util.EvaluationResult(  # pytype: disable=wrong-arg-types  # numpy-scalars
            params_evaluated=np.array([10.0, 11.0, 14.0, 15.0]),
            value=10,
            metrics={'reward_arm': 10, 'reward_opp': -10},
        ),
        worker_util.EvaluationResult(  # pytype: disable=wrong-arg-types  # numpy-scalars
            params_evaluated=np.empty(0),
            value=0,
            metrics={'reward_arm': 0, 'reward_opp': 0},
        ),
        worker_util.EvaluationResult(  # pytype: disable=wrong-arg-types  # numpy-scalars
            params_evaluated=np.array([1.0, 2.0, 3.0, 4.0]),
            value=10,
            metrics={'reward_arm': 10, 'reward_opp': -10},
        ),
        worker_util.EvaluationResult(  # pytype: disable=wrong-arg-types  # numpy-scalars
            params_evaluated=np.array([10.0, 11.0, 12.0, 13.0]),
            value=-10,
            metrics={'reward_arm': -10, 'reward_opp': 5},
        ),
        worker_util.EvaluationResult(  # pytype: disable=wrong-arg-types  # numpy-scalars
            params_evaluated=np.array([10.0, 11.0, 14.0, 15.0]),
            value=-10,
            metrics={'reward_arm': -10, 'reward_opp': 10},
        ),
        worker_util.EvaluationResult(  # pytype: disable=wrong-arg-types  # numpy-scalars
            params_evaluated=np.array([5.0, 6.0, 7.0, 8.0]),
            value=-10,
            metrics={'reward_arm': -10, 'reward_opp': 10},
        ),
        worker_util.EvaluationResult(  # pytype: disable=wrong-arg-types  # numpy-scalars
            params_evaluated=np.empty(0),
            value=0,
            metrics={'reward_arm': 0, 'reward_opp': 0},
        ),
    ]
    return eval_results

  @parameterized.parameters(
      (['arm', 'opp'], [[10.0, 11.0], [12.0, 13.0]], 2),
      (['1', '2', '3', '4'], [[10.0], [11.0], [12.0], [13.0]], 4),
  )
  def test_split_params(self, agent_keys, expected_split_params, num_agents):
    algo = self._init_algo(agent_keys=agent_keys)
    params = np.array([10.0, 11.0, 12.0, 13.0])
    split_params = algo._split_params(params)
    self.assertLen(split_params, num_agents)
    for p, exp_p in zip(split_params, expected_split_params):
      np.testing.assert_array_equal(p, np.asarray(exp_p))

  @parameterized.parameters(
      (
          [np.asarray([10.0, 11.0]), np.asarray([12.0, 13.0])],
          np.asarray([10.0, 11.0, 12.0, 13.0]),
      ),
      (
          [
              np.asarray([10.0]),
              np.asarray([11.0]),
              np.asarray([12.0]),
              np.asarray([13.0]),
          ],
          np.asarray([10.0, 11.0, 12.0, 13.0]),
      ),
  )
  def test_combine_params(self, split_params, expected_combined_params):
    algo = self._init_algo()
    combined_params = algo._combine_params(split_params)
    np.testing.assert_array_equal(combined_params, expected_combined_params)

  @parameterized.parameters(
      ('arm', np.asarray([10, 10]), np.asarray([-10, -10]), np.asarray([0, 1])),
      ('opp', np.asarray([-10, -5]), np.asarray([10, 5]), np.asarray([1, 0])),
  )
  def test_get_top_evaluation_results(
      self, agent_key, expected_pos_evals, expected_neg_evals, expected_idx
  ):
    algo = self._init_algo()
    eval_results = self._build_evaluation_results()
    filtered_pos_eval_results = eval_results[:2]
    filtered_neg_eval_results = eval_results[4:6]
    pos_evals, neg_evals, idx = algo._get_top_evaluation_results(
        agent_key=agent_key,
        pos_eval_results=filtered_pos_eval_results,
        neg_eval_results=filtered_neg_eval_results,
    )
    np.testing.assert_array_equal(pos_evals, expected_pos_evals)
    np.testing.assert_array_equal(neg_evals, expected_neg_evals)
    np.testing.assert_array_equal(idx, expected_idx)

  def test_multi_agent_ars_gradient(self):
    algo = self._init_algo()
    init_state = {'init_params': np.array([10.0, 10.0, 10.0, 10.0])}
    algo.initialize(init_state)
    suggestions = algo.get_param_suggestions()
    self.assertLen(suggestions, 8)
    eval_results = self._build_evaluation_results()
    algo.process_evaluations(eval_results)
    np.testing.assert_array_almost_equal(
        algo._opt_params, np.array([10.0, 11.0, 6.83772234, 5.88903904])
    )

  @parameterized.parameters(
      ({'params_to_eval': np.asarray([1, 2]),
        'obs_norm_state': None},
       {'params_to_eval': np.asarray([1, 2]),
        'obs_norm_state': None},
       False, 2),
      ({'params_to_eval': np.asarray([1, 2]),
        'obs_norm_state': {'mean': np.asarray([3., 4.]),
                           'std': np.asarray([5., 6.]),
                           'n': 5}},
       {'params_to_eval': np.asarray([1, 2]),
        'obs_norm_state': {'mean': np.asarray([3., 4.]),
                           'std': np.asarray([5., 6.]),
                           'n': 5}},
       False, 3),
      ({'params_to_eval': np.asarray([1, 2,]),
        'obs_norm_state': None},
       {'params_to_eval': np.asarray([1, 2, 1, 2]),
        'obs_norm_state': None},
       True, 2),
      ({'params_to_eval': np.asarray([1, 2]),
        'obs_norm_state': {'mean': np.asarray([3., 4.,]),
                           'std': np.asarray([5., 6.,]),
                           'n': 5}},
       {'params_to_eval': np.asarray([1, 2, 1, 2]),
        'obs_norm_state': {'mean': np.asarray([3., 4., 3., 4.]),
                           'std': np.asarray([5., 6., 5., 6.]),
                           'n': 5}},
       True, 2),
      ({'params_to_eval': np.asarray([1, 2]),
        'obs_norm_state': {'mean': np.asarray([3., 4.]),
                           'std': np.asarray([5., 6.]),
                           'n': 5}},
       {'params_to_eval': np.asarray([1, 2, 1, 2, 1, 2, 1, 2]),
        'obs_norm_state': {'mean': np.asarray([3., 4., 3., 4., 3., 4., 3., 4.]),
                           'std': np.asarray([5., 6., 5., 6., 5., 6., 5., 6.]),
                           'n': 5}},
       True, 4),
      ({'params_to_eval': np.asarray([1, 2]),
        'obs_norm_state': {'mean': np.asarray([3., 4.]),
                           'std': np.asarray([5., 6.]),
                           'n': 5}},
       {'params_to_eval': np.asarray([1, 2]),
        'obs_norm_state': {'mean': np.asarray([3., 4.]),
                           'std': np.asarray([5., 6.]),
                           'n': 5}},
       True, 1),
  )
  def test_restore_state_from_checkpoint(
      self, state, expected_state, restore_state_from_single_agent, num_agents
  ):
    algo = multi_agent_ars_algorithm.MultiAgentAugmentedRandomSearch(
        num_suggestions=3,
        step_size=0.5,
        std=1.0,
        top_percentage=1,
        orthogonal_suggestions=True,
        quasirandom_suggestions=True,
        obs_norm_data_buffer=buffer.MeanStdBuffer()
        if state['obs_norm_state'] is not None
        else None,
        agent_keys=[str(i) for i in range(num_agents)],
        restore_state_from_single_agent=restore_state_from_single_agent,
        random_seed=7,
    )
    self.assertEqual(algo._num_agents, num_agents)
    init_state = {'init_params': np.array([10.0, 10.0])}
    if state['obs_norm_state'] is not None:
      init_state['obs_norm_buffer_data'] = {
          'mean': np.asarray([0.0, 0.0]),
          'std': np.asarray([1.0, 1.0]),
          'n': 0,
      }
    algo.initialize(init_state)

    with self.subTest('init-mean'):
      self.assertAllClose(np.array(algo._opt_params), init_state['init_params'])
    if (
        state['obs_norm_state'] is not None
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

    algo.restore_state_from_checkpoint(state)

    self.assertAllClose(algo._opt_params, expected_state['params_to_eval'])
    if (
        expected_state['obs_norm_state'] is not None
        and algo._obs_norm_data_buffer is not None
    ):
      std = expected_state['obs_norm_state']['std']
      var = np.square(std)
      expected_unnorm_var = var * 4
      with self.subTest('restore-obs-mean'):
        self.assertAllClose(
            np.asarray(algo._obs_norm_data_buffer.data['mean']),
            np.asarray(expected_state['obs_norm_state']['mean']),
        )
      with self.subTest('restore-obs-n'):
        self.assertAllClose(
            np.asarray(algo._obs_norm_data_buffer.data['n']),
            np.asarray(expected_state['obs_norm_state']['n']),
        )
      with self.subTest('restore-obs-std'):
        self.assertAllClose(
            np.asarray(algo._obs_norm_data_buffer.data['unnorm_var']),
            expected_unnorm_var,
        )

  @parameterized.parameters(
      (
          {'params_to_eval': np.asarray([1, 2]), 'obs_norm_state': None},
          [
              {'params_to_eval': np.asarray([1]), 'obs_norm_state': None},
              {'params_to_eval': np.asarray([2]), 'obs_norm_state': None},
          ],
          2,
      ),
      (
          {
              'params_to_eval': np.asarray([1, 2, 3, 4, 5, 6]),
              'obs_norm_state': None,
          },
          [
              {'params_to_eval': np.asarray([1, 2, 3]), 'obs_norm_state': None},
              {'params_to_eval': np.asarray([4, 5, 6]), 'obs_norm_state': None},
          ],
          2,
      ),
      (
          {
              'params_to_eval': np.asarray([1, 2, 3, 4, 5, 6]),
              'obs_norm_state': {
                  'mean': np.asarray([6.0, 7.0, 8.0, 9.0]),
                  'std': np.asarray([10.0, 11.0, 12.0, 13.0]),
                  'n': 5,
              },
          },
          [
              {
                  'params_to_eval': np.asarray([1, 2, 3]),
                  'obs_norm_state': {
                      'mean': np.asarray([6.0, 7.0]),
                      'std': np.asarray([10.0, 11.0]),
                      'n': 5,
                  },
              },
              {
                  'params_to_eval': np.asarray([4, 5, 6]),
                  'obs_norm_state': {
                      'mean': np.asarray([8.0, 9.0]),
                      'std': np.asarray([12.0, 13.0]),
                      'n': 5,
                  },
              },
          ],
          2,
      ),
      (
          {
              'params_to_eval': np.asarray([1, 2, 3, 4, 5, 6]),
              'obs_norm_state': {
                  'mean': np.asarray(
                      [[6.0, 7.0, 8.0, 9.0], [10.0, 11.0, 12.0, 13.0]]
                  ),
                  'std': np.asarray(
                      [[14.0, 15.0, 16.0, 17.0], [18.0, 19.0, 20.0, 21.0]]
                  ),
                  'n': 5,
              },
          },
          [
              {
                  'params_to_eval': np.asarray([1, 2, 3]),
                  'obs_norm_state': {
                      'mean': np.asarray([[6.0, 7.0], [10.0, 11.0]]),
                      'std': np.asarray([[14.0, 15.0], [18.0, 19.0]]),
                      'n': 5,
                  },
              },
              {
                  'params_to_eval': np.asarray([4, 5, 6]),
                  'obs_norm_state': {
                      'mean': np.asarray([[8.0, 9.0], [12.0, 13.0]]),
                      'std': np.asarray([[16.0, 17.0], [20.0, 21.0]]),
                      'n': 5,
                  },
              },
          ],
          2,
      ),
      (
          {
              'params_to_eval': np.asarray([1, 2, 3, 4, 5, 6]),
              'obs_norm_state': None,
          },
          [
              {'params_to_eval': np.asarray([1, 2]), 'obs_norm_state': None},
              {'params_to_eval': np.asarray([3, 4]), 'obs_norm_state': None},
              {'params_to_eval': np.asarray([5, 6]), 'obs_norm_state': None},
          ],
          3,
      ),
      (
          {
              'params_to_eval': np.asarray([1, 2, 3, 4, 5, 6]),
              'obs_norm_state': {
                  'mean': np.asarray([
                      [6.0, 7.0, 8.0, 9.0, 10.0, 11.0],
                      [12.0, 13.0, 14.0, 15.0, 16.0, 17.0],
                  ]),
                  'std': np.asarray([
                      [14.0, 15.0, 16.0, 17.0, 18.0, 19.0],
                      [20.0, 21.0, 22.0, 23.0, 24.0, 25.0],
                  ]),
                  'n': 5,
              },
          },
          [
              {
                  'params_to_eval': np.asarray([1, 2]),
                  'obs_norm_state': {
                      'mean': np.asarray([[6.0, 7.0], [12.0, 13.0]]),
                      'std': np.asarray([[14.0, 15.0], [20.0, 21.0]]),
                      'n': 5,
                  },
              },
              {
                  'params_to_eval': np.asarray([3, 4]),
                  'obs_norm_state': {
                      'mean': np.asarray([[8.0, 9.0], [14.0, 15.0]]),
                      'std': np.asarray([[16.0, 17.0], [22.0, 23.0]]),
                      'n': 5,
                  },
              },
              {
                  'params_to_eval': np.asarray([5, 6]),
                  'obs_norm_state': {
                      'mean': np.asarray([[10.0, 11.0], [16.0, 17.0]]),
                      'std': np.asarray([[18.0, 19.0], [24.0, 25.0]]),
                      'n': 5,
                  },
              },
          ],
          3,
      ),
  )
  def test_maybe_save_custom_checkpoint(
      self, state, expected_states, num_agents
  ):
    tempdir = self.create_tempdir()
    path = 'checkpoint_iteration_0'
    full_path = os.path.join(tempdir, path)
    algo = multi_agent_ars_algorithm.MultiAgentAugmentedRandomSearch(
        num_suggestions=3,
        step_size=0.5,
        std=1.0,
        top_percentage=1,
        orthogonal_suggestions=True,
        quasirandom_suggestions=True,
        obs_norm_data_buffer=buffer.MeanStdBuffer()
        if state['obs_norm_state'] is not None
        else None,
        agent_keys=[str(i) for i in range(num_agents)],
        random_seed=7,
    )
    algo.maybe_save_custom_checkpoint(state, full_path)
    for i in range(num_agents):
      agent_checkpoint_path = f'{full_path}_agent_{i}'
      agent_state = checkpoint_util.load_checkpoint_state(agent_checkpoint_path)
      self.assertAllClose(
          agent_state['params_to_eval'], expected_states[i]['params_to_eval']
      )
      if expected_states[i]['obs_norm_state'] is not None:
        self.assertAllClose(
            agent_state['obs_norm_state']['mean'],
            expected_states[i]['obs_norm_state']['mean'],
        )
        self.assertAllClose(
            agent_state['obs_norm_state']['std'],
            expected_states[i]['obs_norm_state']['std'],
        )
        self.assertAllClose(
            agent_state['obs_norm_state']['n'],
            expected_states[i]['obs_norm_state']['n'],
        )

  def test_split_checkpoint(self):
    tempdir = self.create_tempdir()
    path = 'checkpoint_iteration_0'
    full_path = os.path.join(tempdir, path)
    algo = multi_agent_ars_algorithm.MultiAgentAugmentedRandomSearch(
        num_suggestions=3,
        step_size=0.5,
        std=1.0,
        top_percentage=1,
        orthogonal_suggestions=True,
        quasirandom_suggestions=True,
        obs_norm_data_buffer=buffer.MeanStdBuffer(),
        agent_keys=[str(i) for i in range(3)],
        random_seed=7,
    )
    state = {
        'params_to_eval': np.asarray([1, 2, 3, 4, 5, 6]),
        'obs_norm_state': {
            'mean': np.asarray([
                [6.0, 7.0, 8.0, 9.0, 10.0, 11.0],
                [12.0, 13.0, 14.0, 15.0, 16.0, 17.0],
            ]),
            'std': np.asarray([
                [14.0, 15.0, 16.0, 17.0, 18.0, 19.0],
                [20.0, 21.0, 22.0, 23.0, 24.0, 25.0],
            ]),
            'n': 5,
        },
    }
    expected_states = [
        {
            'params_to_eval': np.asarray([1, 2]),
            'obs_norm_state': {
                'mean': np.asarray([[6.0, 7.0], [12.0, 13.0]]),
                'std': np.asarray([[14.0, 15.0], [20.0, 21.0]]),
                'n': 5,
            },
        },
        {
            'params_to_eval': np.asarray([3, 4]),
            'obs_norm_state': {
                'mean': np.asarray([[8.0, 9.0], [14.0, 15.0]]),
                'std': np.asarray([[16.0, 17.0], [22.0, 23.0]]),
                'n': 5,
            },
        },
        {
            'params_to_eval': np.asarray([5, 6]),
            'obs_norm_state': {
                'mean': np.asarray([[10.0, 11.0], [16.0, 17.0]]),
                'std': np.asarray([[18.0, 19.0], [24.0, 25.0]]),
                'n': 5,
            },
        },
    ]
    checkpoint_util.save_checkpoint(full_path, state)
    algo.split_and_save_checkpoint(checkpoint_path=full_path)
    for i in range(3):
      agent_checkpoint_path = f'{full_path}_agent_{i}'
      agent_state = checkpoint_util.load_checkpoint_state(agent_checkpoint_path)
      self.assertAllClose(
          agent_state['params_to_eval'], expected_states[i]['params_to_eval']
      )
      if expected_states[i]['obs_norm_state'] is not None:
        self.assertAllClose(
            agent_state['obs_norm_state']['mean'],
            expected_states[i]['obs_norm_state']['mean'],
        )
        self.assertAllClose(
            agent_state['obs_norm_state']['std'],
            expected_states[i]['obs_norm_state']['std'],
        )
        self.assertAllClose(
            agent_state['obs_norm_state']['n'],
            expected_states[i]['obs_norm_state']['n'],
        )


if __name__ == '__main__':
  absltest.main()
