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

from unittest import mock

from iris import buffer
from iris.algorithms import algorithm
from iris.algorithms import pyribs_algorithm
from iris.workers import worker_util
import numpy as np
from ribs import archives

from absl.testing import absltest

# Define two arbitrary specs with different ranges so we can test for them.
_X_SPEC = pyribs_algorithm.MeasureSpec('x', (0, 10), 10)
_Y_SPEC = pyribs_algorithm.MeasureSpec('y', (1, 100), 20)


class PyribsAlgorithmTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    # Basic parameters chosen to be simple enough to not distract from the
    # algorithm logic but with enough complexity to test functionality e.g.
    # using multiple measure specs.
    self.buffer = buffer.MeanStdBuffer(shape=(8,))
    self.num_suggestions_per_emitter = 10
    self.num_emitters = 20
    self.initial_step_size = 1.0
    self.num_evals = 100

    self.initial_params = np.ones((13,))

    self.test_algorithm = pyribs_algorithm.PyRibsAlgorithm(
        measure_specs=[_X_SPEC, _Y_SPEC],
        obs_norm_data_buffer=self.buffer,
        initial_step_size=self.initial_step_size,
        num_suggestions_per_emitter=self.num_suggestions_per_emitter,
        num_emitters=self.num_emitters,
        num_evals=self.num_evals,
    )
    self.test_algorithm.initialize(
        {algorithm.PARAMS_TO_EVAL: self.initial_params}
    )

  def test_initialize_with_obs_norm_state(self):
    self.test_algorithm.initialize({
        algorithm.PARAMS_TO_EVAL: self.initial_params,
        algorithm.OBS_NORM_BUFFER_STATE: self.buffer.state,
    })

    np.testing.assert_equal(
        self.buffer.state,
        self.test_algorithm.state[algorithm.OBS_NORM_BUFFER_STATE],
    )
    np.testing.assert_equal(
        self.initial_params, self.test_algorithm.state[algorithm.PARAMS_TO_EVAL]
    )

  def test_get_param_suggestions(self):
    suggestions = self.test_algorithm.get_param_suggestions()

    self.assertLen(
        suggestions, self.num_emitters * self.num_suggestions_per_emitter
    )
    for suggestion in suggestions:
      self.assertLen(
          suggestion[algorithm.PARAMS_TO_EVAL], self.initial_params.size
      )
      np.testing.assert_equal(
          suggestion[algorithm.OBS_NORM_BUFFER_STATE], self.buffer.state
      )
      self.assertTrue(suggestion[algorithm.UPDATE_OBS_NORM_BUFFER])

  def test_get_param_suggestions_for_eval_is_empty_initially(self):
    self.assertEmpty(self.test_algorithm.get_param_suggestions(evaluate=True))

  def test_get_param_suggestions_for_eval(self):
    suggestions = self.test_algorithm.get_param_suggestions()
    evaluations = [
        worker_util.EvaluationResult(
            params_evaluated=suggestion[algorithm.PARAMS_TO_EVAL],
            value=1,
            obs_norm_buffer_data=suggestion[algorithm.OBS_NORM_BUFFER_STATE]
            | {buffer.N: 1, buffer.UNNORM_VAR: np.ones((8,))},
            metrics={'x': 1, 'y': 10},
        )
        for suggestion in suggestions
    ]
    # Give the first evaluation a high score so it is the elite.
    evaluations[0].value = 1000
    if evaluations[0].obs_norm_buffer_data is not None:
      evaluations[0].obs_norm_buffer_data[buffer.N] = 1000
    self.test_algorithm.process_evaluations(evaluations)

    eval_suggestions = self.test_algorithm.get_param_suggestions(evaluate=True)

    self.assertLen(eval_suggestions, self.num_evals)
    for eval_suggestion in eval_suggestions:
      np.testing.assert_equal(
          eval_suggestion[algorithm.PARAMS_TO_EVAL],
          evaluations[0].params_evaluated
      )
      np.testing.assert_equal(
          eval_suggestion[algorithm.OBS_NORM_BUFFER_STATE][buffer.N],
          evaluations[0].obs_norm_buffer_data[buffer.N],
      )
      self.assertFalse(eval_suggestion[algorithm.UPDATE_OBS_NORM_BUFFER])

  def test_restore_state_from_checkpoint_without_archive(self):
    checkpoint_buffer = buffer.MeanStdBuffer(shape=(8,))
    checkpoint_buffer.push(np.ones(8,))
    checkpoint_state = {
        algorithm.PARAMS_TO_EVAL: np.zeros((13,)),
        algorithm.OBS_NORM_BUFFER_STATE: checkpoint_buffer.state,
    }
    self.test_algorithm.restore_state_from_checkpoint(checkpoint_state)
    state_after_checkpoint = self.test_algorithm.state

    np.testing.assert_equal(
        state_after_checkpoint[algorithm.PARAMS_TO_EVAL],
        checkpoint_state[algorithm.PARAMS_TO_EVAL],
    )
    np.testing.assert_equal(
        state_after_checkpoint[algorithm.OBS_NORM_BUFFER_STATE],
        checkpoint_buffer.state,
    )
    # Archive not restored so it should be empty.
    self.assertEmpty(
        state_after_checkpoint[pyribs_algorithm._ARCHIVE_DATA][
            pyribs_algorithm._SOLUTION
        ]
    )

  def test_restore_state_from_checkpoint_with_archive(self):
    checkpoint_buffer = buffer.MeanStdBuffer(shape=(8,))
    checkpoint_buffer.push(
        np.ones(
            8,
        )
    )
    buffer_state = checkpoint_buffer.state
    checkpoint_archive = archives.GridArchive(
        solution_dim=self.initial_params.size,
        dims=(_X_SPEC.num_buckets, _Y_SPEC.num_buckets),
        ranges=(_X_SPEC.range, _Y_SPEC.range),
        qd_score_offset=0,
        extra_fields={
            pyribs_algorithm._OBS_NORM_MEAN: (
                buffer_state[buffer.MEAN].size,
                np.float32,
            ),
            pyribs_algorithm._OBS_NORM_STD: (
                buffer_state[buffer.STD].size,
                np.float32,
            ),
            pyribs_algorithm._OBS_NORM_N: ((), np.int32),
        },
    )
    # Add 3 solutions to the archive to be restored.
    checkpoint_archive.add(
        solution=[np.ones((13,)), np.ones((13,))*2, np.ones((13,))*3],
        objective=[1, 2, 3],
        measures=[(1, 10), (2, 20), (3, 30)],
        obs_norm_mean=[np.ones((8,)), np.ones((8,))*2, np.ones((8,))*3],
        obs_norm_std=[np.ones((8,)), np.ones((8,))*2, np.ones((8,))*3],
        obs_norm_n=[1, 2, 3],
    )

    checkpoint_state = {
        algorithm.PARAMS_TO_EVAL: np.zeros((13,)),
        algorithm.OBS_NORM_BUFFER_STATE: checkpoint_buffer.state,
        pyribs_algorithm._ARCHIVE_DATA: checkpoint_archive.data(),
    }
    self.test_algorithm.restore_state_from_checkpoint(checkpoint_state)
    state_after_checkpoint = self.test_algorithm.state

    np.testing.assert_equal(
        state_after_checkpoint[algorithm.PARAMS_TO_EVAL],
        checkpoint_state[algorithm.PARAMS_TO_EVAL],
    )
    np.testing.assert_equal(
        state_after_checkpoint[algorithm.OBS_NORM_BUFFER_STATE],
        checkpoint_buffer.state,
    )
    # Archive should have 3 restored elements in it.
    self.assertLen(
        state_after_checkpoint[pyribs_algorithm._ARCHIVE_DATA][
            pyribs_algorithm._SOLUTION
        ],
        3,
    )

  def test_process_evaluations(self):
    evaluations = [
        worker_util.EvaluationResult(
            params_evaluated=np.ones((13,)),
            value=1,
            obs_norm_buffer_data={
                buffer.N: 1,
                buffer.STD: np.ones((8,)),
                buffer.MEAN: np.ones((8,)),
                buffer.UNNORM_VAR: np.ones((8,)),
            },
            metrics={'x': 1, 'y': 10},
        ),
        worker_util.EvaluationResult(
            params_evaluated=np.ones((13,) * 2),
            value=2,
            obs_norm_buffer_data={
                buffer.N: 2,
                buffer.STD: np.ones((8,)) * 2,
                buffer.MEAN: np.ones((8,)) * 2,
                buffer.UNNORM_VAR: np.ones((8,)) * 2,
            },
            metrics={'x': 2, 'y': 20},
        ),
    ]

    with mock.patch.object(
        self.test_algorithm._scheduler, 'tell', autospec=True
    ) as mock_tell:
      self.test_algorithm.process_evaluations(evaluations)

      # There are no outputs, so just check the function was called correctly.
      mock_tell.assert_called_once()
      self.assertEqual(mock_tell.call_args.kwargs['objective'], [1, 2])
      self.assertEqual(
          mock_tell.call_args.kwargs['measures'], [[1, 10], [2, 20]]
      )
      np.testing.assert_equal(
          mock_tell.call_args.kwargs[pyribs_algorithm._OBS_NORM_MEAN],
          [
              np.ones((8,)),
              np.ones((8,)) * 2,
          ],
      )
      np.testing.assert_equal(
          mock_tell.call_args.kwargs[pyribs_algorithm._OBS_NORM_STD],
          [
              np.ones((8,)),
              np.ones((8,)) * 2,
          ],
      )
      self.assertEqual(
          mock_tell.call_args.kwargs[pyribs_algorithm._OBS_NORM_N], [1, 2]
      )


if __name__ == '__main__':
  absltest.main()
