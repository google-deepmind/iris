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

"""Algorithm class for PyRibs Quality Diversity Search.

See https://arxiv.org/abs/2303.00191 for a description of this library and
summary of Quality Diversity search algorithms.

For simplicity, this implementation only exposes a subset of the functionality
in PyRibs. Currently just Covariance Matrix Adaptation MAP-Elites (CMA-ME) with
a grid-based archive.
"""

import dataclasses
from typing import Any, Dict, Sequence

from iris import buffer as buffer_lib
from iris.algorithms import algorithm
from iris.workers import worker_util
import numpy as np
from ribs import archives
from ribs import emitters
from ribs import schedulers
from typing_extensions import override


_ARCHIVE_DATA = "archive_data"
# Pyribs internal column names for the archive.
_INDEX = "index"
_SOLUTION = "solution"
# Extra column names for storing normalizer data with solutions.
_OBS_NORM_PREFIX = "obs_norm_"
_OBS_NORM_MEAN = _OBS_NORM_PREFIX + buffer_lib.MEAN
_OBS_NORM_STD = _OBS_NORM_PREFIX + buffer_lib.STD
_OBS_NORM_N = _OBS_NORM_PREFIX + buffer_lib.N


@dataclasses.dataclass(frozen=True)
class MeasureSpec:
  """Specifications for behavior measures."""

  # Name of the behavior measure, must be a metric exported blackbox.
  name: str
  # Range of values the measure can take.
  range: tuple[float, float]
  # Number of buckets to divide the above range into.
  num_buckets: int


class PyRibsAlgorithm(algorithm.BlackboxAlgorithm):
  """Quality Diversity search for the blackbox optimization framework.

  Defines a quality diversity search that can be executed with the blackbox
  optimization framework (BBV2). The search uses the CMA-ME algorithm and a grid
  archive for tracking solutions.
  """

  def __init__(
      self,
      measure_specs: Sequence[MeasureSpec],
      obs_norm_data_buffer: buffer_lib.MeanStdBuffer,
      initial_step_size: float,
      num_suggestions_per_emitter: int,
      num_emitters: int,
      num_evals: int,
      qd_score_offset: float = 0,
      solution_ranker: str = "2imp",
  ) -> None:
    """Initializes a PyRibsAlgorithm.

    Args:
      measure_specs: List of behevaior measure to optimize over. These must be
        defined in metrics exported by the workers.
      obs_norm_data_buffer: Buffer to sync statistics from all workers for
        online mean std observation normalizer.
      initial_step_size: Starting step size of the search.
      num_suggestions_per_emitter: Number of suggestions each emitter. Total
        suggestions = num_suggestions_per_emitter * num_emitters.
      num_emitters: Number of suggestion emitters.  More emitters imples more
        varied exploration.
      num_evals: Number of evaluations to perform on the top solution for
        reporting the top score.
      qd_score_offset: Value to add to rewards such that good solutions are
        non-negative, see ribs/archives/_archive_base.py
        for details. Default of 0 means no adjustment is applied and negative
        rewards are not admitted to the archive.
      solution_ranker: String abbreviation of the ranker for emitting solutions.
        See ribs/emitters/rankers.py;l=13;rcl=601096249
        for options.  Default is TwoStageRandomDirectionRanker.
    """
    self._initial_step_size = initial_step_size
    self._num_suggestions_per_emitter = num_suggestions_per_emitter
    self._num_emitters = num_emitters
    self._obs_norm_data_buffer = obs_norm_data_buffer
    self._measure_specs = measure_specs
    self._qd_score_offset = qd_score_offset
    self._solution_ranker = solution_ranker

    self._measure_names = [measure.name for measure in measure_specs]
    self._archive_dims = [measure.num_buckets for measure in measure_specs]
    self._archive_ranges = [measure.range for measure in self._measure_specs]
    self._opt_params = np.empty(0)
    self._scheduler = None
    self._init_scheduler()
    super().__init__(
        num_suggestions=num_suggestions_per_emitter * num_emitters,
        random_seed=42,  # Unused.
        num_evals=num_evals,
    )

  def _init_scheduler(
      self, saved_archive: dict[str, np.ndarray] | None = None
  ) -> None:
    """Initializes the archive and scheduler for PyRibs.

    Args:
      saved_archive: Optional saved archive state to restore from.
    """

    # TODO: Eventually have a `state_spec` in the buffer class.
    buffer_state = self._obs_norm_data_buffer.state
    self._archive = archives.GridArchive(
        solution_dim=self._opt_params.size,
        dims=self._archive_dims,
        ranges=self._archive_ranges,
        qd_score_offset=self._qd_score_offset,
        extra_fields={
            _OBS_NORM_MEAN: (buffer_state[buffer_lib.MEAN].size, np.float32),
            _OBS_NORM_STD: (buffer_state[buffer_lib.STD].size, np.float32),
            _OBS_NORM_N: ((), np.int32),
        },
    )

    if saved_archive is not None:
      del saved_archive[_INDEX]  # Index is not needed to restore state.
      self._archive.add(**saved_archive)

    archive_emitters = [
        emitters.EvolutionStrategyEmitter(
            archive=self._archive,
            x0=self._opt_params.flatten(),
            sigma0=self._initial_step_size,
            ranker=self._solution_ranker,
            batch_size=self._num_suggestions_per_emitter,
        )
        for _ in range(self._num_emitters)
    ]
    self._scheduler = schedulers.Scheduler(self._archive, archive_emitters)

  @override
  def initialize(self, state: dict[str, Any]):
    self._opt_params = state[algorithm.PARAMS_TO_EVAL]
    if algorithm.OBS_NORM_BUFFER_STATE in state:
      self._obs_norm_data_buffer.data = state[algorithm.OBS_NORM_BUFFER_STATE]
    self._init_scheduler()

  @override
  def get_param_suggestions(
      self, evaluate: bool = False
  ) -> Sequence[Dict[str, Any]]:
    if evaluate and self._archive.best_elite is None:
      return []

    if evaluate:
      elite = self._archive.best_elite
      param_suggestions = [elite[_SOLUTION]] * self._num_evals
      buffer = {
          buffer_lib.N: elite[_OBS_NORM_N],
          buffer_lib.MEAN: elite[_OBS_NORM_MEAN],
          buffer_lib.STD: elite[_OBS_NORM_STD],
      }
    else:
      param_suggestions = self._scheduler.ask()
      buffer = self._obs_norm_data_buffer.state

    return [
        {
            algorithm.PARAMS_TO_EVAL: params,
            algorithm.OBS_NORM_BUFFER_STATE: buffer,
            algorithm.UPDATE_OBS_NORM_BUFFER: not evaluate,
        }
        for params in param_suggestions
    ]

  def process_evaluations(
      self, eval_results: Sequence[worker_util.EvaluationResult]
  ) -> None:
    objective = []
    measures = []
    obs_norm_n = []
    obs_norm_std = []
    obs_norm_mean = []
    for result in eval_results:
      self._obs_norm_data_buffer.merge(result.obs_norm_buffer_data)
      objective.append(result.value)
      measures.append([result.metrics[name] for name in self._measure_names])
      obs_norm_n.append(result.obs_norm_buffer_data[buffer_lib.N])
      obs_norm_std.append(result.obs_norm_buffer_data[buffer_lib.STD])
      obs_norm_mean.append(result.obs_norm_buffer_data[buffer_lib.MEAN])

    # Store the state of the obs_norm_buffer for each solution so that it can be
    # reproduced later when evaluating the policy, similar to other algorithms
    # that use the Blackbox framework.
    extra_fields = {
        _OBS_NORM_MEAN: obs_norm_mean,
        _OBS_NORM_STD: obs_norm_std,
        _OBS_NORM_N: obs_norm_n,
    }

    self._scheduler.tell(
        objective=objective,
        measures=measures,
        **extra_fields,
    )

  @property
  @override
  def state(self) -> Dict[str, Any]:
    return {
        algorithm.PARAMS_TO_EVAL: self._opt_params,
        algorithm.OBS_NORM_BUFFER_STATE: self._obs_norm_data_buffer.state,
        # Phoenix cannot serialize the archive directly, so use the dict state.
        _ARCHIVE_DATA: self._archive.data(),
    }

  @override
  def restore_state_from_checkpoint(self, new_state: Dict[str, Any]) -> None:
    self.state = new_state

  @state.setter
  @override
  def state(self, new_state: Dict[str, Any]) -> None:
    self._opt_params = new_state[algorithm.PARAMS_TO_EVAL]
    self._obs_norm_data_buffer.state = new_state[
        algorithm.OBS_NORM_BUFFER_STATE
    ]
    self._init_scheduler(new_state.get(_ARCHIVE_DATA, None))
