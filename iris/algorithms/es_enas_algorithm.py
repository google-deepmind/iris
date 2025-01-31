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

"""Algorithm class for ES-ENAS algorithm."""

import functools
from multiprocessing import dummy as mp_threads
from typing import Any, Dict, Sequence
from iris.algorithms import ars_algorithm
from iris.algorithms import controllers
from iris.workers import worker_util
import pyglove as pg


class ES_ENAS(ars_algorithm.AugmentedRandomSearch):  # pylint: disable=invalid-name
  """ES-ENAS algorithm for NAS-related blackbox optimization.

  Adds PyGlove as an additional optimizer for discrete/combinatorial search
  spaces, making this a combination of two different algorithms (ARS and PyGlove
  controllers).

  At its core logic, mainly appends an extra "dna" (model architecture) to the
  AugmentedRandomSearch request. This "dna" is then processed by the PyGlove
  controller.
  """

  def __init__(self,
               controller_str: str = "regularized_evolution",
               dna_proposal_interval: int = 50,
               multithreading: bool = False,
               **kwargs) -> None:
    """Initializes the ES-ENAS algorithm, as well as ARS parent class.

    Args:
      controller_str: Which controller algorithm to use on PyGlove side.
      dna_proposal_interval: Iteration interval at which to propose new
        architectures.
      multithreading: Whether to multithread PyGlove DNA serialization. Pool
        created after __init__ to avoid Launchpad pickling issues.
      **kwargs: Arguments to parent classes (e.g. AugmentedRandomSearch)
    """
    super().__init__(**kwargs)
    self._interval_counter = 0
    self._dna_proposal_interval = dna_proposal_interval
    self._controller_fn = functools.partial(
        controllers.CONTROLLER_DICT[controller_str],
        batch_size=2 * self._num_suggestions)

    self._multithreading = multithreading

  def initialize(self, state: Dict[str, Any]) -> None:
    super().initialize(state)
    if self._multithreading:
      self._pool = mp_threads.Pool(self._num_suggestions)

    self._dna_spec = pg.from_json_str(state["serialized_dna_spec"])
    self._controller = self._controller_fn(dna_spec=self._dna_spec)
    self._interval_counter = 0
    self._evaluated_serialized_dnas = []
    self._evaluated_rewards = []

  def process_evaluations(
      self, eval_results: Sequence[worker_util.EvaluationResult]) -> None:
    super().process_evaluations(eval_results)

    eval_metadatas = []
    eval_rewards = []
    for eval_result in eval_results:
      if eval_result.metadata:
        eval_metadatas.append(eval_result.metadata)
        eval_rewards.append(eval_result.value)

    if eval_metadatas:

      def proper_unserialize(metadata: str) -> pg.DNA:
        dna = pg.from_json_str(metadata)
        # Put back the DNASpec into DNA, since serialization removed it.
        dna.use_spec(self._dna_spec)
        return dna

      if self._multithreading:
        dna_list = self._pool.map(proper_unserialize, eval_metadatas)  # pytype:disable=attribute-error
      else:
        dna_list = map(proper_unserialize, eval_metadatas)
      dna_list = list(dna_list)
      self._controller.collect_rewards_and_train(eval_rewards, dna_list)

  def get_param_suggestions(self,
                            evaluate: bool = False) -> Sequence[Dict[str, Any]]:
    vanilla_suggestions = super().get_param_suggestions(evaluate)
    # Evaluation never calls `process_evaluations`, but we need to update eval
    # worker DNAs.
    suggest_dna_bool = (self._interval_counter %
                        self._dna_proposal_interval) == 0

    if suggest_dna_bool:
      # Note that for faster serialization, DNASpec is removed from DNA.
      dna_list = [self._controller.propose_dna() for _ in vanilla_suggestions]
      if self._multithreading:
        metadata_list = self._pool.map(pg.to_json_str, dna_list)  # pytype:disable=attribute-error
      else:
        metadata_list = map(pg.to_json_str, dna_list)
      metadata_list = list(metadata_list)
    else:
      metadata_list = [None] * len(vanilla_suggestions)

    for i, vanilla_suggestion in enumerate(vanilla_suggestions):
      vanilla_suggestion["metadata"] = metadata_list[i]

    if not evaluate:
      self._interval_counter += 1
    return vanilla_suggestions

  def _get_state(self) -> Dict[str, Any]:
    vanilla_state = super()._get_state()
    vanilla_state["interval_counter"] = self._interval_counter
    vanilla_state["serialized_dna_spec"] = pg.to_json_str(self._dna_spec)
    vanilla_state["controller_alg_state"] = self._controller.get_state()
    return vanilla_state

  def _set_state(self, new_state: Dict[str, Any]) -> None:
    super()._set_state(new_state)  # pytype: disable=attribute-error
    self._interval_counter = new_state["interval_counter"]
    self._dna_spec = pg.from_json_str(new_state["serialized_dna_spec"])
    self._controller = self._controller_fn(dna_spec=self._dna_spec)
    self._controller.set_state(new_state["controller_alg_state"])
