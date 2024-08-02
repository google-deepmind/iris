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

"""Algorithm class for any PyGlove controller-only algorithm."""
import functools
from multiprocessing import dummy as mp_threads
from typing import Any, Dict, Sequence
from iris import worker_util
from iris.algorithms import algorithm
from iris.algorithms import controllers
import numpy as np
import pyglove as pg


class PyGloveAlgorithm(algorithm.BlackboxAlgorithm):
  """Uses a PyGlove algorithm end-to-end for entire Blackbox Algorithm."""

  def __init__(self,
               controller_str: str = "regularized_evolution",
               multithreading: bool = False,
               **kwargs) -> None:
    """Initializes the PyGlove algorithm.

    Args:
      controller_str: Which controller algorithm to use on PyGlove side.
      multithreading: Whether to multithread PyGlove DNA serialization. Pool
        created after __init__ to avoid Launchpad pickling issues.
      **kwargs: Arguments to parent BlackboxAlgorithm class.
    """
    super().__init__(**kwargs)
    self._controller_fn = functools.partial(
        controllers.CONTROLLER_DICT[controller_str],
        batch_size=self._num_suggestions)

    self._multithreading = multithreading

  def initialize(self, state: Dict[str, Any]) -> None:
    if self._multithreading:
      self._pool = mp_threads.Pool(self._num_suggestions)

    self._dna_spec = pg.from_json_str(state["serialized_dna_spec"])
    self._controller = self._controller_fn(dna_spec=self._dna_spec)
    self._evaluated_serialized_dnas = []
    self._evaluated_rewards = []

  def process_evaluations(
      self, eval_results: Sequence[worker_util.EvaluationResult]) -> None:
    eval_metadatas = []
    eval_rewards = []
    for eval_result in eval_results:
      if eval_result.metadata:
        eval_metadatas.append(eval_result.metadata)
        eval_rewards.append(eval_result.value)

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

    for dna in dna_list:

      dna.use_spec(self._dna_spec)
    self._controller.collect_rewards_and_train(eval_rewards, dna_list)

  def get_param_suggestions(self,
                            evaluate: bool = False) -> Sequence[Dict[str, Any]]:
    vanilla_suggestions = []

    dna_list = [
        self._controller.propose_dna() for _ in range(self._num_suggestions)
    ]
    # Note that for faster serialization, DNASpec is removed from DNA.
    if self._multithreading:
      metadata_list = self._pool.map(pg.to_json_str, dna_list)  # pytype:disable=attribute-error
    else:
      metadata_list = map(pg.to_json_str, dna_list)

    metadata_list = list(metadata_list)

    for metadata in metadata_list:
      suggestion = {"params_to_eval": np.empty((), dtype=np.float64)}
      suggestion["metadata"] = metadata
      vanilla_suggestions.append(suggestion)

    return vanilla_suggestions

  def _get_state(self) -> Dict[str, Any]:
    vanilla_state = {}
    vanilla_state["serialized_dna_spec"] = pg.to_json_str(self._dna_spec)  # pytype:disable=attribute-error
    vanilla_state["controller_alg_state"] = self._controller.get_state()  # pytype:disable=attribute-error
    return vanilla_state

  def _set_state(self, new_state: Dict[str, Any]) -> None:
    self._interval_counter = new_state["interval_counter"]
    self._dna_spec = pg.from_json_str(new_state["serialized_dna_spec"])
    self._controller = self._controller_fn(dna_spec=self._dna_spec)
    self._controller.set_state(new_state["controller_alg_state"])
