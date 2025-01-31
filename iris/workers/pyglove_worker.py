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

"""Class for evaluating a given blackbox function with an entire PyGlove search space."""

from typing import Callable

from absl import logging
from iris.workers import worker
from iris.workers import worker_util
import numpy as np
import pyglove as pg


class PyGloveWorker(worker.Worker):
  """Class for evaluating a given blackbox function with an entire PyGlove search space.

  NOTE: This only works with PyGloveAlgorithm.

  Continuous search spaces are offloaded to PyGlove `floatv()` symbolics.
  Useful for evaluating performance of end-to-end evolutionary algorithms like
  NEAT and Reg-Evo.
  """

  def __init__(
      self,
      dna_spec: pg.DNASpec,
      blackbox_function: Callable[[pg.DNA], worker.FloatLike],
      **kwargs
  ) -> None:
    super().__init__(**kwargs)
    self._init_state["serialized_dna_spec"] = pg.to_json_str(dna_spec)
    self._blackbox_function = blackbox_function

  def work(  # pytype: disable=signature-mismatch  # overriding-parameter-count-checks
      self,
      metadata: str,
      params_to_eval: np.ndarray,  # Ignored.
      enable_logging: bool = False,
  ) -> worker_util.EvaluationResult:
    """Runs the blackbox function on DNA."""
    dna = pg.from_json_str(metadata)
    value = self._blackbox_function(dna)
    if enable_logging:
      logging.info("Value: %f", value)
    evaluation_result = worker_util.EvaluationResult(
        params_to_eval, value, metadata=metadata
    )
    return evaluation_result
