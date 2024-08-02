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

"""Regularized Evolution controller from PyGlove."""
# pylint: disable=protected-access
from typing import Optional
from iris.algorithms.controllers import base_controller
import numpy as np
import pyglove as pg


class RegularizedEvolutionController(base_controller.BaseController):
  """Regularized Evolution Controller."""

  def __init__(self,
               dna_spec: pg.DNASpec,
               batch_size: int,
               seed: Optional[int] = None,
               **kwargs):
    """Initialization. See base class for more details."""

    super().__init__(dna_spec, batch_size)
    # Hyperparameters copied from example colab:
    # http://pyglove/generators/evolution_example.ipynb
    population_size = self._batch_size
    tournament_size = int(np.sqrt(population_size))

    self._controller = pg.evolution.RegularizedEvolution(
        population_size=population_size,
        tournament_size=tournament_size,
        mutator=pg.evolution.mutators.Uniform(seed=seed),
        seed=seed)  # pytype: disable=wrong-arg-types  # gen-stub-imports
    self._controller.setup(self._dna_spec)

  def get_state(self):
    return pg.to_json_str(self._history)  # pytype: disable=attribute-error

  def set_state(self, serialized_state):  # pytype: disable=signature-mismatch  # overriding-parameter-count-checks
    self._history = pg.from_json_str(serialized_state)
    self._controller.recover(self._history)
