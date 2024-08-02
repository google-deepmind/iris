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

"""NEAT Controller from PyGlove."""
from typing import Optional
from iris.algorithms.controllers import base_controller
import pyglove as pg


class NEATController(base_controller.BaseController):
  """NEAT Controller."""

  def __init__(self,
               dna_spec: pg.DNASpec,
               batch_size: int,
               seed: Optional[int] = None,
               **kwargs):
    """Initialization. See base class for more details."""

    super().__init__(dna_spec, batch_size)
    population_size = self._batch_size
    self._controller = pg.evolution.neat(
        population_size=population_size,
        mutator=pg.evolution.mutators.Uniform(),
        seed=seed)  # pytype: disable=wrong-arg-types  # gen-stub-imports
    self._controller.setup(self._dna_spec)

  def get_state(self):
    # TODO: Add checkpointing logic for NEAT.
    return None

  def set_state(self, serialized_state):  # pytype: disable=signature-mismatch  # overriding-parameter-count-checks
    # TODO: See above.
    pass
