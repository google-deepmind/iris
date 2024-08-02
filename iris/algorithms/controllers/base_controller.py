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

"""Base class for all controllers in ES-ENAS."""

import abc
from typing import List, Optional

import pyglove as pg


class BaseController(abc.ABC):
  """Base class for all controllers in ES-ENAS."""

  def __init__(self, dna_spec: pg.DNASpec,
               batch_size: int) -> None:
    """Initialization.

    Args:
      dna_spec: A search space definition for the controller to use.
      batch_size: Number suggestions in a current iteration.
    """
    self._dna_spec = dna_spec
    self._batch_size = batch_size
    self._controller = pg.DNAGenerator()
    self._history = []

  def propose_dna(self) -> pg.DNA:
    """Proposes a topology dna using stored template.

    Args: None.

    Returns:
      dna: A proposed dna.
    """
    return self._controller.propose()

  def collect_rewards_and_train(self, reward_vector: List[float],
                                dna_list: List[pg.DNA]):
    """Collects rewards to update the controller.

    Args:
      reward_vector: list of reward floats.
      dna_list: list of dna's from the proposal function.

    Returns:
      None.
    """

    for i, dna in enumerate(dna_list):
      dna.reward = reward_vector[i]
      self._controller.feedback(dna, dna.reward)
      self._history.append((dna, dna.reward))

  @abc.abstractmethod
  def get_state(self) -> Optional[str]:
    """Returns serialized version of controller algorithm state.

    Serialization is required for compatibility with iris states.

    Returns:
      Serialized state in string format.
    """

  @abc.abstractmethod
  def set_state(self, serialized_state: Optional[str] = None) -> None:
    """Sets the controller algorithm state from a serialized state.

    Args:
      serialized_state: State, serialized in string format.
    """
