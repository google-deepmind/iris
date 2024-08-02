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

"""Tests for controllers."""

import random
from iris.algorithms.controllers import regularized_evolution_controller
import pyglove as pg
from absl.testing import absltest


class ControllersTest(absltest.TestCase):

  def test_checkpointing(self):
    example_dna_spec = pg.dna_spec(pg.one_of(['a', 'b', 'c']))
    batch_size = 4
    seed = 0
    controller = regularized_evolution_controller.RegularizedEvolutionController(
        example_dna_spec, batch_size=batch_size, seed=seed)

    for _ in range(20):
      dna = controller.propose_dna()
      controller.collect_rewards_and_train([random.random()], [dna])

    another_controller = regularized_evolution_controller.RegularizedEvolutionController(
        example_dna_spec, batch_size=batch_size, seed=seed)
    another_controller.set_state(controller.get_state())

    feedback_in_controller = [
        (dna, pg.evolution.base.get_fitness(dna))
        for dna in list(controller._controller._population)
    ]

    feedback_in_another_controller = [
        (dna, pg.evolution.base.get_fitness(dna))
        for dna in list(another_controller._controller._population)
    ]
    self.assertEqual(feedback_in_controller, feedback_in_another_controller)
    self.assertEqual(controller.propose_dna(), another_controller.propose_dna())


if __name__ == '__main__':
  absltest.main()
