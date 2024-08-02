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

"""This an updated variant of the original policy gradient-based MetaArchitect controller."""
from iris.algorithms.controllers import base_controller
import pyglove as pg


class PolicyGradientController(base_controller.BaseController):
  """Policy Gradient Controller."""

  def __init__(self,
               dna_spec: pg.DNASpec,
               batch_size: int,
               update_batch_size=64,
               **kwargs):
    """Initialization. See base class for more details."""

    super().__init__(dna_spec, batch_size)
    self._controller = pg.reinforcement_learning.PPO(    # pytype: disable=module-attr
        train_batch_size=self._batch_size, update_batch_size=update_batch_size)
    self._controller.setup(self._dna_spec)
    # If you have:
    # training batch size N (PG proposes a batch N of models, stored in cache)
    # update batch size M, (minibatch update batch size)
    # num. of updates P, (how many minibatch updates)
    # the update rule is:
    #
    # for _ in range(P):
    #  mini_batch = select(M, N)
    #  train(model, mini_batch)

  def get_state(self):
    # TODO: See pyglove policy_gradients generator for previous
    # implementations.
    return None

  def set_state(self, serialized_state):  # pytype: disable=signature-mismatch  # overriding-parameter-count-checks
    # TODO: See above.
    pass
