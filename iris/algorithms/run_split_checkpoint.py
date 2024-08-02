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

"""Splits a Blackbox v2 checkpoint into checkpoints per agent."""

from collections.abc import Sequence

from absl import app
from absl import flags
from iris import normalizer
from iris.algorithms import ars_algorithm

_NUM_AGENTS = flags.DEFINE_integer('num_agents', 2, 'Number of agents.')
_HAS_OBS_NORM = flags.DEFINE_boolean(
    'has_obs_norm', True,
    'Whether the checkpoint has observation normalization')
_CHECKPOINT_PATH = flags.DEFINE_string(
    'checkpoint_path', None, 'Path to checkpoint.', required=True)


def split_and_save_checkpoint(checkpoint_path: str,
                              num_agents: int = 2,
                              has_obs_norm_data_buffer: bool = False) -> None:
  """Splits the checkpoint at checkpoint_path into num_agents checkpoints."""
  algo = ars_algorithm.MultiAgentAugmentedRandomSearch(
      num_suggestions=3,
      step_size=0.5,
      std=1.0,
      top_percentage=1,
      orthogonal_suggestions=True,
      quasirandom_suggestions=True,
      obs_norm_data_buffer=normalizer.MeanStdBuffer()
      if has_obs_norm_data_buffer else None,
      agent_keys=[str(i) for i in range(num_agents)],
      random_seed=7)
  algo.split_and_save_checkpoint(checkpoint_path=checkpoint_path)


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  split_and_save_checkpoint(checkpoint_path=_CHECKPOINT_PATH.value,
                            num_agents=_NUM_AGENTS.value,
                            has_obs_norm_data_buffer=_HAS_OBS_NORM.value)


if __name__ == '__main__':
  app.run(main)
