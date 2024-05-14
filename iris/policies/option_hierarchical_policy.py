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

"""A policy with options style temporally abstracted hierarchical structure."""

from typing import Any, Callable, Dict, Sequence, Union

import gym
from gym.spaces import utils
from iris.policies import base_policy
from iris.policies import gym_space_utils
import numpy as np


class OptionHierarchicalLevel(object):
  """Level of a options style hierarchical policy.

  A level gets an array of latent commands and observations as input. The
  output is another array of latent commands for next level. For highest level,
  there are no input commands and for lowest level, the output latent commands
  are the final actions. A level decides whether to terminate and request new
  input based on the current input + observation.
  """

  def __init__(self,
               idx: int,
               ob_space: gym.Space,
               in_command_dim: int,
               out_command_dim: int,
               policy: Callable[..., base_policy.BasePolicy],
               selected_observations: Sequence[Union[str, int]]) -> None:
    """Initializes the hierarchical level.

    Args:
      idx: Index of this level of hierarchy. Starting from 0 which indicates
        the highest level.
      ob_space: Input observation space.
      in_command_dim: Number of dimensions in input latent command
      out_command_dim: Number of dimensions in output latent command
      policy: Policy to map observations to output for this level.
      selected_observations: A list of sensors or dimensions to include for this
        level.
    """
    self.idx = idx
    self._in_command_dim = in_command_dim
    self._out_command_dim = out_command_dim
    if self.idx > 0:
      self._out_command_dim += 1
    self._selected_observations = selected_observations
    self._selected_ob_space = gym_space_utils.filter_space(
        ob_space, self._selected_observations)
    self._ob_space = gym_space_utils.extend_space(
        self._selected_ob_space,
        "in_command",
        gym.spaces.Box(-1, 1, (self._in_command_dim,)))
    self._ac_space = gym.spaces.Box(-1, 1, (self._out_command_dim,))
    self._terminate = True
    self._output = np.zeros(self._out_command_dim)
    self.policy = policy(ob_space=self._ob_space,
                         ac_space=self._ac_space)

  def reset(self):
    self._terminate = True
    self.policy.reset()

  def terminated(self):
    return self._terminate

  def __call__(self,
               ob: Union[np.ndarray, Dict[str, np.ndarray]],
               in_command: np.ndarray,
               terminated: int) -> Any:
    ob = ob.copy()
    if terminated:
      ob = gym_space_utils.filter_sample(ob, self._selected_observations)
      ob = gym_space_utils.extend_sample(ob, "in_command", in_command)
      self._output = self.policy.act(ob)
      if self.idx > 0:
        self._terminate = bool(np.random.binomial(1, (self._output[0] + 1.)/2.))
        self._output = self._output[1:]
    return self._output


class OptionHierarchicalPolicy(base_policy.BasePolicy):
  """Options style hierarchical policy."""

  def __init__(self,
               ob_space: gym.Space,
               ac_space: gym.Space,
               level_params: Sequence[Dict[str, Any]]):
    """Initializes the hierarchical policy.

    This method initializes all levels and stores them in a list.
    Initial weights of the policy are calculated by concatenating weights of all
    levels.

    Args:
      ob_space: Input observation space.
      ac_space: Output action space.
      level_params: A list of parameters for all levels of hierarchy.
    """
    super().__init__(ob_space, ac_space)
    self.levels = []
    num_levels = len(level_params)
    for idx, params in enumerate(level_params):
      if idx == 0:
        params["in_command_dim"] = 0
      else:
        params["in_command_dim"] = level_params[idx-1]["out_command_dim"]
      if idx == num_levels - 1:
        params["out_command_dim"] = self._ac_dim
      self.levels.append(OptionHierarchicalLevel(idx, ob_space, **params))

  def get_weights(self) -> np.ndarray:
    weights = np.array([])
    for level in self.levels:
      weights = np.concatenate((weights, level.policy.get_weights()))
    return weights

  def update_weights(self, new_weights: np.ndarray) -> None:
    start = 0
    for level in self.levels:
      weight_size = len(level.policy.get_weights())
      level.policy.update_weights(new_weights[start:start + weight_size])
      start += weight_size

  def reset(self):
    for level in self.levels:
      level.reset()

  def act(self, ob: Union[np.ndarray, Dict[str, np.ndarray]]
          ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
    """Maps the observation to action.

    Args:
      ob: The observations in reinforcement learning.
    Returns:
      The actions in reinforcement learning.
    """
    command = np.empty(0)
    for idx, level in enumerate(self.levels):
      if idx == len(self.levels)-1:
        terminated = True
      else:
        terminated = self.levels[idx+1].terminated()
      command = level(ob, command, terminated)
    action = utils.unflatten(self._ac_space, command)
    return action
