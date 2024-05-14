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

"""A policy with temporally abstracted hierarchical structure."""

from typing import Any, Callable, Dict, Optional, Sequence, Union
import gym
from gym.spaces import utils
from iris.policies import base_policy
from iris.policies import gym_space_utils
import numpy as np


class HierarchicalLevel(object):
  """Level of a hierarchical policy.

  A level gets an array of latent commands and observations as input. The
  output is another array of latent commands for next level. For highest level,
  there are no input commands and for lowest level, the output latent commands
  are the final actions. The level can act at a fixed or variable timescale.
  For each level, part of observation space could be hidden.
  """

  def __init__(
      self,
      idx: int,
      ob_space: gym.Space,
      in_command_dim: int,
      out_command_dim: int,
      policy: Callable[..., base_policy.BasePolicy],
      selected_observations: Sequence[Union[str, int]],
      fixed_timescale: Optional[int] = None,
      timescale_range: Optional[Sequence[int]] = None,
  ) -> None:
    """Initializes the hierarchical level.

    Args:
      idx: Index of this level of hierarchy. Starting from 0 which indicates the
        highest level.
      ob_space: Input observation space.
      in_command_dim: Number of dimensions in input latent command
      out_command_dim: Number of dimensions in output latent command
      policy: Policy to map observations to output for this level.
      selected_observations: A list of sensors or dimensions to iclude for this
        level.
      fixed_timescale: Number of control timesteps which defines a fixed
        interval at which this level of hierarchical will be invoked. If
        fixed_timescale is None, then a variable interval will be used for this
        level. The first dimension of output command will be used to calculate
        the variable interval and will not be sent to the next level.
      timescale_range: If fixed_timescale is None, then timescale_range for the
        variable interval has to be defined. It is a tuple of the form
        (min_interval, max_interval).
    """
    self.idx = idx
    self._in_command_dim = in_command_dim
    self._out_command_dim = out_command_dim
    self._selected_observations = selected_observations
    self._selected_ob_space = gym_space_utils.filter_space(
        ob_space, self._selected_observations
    )
    self._ob_space = gym_space_utils.extend_space(
        self._selected_ob_space,
        "in_command",
        gym.spaces.Box(-1, 1, (self._in_command_dim,)),
    )
    self._ac_space = gym.spaces.Box(-1, 1, (self._out_command_dim,))
    self._timescale = fixed_timescale
    if self._timescale is None:
      self._timescale_low, self._timescale_high = timescale_range
    self._act_after_steps = 0
    self._output = np.zeros(self._out_command_dim)
    self.policy = policy(ob_space=self._ob_space, ac_space=self._ac_space)

  def reset(self):
    self._act_after_steps = 0
    self.policy.reset()

  def __call__(
      self, ob: Union[np.ndarray, Dict[str, np.ndarray]], in_command: np.ndarray
  ) -> Any:
    ob = ob.copy()
    if not self._act_after_steps:
      ob = gym_space_utils.filter_sample(ob, self._selected_observations)
      ob = gym_space_utils.extend_sample(ob, "in_command", in_command)
      self._output = self.policy.act(ob)
      if self._timescale is not None:
        self._act_after_steps = self._timescale
      else:
        self._act_after_steps = int(
            self._output[0] * (self._timescale_high - self._timescale_low) / 2.0
            + (self._timescale_high + self._timescale_low) / 2.0
        )
        self._output = self._output[1:]
    self._act_after_steps -= 1
    return self._output


class HierarchicalPolicy(base_policy.BasePolicy):
  """Latent command space hierarchical policy.

  See: https://arxiv.org/abs/1905.08926.
  """

  def __init__(
      self,
      ob_space: gym.Space,
      ac_space: gym.Space,
      level_params: Sequence[Dict[str, Any]],
  ):
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
      if idx == num_levels - 1:
        params["out_command_dim"] = self._ac_dim
      self.levels.append(HierarchicalLevel(idx, ob_space, **params))

  def get_weights(self) -> np.ndarray:
    weights = np.array([])
    for level in self.levels:
      weights = np.concatenate((weights, level.policy.get_weights()))
    return weights

  def update_weights(self, new_weights: np.ndarray) -> None:
    start = 0
    for level in self.levels:
      weight_size = len(level.policy.get_weights())
      level.policy.update_weights(new_weights[start : start + weight_size])
      start += weight_size

  def reset(self):
    for level in self.levels:
      level.reset()

  def act(
      self, ob: Union[np.ndarray, Dict[str, np.ndarray]]
  ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
    """Maps the observation to action.

    Args:
      ob: The observations in reinforcement learning.

    Returns:
      The actions in reinforcement learning.
    """
    command = np.empty(0)
    for level in self.levels:
      command = level(ob, command)
    action = utils.unflatten(self._ac_space, command)
    return action
