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

"""Implicit EBM-policies.

Implicit policies defined via energy-based models (EBMs). The policy returns an
action as the solution to the energy minimization problem, where the energy is
a function of the dot-product of the learnable latent representations of the
state and action. Alternatively, an action is sampled from the distribution that
prioritizes actions corresponding to lower energies (for a given state).
The library contains also a model where action is concatenated with state before
being fed to one tower producing corresponding energy value (as a baseline for
comparison).

"""

from typing import Dict, List, Sequence, Union
import gym
from gym.spaces import utils
from iris.policies import base_policy
import numpy as np


class BaseEnergy(object):
  """Calculates energy for the given pair (state, action)."""

  def update_weights(self, new_weights: np.ndarray) -> bool:
    """Tries to update energy-weights.

    Returns True if all weights are updated.

    Updates the energy-weights with the proposed weights <new_weights> if the
    energy-object is in the update-mode. Returns boolean indicating whether
    weights were updated.

    Args:
      new_weights: new weights to update energy-object

    Returns:
      True if the weights weere updated with <new_weights> and False otherwise.
    """
    raise NotImplementedError(
        "Should be implemented in derived classes for specific energies.")

  def get_weights(self) -> np.ndarray:
    """Returns energy weights."""
    raise NotImplementedError(
        "Should be implemented in derived classes for specific energies.")

  def energy(self, state: np.ndarray, action: np.ndarray) -> float:
    """Main method calculating energy for the given pair (state, action)."""
    raise NotImplementedError(
        "Should be implemented in derived classes for specific energies.")

  def linearized_energy_state(self,
                              state: np.ndarray) -> Union[np.ndarray, None]:
    r"""Main method calculating state representat.

    for the linearized energy.

    Returns the representation phi(s) of the state in the given linearization of
    the energy function of the form: E(s,a) ~ phi(s)^T \phi(a).

    Args:
      state: state np.array

    Returns:
      representation phi(s) of the state in the given linearization of the
      energy function and None if the linearization is not provided.
    """
    raise NotImplementedError(
        "Should be implemented in derived classes for specific energies.")

  def linearized_energy_action(self,
                               action: np.ndarray) -> Union[np.ndarray, None]:
    r"""Main method calculating action representat.

    for the linearized energy.

    Returns the representation phi(a) of the action in the given linearization
    of the energy function of the form: E(s,a) ~ phi(s)^T \phi(a).

    Args:
      action: action np.array

    Returns:
      representation phi(a) of the action in the given linearization of the
      energy function and None if the linearization is not provided.
    """
    raise NotImplementedError(
        "Should be implemented in derived classes for specific energies.")


class BaseTower(object):
  """Base tower producing latent representations of states/actions."""

  def __init__(self, input_dim: int, latent_dim: int) -> None:
    """Initializes base tower.

    Args:
      input_dim: Dimensionality of the tower input.
      latent_dim: Dimensionality of the tower output (latent representation).
    """
    self._input_dim = input_dim
    self._latent_dim = latent_dim
    self._weights = np.empty(0)
    self._num_weights = 0

  def update_weights(self, new_weights: np.ndarray) -> bool:
    self._weights[:] = new_weights[:]
    return True

  def get_weights(self) -> np.ndarray:
    return self._weights

  def get_num_weights(self) -> int:
    return self._num_weights

  def latent_rep(self, input_vector: np.ndarray) -> np.ndarray:
    """Maps the input state/action to its corresponding latent representation."""
    raise NotImplementedError(
        "Should be implemented in derived classes for specific towers.")


class BaseActionCalculator(object):
  """Calculates an action that will be assigned to the given state."""

  def __init__(self, energy: BaseEnergy, low: np.ndarray,
               high: np.ndarray) -> None:
    """Initializes base action calculator.

    Args:
      energy: energy object.
      low: numpy array of the lower bounds for the values of actions' dims
      high: numpy array of the upper bounds for the values of actions' dims
    """
    self._energy = energy
    self._low = low
    self._high = high

  def update_weights(self, new_weights: np.ndarray) -> None:
    """Updates the weights of the calculator."""
    self._energy.update_weights(new_weights)

  def get_weights(self) -> np.ndarray:
    """Gets the weights of the calculator."""
    return self._energy.get_weights()

  def act(self, state: np.ndarray) -> np.ndarray:
    """Calculates the action for a given state."""
    raise NotImplementedError(
        "Should be implemented in derived classes for specific towers.")


class OneTowerEnergy(BaseEnergy):
  """Computes energy as the output of the tower taking concat. state/action."""

  def __init__(self, tower: BaseTower):
    self._tower = tower

  def update_weights(self, new_weights: np.ndarray) -> bool:
    self._tower.update_weights(new_weights)
    return True

  def get_weights(self) -> np.ndarray:
    return self._tower.get_weights()

  def energy(self, state: np.ndarray, action: np.ndarray) -> float:
    return self._tower.latent_rep(np.concatenate((state, action)))[0]

  def linearized_energy_state(self,
                              state: np.ndarray) -> Union[np.ndarray, None]:
    """Not applicable."""
    return None

  def linearized_energy_action(self,
                               action: np.ndarray) -> Union[np.ndarray, None]:
    """Not applicable."""
    return None


class TwoTowersEnergy(BaseEnergy):
  """Computes energy as the output of the two-tower model.

  Computes the energy as the output of the two-tower model with action-tower
  producing the latent representation of the action, state-tower producing the
  latent representation of the state and energy defined as some function of the
  dot-product of these two latent representations.
  """

  def __init__(self, state_tower: BaseTower, action_tower: BaseTower,
               alpha_act: float):
    self._state_tower = state_tower
    self._action_tower = action_tower
    self._alpha_act = alpha_act
    self._time = 0

  def update_weights(self, new_weights: np.ndarray) -> bool:
    self._state_tower.update_weights(
        new_weights[0:self._state_tower.get_num_weights()])
    ac_upd_prob = np.exp(-self._alpha_act * self._time)
    self._time += 1
    np.random.seed(self._time)
    if np.random.uniform() < ac_upd_prob:
      self._action_tower.update_weights(
          new_weights[self._state_tower.get_num_weights():])
      return True
    return False

  def get_weights(self) -> np.ndarray:
    state_tower_weights = self._state_tower.get_weights()
    action_tower_weights = self._action_tower.get_weights()
    return np.concatenate([state_tower_weights, action_tower_weights])

  def energy(self, state: np.ndarray, action: np.ndarray) -> float:
    raise NotImplementedError(
        "Should be implemented in derived classes for specific two-tower-mods.")

  def linearized_energy_state(self,
                              state: np.ndarray) -> Union[np.ndarray, None]:
    raise NotImplementedError(
        "Should be implemented in derived classes for specific two-tower-mods.")

  def linearized_energy_action(self,
                               action: np.ndarray) -> Union[np.ndarray, None]:
    raise NotImplementedError(
        "Should be implemented in derived classes for specific two-tower-mods.")


class NegatedDotProductEnergy(TwoTowersEnergy):
  """Computes energy as negated dot-product of latent state/action represent."""

  def __init__(self,
               state_tower: BaseTower,
               action_tower: BaseTower,
               alpha_act: float = 0):
    super().__init__(state_tower, action_tower, alpha_act)

  def energy(self, state: np.ndarray, action: np.ndarray) -> float:
    latent_state_rep = self._state_tower.latent_rep(state)
    latent_action_rep = self._action_tower.latent_rep(action)
    return 0.0 - np.dot(latent_state_rep, latent_action_rep)

  def linearized_energy_state(self,
                              state: np.ndarray) -> Union[np.ndarray, None]:
    return self._state_tower.latent_rep(state)

  def linearized_energy_action(self,
                               action: np.ndarray) -> Union[np.ndarray, None]:
    return 0.0 - self._action_tower.latent_rep(action)


def positive_random_features(input_vector: np.ndarray,
                             nb_rfs: int) -> np.ndarray:
  """Computes positive random features corresponding to the softmax kernel.

  Args:
    input_vector: vector for which random features are computed.
    nb_rfs: number of random features used.

  Returns:
    positive random features form the "Rethinking Attention with Performers"
    paper for the <input_vector>.
  """
  np.random.seed(0)
  proj_matrix = np.random.normal(size=(len(input_vector), nb_rfs))
  proj_vector = np.dot(input_vector, proj_matrix)
  rfs_vector = np.exp(proj_vector)
  vector_norm = np.linalg.norm(input_vector)
  vector_squared__norm = vector_norm * vector_norm
  det_factor = np.exp(0.0 - vector_squared__norm / 2.0)
  rfs_vector = (1.0 / np.sqrt(float(nb_rfs))) * det_factor * rfs_vector
  return rfs_vector


class NegatedSoftmaxEnergy(TwoTowersEnergy):
  """Computes energy as negated renormalized softmax kernel value."""

  def __init__(self, state_tower: BaseTower, action_tower: BaseTower,
               beta: float, nb_rfs: int, alpha_act: float):
    super().__init__(state_tower, action_tower, alpha_act)
    self._beta = beta
    self._nb_rfs = nb_rfs

  def energy(self, state: np.ndarray, action: np.ndarray) -> float:
    latent_state_rep = self._state_tower.latent_rep(state)
    latent_action_rep = self._action_tower.latent_rep(action)
    return 0.0 - np.exp(
        self._beta * np.dot(latent_state_rep, latent_action_rep))

  def linearized_energy_state(self,
                              state: np.ndarray) -> Union[np.ndarray, None]:
    return positive_random_features(
        np.sqrt(self._beta) * self._state_tower.latent_rep(state), self._nb_rfs)

  def linearized_energy_action(self,
                               action: np.ndarray) -> Union[np.ndarray, None]:
    return 0.0 - positive_random_features(
        np.sqrt(self._beta) * self._action_tower.latent_rep(action),
        self._nb_rfs)


class NeuralNetworkTower(BaseTower):
  """Single neural-network-encoded tower for state/action."""

  def __init__(self,
               input_dim: int,
               latent_dim: int,
               hidden_layer_sizes: Sequence[int],
               activation: str = "tanh",
               normalize_output: bool = False) -> None:
    """Initializes a tower encoded by the neural network."""

    super().__init__(input_dim, latent_dim)
    self._input_dim = input_dim
    self._latent_dim = latent_dim
    self._hidden_layer_sizes = hidden_layer_sizes
    self._normalize_output = normalize_output
    if activation == "tanh":
      self._activation = np.tanh
    elif activation == "relu":
      self._activation = lambda x: np.maximum(x, 0.0)
    elif activation == "clip":
      self._activation = lambda x: np.clip(x, -1.0, 1.0)
    else:
      raise ValueError("Non-supported nonlinearity.")

    self._layer_sizes = [self._input_dim]
    self._layer_sizes.extend(self._hidden_layer_sizes)
    self._layer_sizes.append(self._latent_dim)
    self._layer_weight_start_idx = []
    self._layer_weight_end_idx = []
    num_weights = 0
    num_layers = len(self._layer_sizes)
    for ith_layer in range(num_layers - 1):
      self._layer_weight_start_idx.append(num_weights)
      num_weights += (
          self._layer_sizes[ith_layer] * self._layer_sizes[ith_layer + 1])
      self._layer_weight_end_idx.append(num_weights)
    self._num_weights = num_weights
    self._weights = np.zeros(num_weights, dtype=np.float64)

  def latent_rep(self, input_vector: np.ndarray) -> np.ndarray:
    ith_layer_result = input_vector
    num_layers = len(self._layer_sizes)
    for ith_layer in range(num_layers - 1):
      start = self._layer_weight_start_idx[ith_layer]
      end = self._layer_weight_end_idx[ith_layer]
      mat_weight = np.reshape(
          self._weights[start:end],
          (self._layer_sizes[ith_layer + 1], self._layer_sizes[ith_layer]))
      ith_layer_result = np.dot(mat_weight, ith_layer_result)
      ith_layer_result = self._activation(ith_layer_result)
    latent_rep = ith_layer_result
    if self._normalize_output:
      latent_rep /= np.linalg.norm(latent_rep)
    return latent_rep


def update_latent_reps_for_actions(actions: List[np.ndarray],
                                   energy: BaseEnergy) -> np.ndarray:
  """Updates the latent representations of sampled actions.

  Args:
    actions: sampled actions
    energy: used energy-object

  Returns:
    new latent representations for actions.
  """
  lat_reps_for_actions = []
  for i in range(len(actions)):
    phi_action = energy.linearized_energy_action(actions[i])
    if phi_action is None:
      raise ValueError("The linearization of the energy is not provided.")
    lat_reps_for_actions.append(-phi_action)
  return np.array(lat_reps_for_actions)


class MinEnergyActionCalculator(BaseActionCalculator):
  """Returns sampled action corresponding to the smallest energy.

  Returns this sampled action for which the energy for the (state, action) pair
  is minimized.
  """

  def __init__(self,
               energy: BaseEnergy,
               low: np.ndarray,
               high: np.ndarray,
               num_samples: int,
               bootstrapped_samples: int = 0) -> None:
    """Initializes MinEnergyActionCalculator."""

    super().__init__(energy, low, high)
    self._num_samples = num_samples
    self._bootstrapped_samples = bootstrapped_samples
    if bootstrapped_samples:
      actions = []
      np.random.seed(0)
      for _ in range(self._num_samples):
        action = np.random.uniform(
            low=self._low, high=self._high, size=(len(self._low)))
        actions.append(action)
      self._actions = actions
      self._lat_reps_for_actions = update_latent_reps_for_actions(
          self._actions, self._energy)

  def update_weights(self, new_weights: np.ndarray) -> None:
    all_weights_updated = self._energy.update_weights(new_weights)
    if all_weights_updated and self._bootstrapped_samples:
      self._lat_reps_for_actions = update_latent_reps_for_actions(
          self._actions, self._energy)

  def act(self, state: np.ndarray) -> np.ndarray:
    if not self._bootstrapped_samples:
      best_energy = float("inf")
      best_action = np.zeros(len(self._low))
      for _ in range(self._num_samples):
        action = np.random.uniform(
            low=self._low, high=self._high, size=(len(self._low)))
        current_energy = self._energy.energy(state, action)
        if current_energy < best_energy:
          best_energy = current_energy
          best_action = action
      return best_action
    else:
      phi_state = self._energy.linearized_energy_state(state)
      if self._bootstrapped_samples == self._num_samples:
        return self._actions[np.argmax(
            np.dot(self._lat_reps_for_actions, phi_state))]
      else:
        random_indices = np.random.choice(np.arange(len(self._actions)))
        return self._actions[random_indices[np.argmax(
            np.dot(self._lat_reps_for_actions[random_indices], phi_state))]]


def update_rft_for_actions(actions: List[np.ndarray],
                           energy: BaseEnergy) -> List[np.ndarray]:
  """Updates the random feature tree structure of sampled actions.

  Updates the random feeature tree structure (of sampled actions) encoded by the
  prefix-sum list.

  Args:
    actions: sampled actions
    energy: used energy-object

  Returns:
    new random feature tree encoded by the prefix-sum list.
  """
  prefix_sum_table = []
  for i in range(len(actions)):
    phi_action = energy.linearized_energy_action(actions[i])
    if phi_action is None:
      raise ValueError("The linearization of the energy is not provided.")
    if not prefix_sum_table:
      prefix_sum_table.append(-phi_action)
    else:
      prefix_sum_table.append(-phi_action + prefix_sum_table[-1])
  return prefix_sum_table


class SoftmaxEnergyRFSActionCalculator(BaseActionCalculator):
  """Returns softmax-sampled action via Random FeatureS tree.

  Returns an action sampled proportionally to the softmax-kernel-defined energy
  with random feature trees used for state-action pairing in logarithmic time.
  """

  def __init__(self, energy: BaseEnergy, low: np.ndarray, high: np.ndarray,
               num_samples: int, search_tree_deg: int,
               nb_tree_search_repeats: int) -> None:
    """Initializes SoftmaxEnergyRFSActionCalculator."""

    super().__init__(energy, low, high)
    self._num_samples = num_samples
    self._search_tree_deg = search_tree_deg
    self._nb_tree_search_repeats = nb_tree_search_repeats
    actions = []
    np.random.seed(0)
    for _ in range(self._num_samples):
      action = np.random.uniform(
          low=self._low, high=self._high, size=(len(self._low)))
      actions.append(action)
    self._actions = actions
    self._prefix_sum_table = update_rft_for_actions(self._actions, self._energy)

  def update_weights(self, new_weights: np.ndarray) -> None:
    all_weights_updated = self._energy.update_weights(new_weights)
    if all_weights_updated:
      self._prefix_sum_table = update_rft_for_actions(self._actions,
                                                      self._energy)

  def act(self, state: np.ndarray) -> np.ndarray:
    phi_state = self._energy.linearized_energy_state(state)
    hit_table = [0] * len(self._actions)
    for _ in range(self._nb_tree_search_repeats):
      low_index = 0
      high_index = len(self._prefix_sum_table)
      while low_index < high_index - 1:
        length = high_index - low_index
        seg_length = np.maximum(1, int(length / self._search_tree_deg))
        probs = []
        start_end_indices = []
        seg_start_index = low_index
        seg_end_index = low_index + seg_length
        while seg_start_index <= high_index - 1:
          base_prefix_sum = 0.0
          if seg_start_index > 0:
            base_prefix_sum = self._prefix_sum_table[seg_start_index - 1]
          prob = np.dot(
              self._prefix_sum_table[seg_end_index - 1] - base_prefix_sum,
              phi_state)
          probs.append(prob)
          start_end_indices.append([seg_start_index, seg_end_index])
          seg_start_index = seg_end_index
          seg_end_index = np.minimum(seg_end_index + seg_length, high_index)
        probs = [x / sum(probs) for x in probs]
        low_index, high_index = start_end_indices[np.random.choice(
            np.arange(len(start_end_indices)), p=probs)]
      hit_table[low_index] += 1
    return self._actions[hit_table.index(max(hit_table))]


class ImplicitEBMPolicy(base_policy.BasePolicy):
  """Implicit EBM policy."""

  def __init__(self, ob_space: gym.Space, ac_space: gym.Space,
               action_calculator: BaseActionCalculator) -> None:
    """Initializes the implicit EBM policy."""

    super().__init__(ob_space, ac_space)
    self._action_calculator = action_calculator

  def update_weights(self, new_weights: np.ndarray) -> None:
    self._action_calculator.update_weights(new_weights)

  def get_weights(self) -> np.ndarray:
    return self._action_calculator.get_weights()

  def act(
      self, ob: Union[np.ndarray, Dict[str, np.ndarray]]
  ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
    """Maps the observation/state to action.

    Args:
      ob: The observations in reinforcement learning.

    Returns:
      The actions in reinforcement learning.
    """
    ob = utils.flatten(self._ob_space, ob)
    action = self._action_calculator.act(ob)
    action = utils.unflatten(self._ac_space, action)
    return action
