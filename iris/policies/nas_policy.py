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

"""Contains special policies which possess NAS search space to define neural networks."""

import abc
import collections
from typing import Dict, Sequence, Union

import gym
from gym.spaces import utils
from iris.policies import base_policy
import numpy as np
import pyglove as pg


class PyGlovePolicy(abc.ABC, base_policy.BasePolicy):
  """Base class for all policies involving NAS search."""

  @abc.abstractmethod
  def update_dna(self, dna: pg.DNA) -> None:
    """Should update the network's architecture given a DNA."""
    raise NotImplementedError('Abstract method')

  @property
  def dna_spec(self) -> pg.DNASpec:
    """Contains the search space definition for the network architecture."""
    raise NotImplementedError('Abstract method')


class NumpyTopologyPolicy(PyGlovePolicy):
  """Parent class for numpy-based policies."""

  def __init__(self,
               ob_space: gym.Space,
               ac_space: gym.Space,
               hidden_layer_sizes: Sequence[int],
               seed: int = 0,
               **kwargs):
    base_policy.BasePolicy.__init__(self, ob_space, ac_space)

    self._hidden_layer_sizes = hidden_layer_sizes
    self._total_nb_nodes = sum(
        self._hidden_layer_sizes) + self._ob_dim + self._ac_dim
    self._all_layer_sizes = [self._ob_dim] + list(
        self._hidden_layer_sizes) + [self._ac_dim]

    self._total_weight_parameters = self._total_nb_nodes**2
    self._total_bias_parameters = self._total_nb_nodes
    self._total_nb_parameters = self._total_weight_parameters + self._total_bias_parameters

    np.random.seed(seed)
    self._weights = np.random.uniform(
        low=-1.0, high=1.0, size=(self._total_nb_nodes, self._total_nb_nodes))
    self._biases = np.random.uniform(
        low=-1.0, high=1.0, size=self._total_nb_nodes)

    self._edge_dict = {}

  def act(self, ob: Union[np.ndarray, Dict[str, np.ndarray]]
          ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
    ob = utils.flatten(self._ob_space, ob)
    values = [0.0] * self._total_nb_nodes
    for i in range(self._ob_dim):
      values[i] = ob[i]
    for i in range(self._total_nb_nodes):
      if ((i > self._ob_dim) and (i < self._total_nb_nodes - self._ac_dim)):
        values[i] = np.tanh(values[i] + self._biases[i])
      if i in self._edge_dict:
        j_list = self._edge_dict[i]
        for j in j_list:
          t = self._weights[i][j]
          values[j] += t * values[i]

    action = np.reshape(values[len(values) - self._ac_dim:len(values)],
                        (self._ac_dim))
    action = np.tanh(action)
    action = utils.unflatten(self._ac_space, action)
    return action

  def update_weights(self, new_weights: np.ndarray) -> None:
    self._weights = np.reshape(new_weights[:self._total_weight_parameters],
                               (self._total_nb_nodes, self._total_nb_nodes))
    self._biases = new_weights[self._total_weight_parameters:]

  def get_weights(self) -> np.ndarray:
    return np.concatenate((self._weights.flatten(), self._biases.flatten()))


def list_to_edge_dict(list_of_edges):
  """Converts list of edges to adjacency list (dict) format."""
  temp_dict = collections.defaultdict(list)
  for edge_pair in list_of_edges:
    small_vertex = min(edge_pair[0], edge_pair[1])
    large_vertex = max(edge_pair[0], edge_pair[1])
    temp_dict[small_vertex].append(large_vertex)
  return temp_dict


class NumpyEdgeSparsityPolicy(NumpyTopologyPolicy):
  """This policy prunes edges in the neural network."""

  def __init__(self,
               ob_space: gym.Space,
               ac_space: gym.Space,
               hidden_layer_sizes: Sequence[int],
               hidden_layer_edge_num: Sequence[int],
               edge_sample_mode='aggregate',
               **kwargs):

    self._hidden_layer_edge_num = hidden_layer_edge_num
    self._edge_sample_mode = edge_sample_mode
    super().__init__(ob_space, ac_space, hidden_layer_sizes, **kwargs)
    self.make_all_possible_edges()
    self.make_search_space()
    self.init_topology()

  def init_topology(self):
    """Sets the edge_dict (needed for parent get_action function) to be complete."""
    init_dna = self.template.encode(next(pg.random_sample(self._search_space)))
    self.update_dna(init_dna)

  def update_dna(self, dna):
    decoded = self.template.decode(dna)

    if self._edge_sample_mode == 'independent':
      list_of_edges = []
      for sector_list_of_edges in decoded:
        list_of_edges.extend(sector_list_of_edges)
    else:
      list_of_edges = decoded

    self._edge_dict = list_to_edge_dict(list_of_edges)

  def make_all_possible_edges(self):
    if self._edge_sample_mode == 'aggregate':
      # Samples edges from a normal NN, but samples across the entire edge set.
      self._all_possible_edges = []

      chunk_index = 0
      for i in range(len(self._all_layer_sizes) - 1):
        sector_before = list(
            range(chunk_index, chunk_index + self._all_layer_sizes[i]))
        sector_after = list(
            range(
                chunk_index + self._all_layer_sizes[i], chunk_index +
                self._all_layer_sizes[i] + self._all_layer_sizes[i + 1]))
        for a in sector_before:
          for b in sector_after:
            self._all_possible_edges.append((a, b))
        chunk_index += self._all_layer_sizes[i]

    elif self._edge_sample_mode == 'independent':
      # Samples edges from a normal NN, layer by layer, each with edge sizes.
      self._all_possible_edges = []
      self._ssd_list = []

      chunk_index = 0
      for i in range(len(self._all_layer_sizes) - 1):
        layer_possible_edges = []
        sector_before = list(
            range(chunk_index, chunk_index + self._all_layer_sizes[i]))
        sector_after = list(
            range(
                chunk_index + self._all_layer_sizes[i], chunk_index +
                self._all_layer_sizes[i] + self._all_layer_sizes[i + 1]))
        for a in sector_before:
          for b in sector_after:
            layer_possible_edges.append((a, b))
            self._all_possible_edges.append((a, b))
        chunk_index += self._all_layer_sizes[i]
        ssd_i = pg.sublist_of(
            self._hidden_layer_edge_num[i],
            candidates=layer_possible_edges,
            choices_sorted=False,
            choices_distinct=True)

        self._ssd_list.append(ssd_i)
    elif self._edge_sample_mode == 'residual':
      # Allows residual connections between all hidden layer pairs.
      self._all_possible_edges = []

      for i in range(len(self._all_layer_sizes) - 1):
        for j in range(i + 1, len(self._all_layer_sizes)):
          sector_before = list(
              range(
                  sum(self._all_layer_sizes[0:i]),
                  sum(self._all_layer_sizes[0:i + 1])))
          sector_after = list(
              range(
                  sum(self._all_layer_sizes[0:j]),
                  sum(self._all_layer_sizes[0:j + 1])))

          for a in sector_before:
            for b in sector_after:
              self._all_possible_edges.append((a, b))
    else:
      raise ValueError('Edge sample mode not in allowed set.')

  def make_search_space(self):
    if self._edge_sample_mode == 'aggregate':
      total_number_k = sum(self._hidden_layer_edge_num)
      self._search_space = pg.sublist_of(
          total_number_k,
          candidates=self._all_possible_edges,
          choices_sorted=False,
          choices_distinct=True)
    elif self._edge_sample_mode == 'independent':
      self._search_space = pg.List(self._ssd_list)
    elif self._edge_sample_mode == 'residual':
      total_number_k = sum(self._hidden_layer_edge_num)
      self._search_space = pg.sublist_of(
          total_number_k,
          candidates=self._all_possible_edges,
          choices_sorted=False,
          choices_distinct=True)
    else:
      raise ValueError('Edge sample mode not in allowed set.')
    self.template = pg.template(self._search_space)

  @property
  def dna_spec(self):
    return self.template.dna_spec()
