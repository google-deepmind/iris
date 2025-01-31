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

"""Example configuration for Iris experiments."""

from iris.algorithms import ars_algorithm
from iris.workers import simple_worker
from ml_collections import config_dict
import numpy as np


def get_coordinator_config():
  """Coordinator config."""

  return config_dict.ConfigDict(
      dict(
          save_rate=10,
          eval_rate=1,
          num_iterations=400,
          num_evals_per_suggestion=1))


def get_worker_config():
  """Worker config."""

  return config_dict.ConfigDict(
      dict(
          worker_class=simple_worker.SimpleWorker,
          worker_args={
              'initial_params': np.ones(10),
              'blackbox_function': lambda x: -1 * np.sum(x**2),
          },
      )
  )


def get_algo_config():
  """Algorithm config."""

  return config_dict.ConfigDict(
      dict(
          algorithm_class=ars_algorithm.AugmentedRandomSearch,
          algorithm_args=dict(
              num_suggestions=8,
              num_evals=10,
              top_percentage=0.5,
              std=0.1,
              step_size=0.1,
              random_seed=7)))


def get_config():
  """Main config."""
  return config_dict.ConfigDict(
      dict(
          coordinator=get_coordinator_config(),
          worker=get_worker_config(),
          algo=get_algo_config()))
