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

"""Example RL configuration for Iris experiments."""

import gym
from iris.algorithms import ars_algorithm
from iris.policies import keras_nn_policy
from iris.workers import rl_worker
from ml_collections import config_dict


def get_coordinator_config() -> config_dict.ConfigDict:
  """Coordinator config."""

  return config_dict.ConfigDict(
      dict(
          save_rate=1,
          eval_rate=1,
          num_iterations=2,
          num_evals_per_suggestion=1,
      )
  )


def get_worker_config() -> config_dict.ConfigDict:
  """Worker config."""

  config = config_dict.ConfigDict()
  config.worker_class = rl_worker.RLWorker
  config.worker_args = config_dict.ConfigDict(
      dict(
          env=gym.make,
          env_args=dict(id="Pendulum-v1"),
          policy=keras_nn_policy.KerasNNPolicy,
          policy_args=dict(
              hidden_layer_sizes=[10, 10],
              activation="tanh",
              kernel_initializer="glorot_uniform",
          ),
          rollout_length=1000,
      )
  )
  return config


def get_algo_config() -> config_dict.ConfigDict:
  """Algorithm config."""

  return config_dict.ConfigDict(
      dict(
          algorithm_class=ars_algorithm.AugmentedRandomSearch,
          algorithm_args=dict(
              num_suggestions=2,
              num_evals=2,
              top_percentage=0.5,
              std=0.1,
              step_size=0.1,
              random_seed=7,
          ),
      )
  )


def get_config() -> config_dict.ConfigDict:
  """Main config."""
  return config_dict.ConfigDict(
      dict(
          coordinator=get_coordinator_config(),
          worker=get_worker_config(),
          algo=get_algo_config(),
      )
  )
