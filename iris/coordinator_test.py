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

import glob
import os
import pathlib
import time
from typing import cast

from absl.testing import absltest
import gym
from iris import checkpoint_util
from iris import coordinator
from iris.algorithms import ars_algorithm
from iris.policies import nn_policy
from iris.workers import rl_worker
import launchpad as lp
from ml_collections import config_dict
import numpy as np

from absl.testing import absltest


_TEST_CHECKPOINT = "./testdata/test_checkpoint.pkl"


class TestEnv(gym.Env):

  def __init__(self):
    self._ac_dim = 6
    self._ob_dim = 14
    self.action_space = gym.spaces.Box(
        -1 * np.ones(self._ac_dim), np.ones(self._ac_dim), dtype=np.float32
    )
    self.observation_space = gym.spaces.Box(
        -1 * np.ones(self._ob_dim), np.ones(self._ob_dim), dtype=np.float32
    )

  def step(self, action):
    del action
    return np.zeros(self._ob_dim), 1.0, False, {}

  def reset(self):
    return np.zeros(self._ob_dim)

  def render(self, mode: str = "rgb_array"):
    return np.zeros((16, 16))


def make_bb_program(
    num_workers: int,
    num_eval_workers: int,
    config: config_dict.ConfigDict,
    logdir: absltest._TempDir,
    experiment_name: str,
    warmstartdir: str | None,
    random_seed: int = 1,
) -> lp.Program:
  """Defines the program topology for blackbox training."""

  del random_seed
  program = lp.Program(experiment_name)

  coordinator_config = config.coordinator
  worker_config = config.worker
  algo_config = config.algo

  # Launches worker instances.
  workers = []

  with program.group("worker"):
    for worker_id in range(num_workers):
      worker_handle = program.add_node(
          lp.CourierNode(
              worker_config["worker_class"],
              worker_id=worker_id,
              worker_type="main",
              **worker_config["worker_args"]
          )
      )
      worker_handle.set_client_kwargs()
      workers.append(worker_handle)

  logdir = pathlib.Path(logdir)
  if warmstartdir:
    warmstartdir = pathlib.Path(warmstartdir)
  algo = algo_config["algorithm_class"](**algo_config["algorithm_args"])

  # Launches eval worker instances if there is at least one num_eval_workers.
  if num_eval_workers >= 1:
    eval_workers = []
    with program.group("eval_worker"):
      for worker_id in range(num_eval_workers):
        eva_worker_handle = program.add_node(
            lp.CourierNode(
                worker_config["worker_class"],
                worker_id=worker_id,
                worker_type="eval",
                **worker_config["worker_args"]
            )
        )
        eva_worker_handle.set_client_kwargs()
        eval_workers.append(eva_worker_handle)

    evaluator_node = lp.CourierNode(
        coordinator.Coordinator,
        algo=algo,
        workers=eval_workers,
        logdir=logdir.joinpath(experiment_name),
        **coordinator_config
    )
    evaluator_node.disable_run()
    evaluator = program.add_node(evaluator_node, label="evaluator")
  else:
    evaluator = None

  # Launch the coordinator node that connects to the list of workers.
  program.add_node(
      lp.PyClassNode(
          coordinator.Coordinator,
          algo=algo,
          workers=workers,
          evaluator=evaluator,
          logdir=logdir.joinpath(experiment_name),
          warmstartdir=warmstartdir,
          **coordinator_config
      ),
      label="coordinator",
  )
  return program


class CoordinatorTest(absltest.TestCase):

  def setUp(self):
    super().setUp()

    config = config_dict.ConfigDict(
        dict(
            coordinator=config_dict.ConfigDict(
                dict(
                    save_rate=10,
                    eval_rate=1,
                    num_iterations=400,
                    num_evals_per_suggestion=1,
                    record_video_during_eval=True,
                )
            ),
            worker=config_dict.ConfigDict(
                dict(
                    worker_class=rl_worker.RLWorker,
                    worker_args=dict(
                        env=TestEnv,
                        policy=nn_policy.FullyConnectedNeuralNetworkPolicy,
                        policy_args=dict(hidden_layer_sizes=[64, 64]),
                        rollout_length=20,
                    ),
                )
            ),
            algo=config_dict.ConfigDict(
                dict(
                    algorithm_class=ars_algorithm.AugmentedRandomSearch,
                    algorithm_args=dict(
                        num_suggestions=2,
                        num_evals=4,
                        top_percentage=0.5,
                        std=0.1,
                        step_size=0.1,
                        random_seed=7,
                    ),
                )
            ),
        )
    )
    self.logdir = self.create_tempdir()
    self.program = make_bb_program(
        num_workers=4,
        num_eval_workers=4,
        config=config,
        logdir=self.logdir,
        warmstartdir=_TEST_CHECKPOINT,
        experiment_name="test",
    )
    (coordinator_node,) = self.program.groups["coordinator"]
    coordinator_node = cast(lp.PyClassNode, coordinator_node)
    coordinator_node.disable_run()
    lp.launch(self.program, launch_type="test_mt")
    self.coordinator = coordinator_node._construct_instance()

  def test_coordinator_step(self):
    """Runs one coordinator step."""
    self.coordinator.initialize_algorithm_state()
    self.coordinator.step(0)
    while self.coordinator._evaluator.evaluation_in_progress():
      time.sleep(1)
    self.assertLen(
        self.coordinator._evaluations.keys(),
        len(self.coordinator._aggregate_evaluations),
    )
    self.assertIn(
        os.path.join(self.logdir, "test", "iteration_0000", "video_0.mp4"),
        glob.glob(os.path.join(self.logdir, "**"), recursive=True),
    )

  def test_evaluator(self):
    self.coordinator._algorithm.initialize(
        self.coordinator._workers[0].get_init_state()
    )
    suggestions = self.coordinator._algorithm.get_param_suggestions(
        evaluate=True
    )
    futures = []
    futures.append(
        self.coordinator._evaluator.futures.evaluate(
            iteration=0, suggestions=suggestions
        )
    )
    futures.append(
        self.coordinator._evaluator.futures.save(
            iteration=0, state=self.coordinator._algorithm.state
        )
    )
    futures.append(
        self.coordinator._evaluator.futures.evaluate(
            iteration=10, suggestions=suggestions
        )
    )
    futures.append(
        self.coordinator._evaluator.futures.save(
            iteration=10, state=self.coordinator._algorithm.state
        )
    )
    for future in futures:
      future.result()
    self.assertIn(
        os.path.join(self.logdir, "test", "iteration_0000"),
        glob.glob(os.path.join(self.logdir, "**"), recursive=True),
    )
    self.assertIn(
        os.path.join(self.logdir, "test", "iteration_0010"),
        glob.glob(os.path.join(self.logdir, "**"), recursive=True),
    )

  def test_warmstart(self):
    """Validate warmstart parameters."""
    self.coordinator._algorithm.initialize(
        self.coordinator._workers[0].get_init_state()
    )

    state = checkpoint_util.load_checkpoint_state(_TEST_CHECKPOINT)

    self.coordinator.initialize_algorithm_state()

    np.testing.assert_array_equal(
        self.coordinator._algorithm.state["params_to_eval"],
        state["params_to_eval"],
    )


if __name__ == "__main__":
  absltest.main()
