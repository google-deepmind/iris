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

r"""Launches distributed blackbox optimization using Launchpad."""

import pathlib

from absl import app  # pylint: disable=unused-import
from absl import flags
from flax import traverse_util
from iris import coordinator
import launchpad as lp
from ml_collections import config_dict as configdict
from ml_collections import config_flags


_gib = 2**30

_EXPERIMENT_NAME = flags.DEFINE_string('experiment_name', 'BBv2',
                                       'Name of Experiment.')
_LOGDIR = flags.DEFINE_string('logdir', '/tmp/bbv2_logs/', 'Logging directory.')
_WARMSTARTDIR = flags.DEFINE_string('warmstartdir', None,
                                    'Directory to load warmstarting policy.')
_NUM_WORKERS = flags.DEFINE_integer(
    'num_workers', 64, 'Number of worker instances to launch per trial.')
_NUM_EVAL_WORKERS = flags.DEFINE_integer(
    'num_eval_workers', 16,
    'Number of eval worker instances to launch per trial.')
_NUM_RANDOM_SEEDS = flags.DEFINE_integer(
    'num_random_seeds', 1,
    'Number of program replicas to launch in vanilla case.')
_CONFIG = config_flags.DEFINE_config_file('config', 'path/to/config',
                                          'Configuration file.')


def make_bb_program(
    num_workers: int,
    num_eval_workers: int,
    config: configdict.ConfigDict,
    logdir: str,
    experiment_name: str,
    warmstartdir: str,
    random_seed: int = 1,
) -> lp.Program:
  """Defines the program topology for blackbox training."""

  del random_seed
  program = lp.Program(experiment_name)

  coordinator_config = config.coordinator
  worker_config = config.worker
  eval_worker_config = (
      worker_config if 'eval_worker' not in config else config.eval_worker
  )
  algo_config = config.algo
  replay_config = None if 'replay' not in config else config.replay

  if replay_config is not None:
    replay = program.add_node(
        lp.ReverbNode(priority_tables_fn=replay_config.replay_fn),
        label='reverb')

    with worker_config.unlocked():
      worker_config['worker_args']['reverb_client'] = replay

    with eval_worker_config.unlocked():
      eval_worker_config['worker_args']['reverb_client'] = replay

  # Launches worker instances.
  workers = []
  call_timeout = (
      None
      if 'call_timeout' not in worker_config
      else worker_config.call_timeout
  )
  helper_worker_dict = dict()
  if (
      num_eval_workers == 0
      and num_workers == 1
      and 'helper_workers' in config.worker
  ):
    helper_workers = []
    with program.group('helper_worker'):
      for worker_id, helper_worker_config in enumerate(
          config.worker.helper_workers
      ):
        worker_handle = program.add_node(
            lp.CourierNode(
                helper_worker_config['worker_class'],
                worker_id=worker_id,
                worker_type='main',
                **helper_worker_config['worker_args'],
            )
        )
        worker_handle.set_client_kwargs(call_timeout=call_timeout)
        helper_workers.append(worker_handle)
    helper_worker_dict['workers'] = helper_workers

  with program.group('worker'):
    for worker_id in range(num_workers):
      worker_handle = program.add_node(
          lp.CourierNode(
              worker_config['worker_class'],
              worker_id=worker_id,
              worker_type='main',
              **helper_worker_dict,
              **worker_config['worker_args']))
      worker_handle.set_client_kwargs(call_timeout=call_timeout)
      workers.append(worker_handle)

  logdir = pathlib.Path(logdir)
  if warmstartdir:
    warmstartdir = pathlib.Path(warmstartdir)
  algo = algo_config['algorithm_class'](**algo_config['algorithm_args'])

  # Launches eval worker instances if there is at least one num_eval_workers.
  if num_eval_workers >= 1:
    eval_workers = []
    call_timeout = (
        None
        if 'call_timeout' not in eval_worker_config
        else eval_worker_config.call_timeout
    )
    with program.group('eval_worker'):
      for worker_id in range(num_eval_workers):
        eva_worker_handle = program.add_node(
            lp.CourierNode(
                eval_worker_config['worker_class'],
                worker_id=worker_id,
                worker_type='eval',
                **eval_worker_config['worker_args']))
        eva_worker_handle.set_client_kwargs(call_timeout=call_timeout)
        eval_workers.append(eva_worker_handle)

    evaluator_node = lp.CourierNode(
        coordinator.Coordinator,
        algo=algo,
        workers=eval_workers,
        logdir=logdir.joinpath(experiment_name),
        **coordinator_config)
    evaluator_node.disable_run()
    evaluator = program.add_node(evaluator_node, label='evaluator')
  else:
    # The "coordinator" uses an "evaluator" CourierNode to offload the eval
    # computation. The "evaluator" has its own set of "eval_workers". When the
    # evaluator is None, the "coordinator" uses its own workers to evaluate
    # as well. Thus the coordinator uses its workers to get the returns for the
    # suggestions as well as to evaluate the optimum suggestion. This is a
    # requirement for on robot training/evaluation, as both the training and
    # evaluation phases need to share the same environment that binds with the
    # hardware.
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
          **coordinator_config),
      label='coordinator')
  return program


def launch(make_program):
  """Launches Launchpad program."""
  config = _CONFIG.value
  programs = []
  for i in range(_NUM_RANDOM_SEEDS.value):
    program = make_program(
        num_workers=_NUM_WORKERS.value,
        num_eval_workers=_NUM_EVAL_WORKERS.value,
        config=config,
        logdir=_LOGDIR.value,
        warmstartdir=_WARMSTARTDIR.value,
        random_seed=i,
    )
    programs.append(program)
  lp.launch(programs)


def main(_):
  launch(make_bb_program)


if __name__ == '__main__':
  app.run(main)
