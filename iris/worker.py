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

"""Worker class for distributed blackbox optimization library."""

import collections
import enum
import functools
import threading
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Union

from absl import flags
from absl import logging
import courier
import gym
from iris import normalizer
from iris import worker_util
from iris.maml import adaptation_optimizers
from iris.policies import base_policy
from iris.policies import nas_policy
import numpy as np
import pyglove as pg
import qj_global  # pylint: disable=unused-import
import reverb
from tf_agents.environments import gym_wrapper
from tf_agents.google.utils import mp4_video_recorder
from tf_agents.replay_buffers import reverb_utils
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import policy_step
from tf_agents.trajectories import time_step as ts
from tf_agents.trajectories import trajectory

FloatLike = Union[float, np.float32, np.float64]

workerlock = threading.Lock()

INFO = "info"
_VALID_ROLLOUT = "valid"
_MAX_ROLLOUT_RETRIES = 100

# Needed to avoid bullet error (b/196750790)
if "lp_termination_notice_secs" in flags.FLAGS:
  flags.FLAGS.lp_termination_notice_secs = 0


class RolloutRetryError(Exception):
  """If the environment cannot produce a valid rollout."""


class Worker(object):
  # TODO: Rename Worker to FunctionEvaluator
  """Class for evaluating a blackbox function."""

  def __init__(self, worker_id: int, worker_type: str = "main") -> None:
    self._worker_id = worker_id
    self._worker_type = worker_type
    self._init_state = {}

  def work(
      self,
      params_to_eval: np.ndarray,
      enable_logging: bool = False,
  ) -> worker_util.EvaluationResult:
    """Runs the blackbox function on input vars."""
    raise NotImplementedError("Should be implemented in derived classes.")

  def get_init_state(self):
    return self._init_state


class SimpleWorker(Worker):
  """Class for evaluating a given blackbox function."""

  def __init__(self,
               blackbox_function: Callable[[np.ndarray], FloatLike],
               initial_params: np.ndarray | None = None,
               init_function: Callable[..., Any] | None = None,
               **kwargs) -> None:
    super().__init__(**kwargs)
    self._extra_args = {}
    if init_function is not None:
      initial_params, self._extra_args = init_function()
    self._init_state["init_params"] = initial_params
    self._blackbox_function = blackbox_function

  def work(
      self,
      params_to_eval: np.ndarray,
      enable_logging: bool = False,
  ) -> worker_util.EvaluationResult:
    """Runs the blackbox function on input suggestion."""
    value = self._blackbox_function(params_to_eval, **self._extra_args)
    if enable_logging:
      logging.info("Value: %f", value)
    evaluation_result = worker_util.EvaluationResult(params_to_eval, value)
    return evaluation_result


class PyGloveWorker(Worker):
  """Class for evaluating a given blackbox function with an entire PyGlove search space.

  NOTE: This only works with PyGloveAlgorithm.

  Continuous search spaces are offloaded to PyGlove `floatv()` symbolics.
  Useful for evaluating performance of end-to-end evolutionary algorithms like
  NEAT and Reg-Evo.
  """

  def __init__(self, dna_spec: pg.DNASpec,
               blackbox_function: Callable[[pg.DNA],
                                           FloatLike], **kwargs) -> None:
    super().__init__(**kwargs)
    self._init_state["serialized_dna_spec"] = pg.to_json_str(dna_spec)
    self._blackbox_function = blackbox_function

  def work(  # pytype: disable=signature-mismatch  # overriding-parameter-count-checks
      self,
      metadata: str,
      params_to_eval: np.ndarray,  # Ignored.
      enable_logging: bool = False,
  ) -> worker_util.EvaluationResult:
    """Runs the blackbox function on DNA."""
    dna = pg.from_json_str(metadata)
    value = self._blackbox_function(dna)
    if enable_logging:
      logging.info("Value: %f", value)
    evaluation_result = worker_util.EvaluationResult(
        params_to_eval, value, metadata=metadata)
    return evaluation_result


class RLWorker(Worker):
  """Worker class that evaluates a policy on an RL environment."""

  def __init__(
      self,
      env: Union[gym.Env, Callable[[], gym.Env]],
      policy: Union[
          base_policy.BasePolicy, Callable[..., base_policy.BasePolicy]
      ],
      env_args: Optional[Dict[str, Any]] = None,
      policy_args: Optional[Dict[str, Any]] = None,
      observation_normalizer: Union[
          normalizer.Normalizer, Callable[[gym.Space], normalizer.Normalizer]
      ] = normalizer.NoNormalizer,
      action_denormalizer: Union[
          normalizer.Normalizer, Callable[[gym.Space], normalizer.Normalizer]
      ] = normalizer.NoNormalizer,
      rollout_length: int = 500,
      send_training_state_to_env: bool = False,
      metrics_fn: Optional[
          Callable[[gym.Env, Dict[str, Any]], Dict[str, FloatLike]]
      ] = lambda a, b: {},
      stats_fn: Optional[
          Callable[..., tuple[FloatLike, Dict[str, FloatLike]],]
      ] = lambda a, b: (sum(b), {}),
      stats_fn_args: Optional[Dict[str, Any]] = None,
      retry_rollout: bool = False,
      max_rollout_retries: int = _MAX_ROLLOUT_RETRIES,
      **kwargs,
  ) -> None:
    """Initiates RL Worker.

    Args:
      env: Gym RL environment object to run rollout with.
      policy: Policy object to map observations to actions.
      env_args: Arguments for env constructor.
      policy_args: Arguments for policy constructor.
      observation_normalizer: Observation normalizer to map env observation to a
        normalized range for policy input.
      action_denormalizer: Action denormalizer to map policy output to env
        action space.
      rollout_length: Environment rollout length.
      send_training_state_to_env: Whether to send training state to env.
      metrics_fn: Function to add extra metrics to evaluation result.
      stats_fn: Function to compute aggregate value and stats for rollout
        rewards and metrics.
      stats_fn_args: Additional keyword args for stats_fn.
      retry_rollout: Optionally, retry a rollout, if the rollout ends
        with an invalid state. The environment informs rollout state via the
        info object using the key _VALID_ROLLOUT.
      max_rollout_retries: If retry_rollout := True, maximum number of retires.
        Default (_MAX_ROLLOUT_RETRIES).
      **kwargs: Other keyword arguments for base class.
    """
    super().__init__(**kwargs)

    if env_args is None:
      env_args = {}

    if policy_args is None:
      policy_args = {}

    if not isinstance(env, gym.Env):
      self._env = env(**env_args)
    else:
      self._env = env

    logging.info("RLWorker init: env type: %s", type(self._env))
    logging.info("RLWorker init: obs space: %s", self._env.observation_space)
    logging.info("RLWorker init: action space: %s", self._env.action_space)

    if not isinstance(policy, base_policy.BasePolicy):
      self._policy = policy(
          ob_space=self._env.observation_space,
          ac_space=self._env.action_space,
          **policy_args)
    else:
      self._policy = policy

    if not isinstance(observation_normalizer, normalizer.Normalizer):
      self._observation_normalizer = observation_normalizer(
          self._env.observation_space)
    else:
      self._observation_normalizer = observation_normalizer

    if not isinstance(action_denormalizer, normalizer.Normalizer):
      self._action_denormalizer = action_denormalizer(self._env.action_space)
    else:
      self._action_denormalizer = action_denormalizer

    self._rollout_length = rollout_length
    self._max_rollout_retries = max_rollout_retries
    self._retry_rollout = retry_rollout
    self._send_training_state_to_env = send_training_state_to_env
    self._metrics_fn = metrics_fn
    self._stats_fn = stats_fn
    self._stats_fn_args = {} if stats_fn_args is None else stats_fn_args
    self._step = 0
    self._obs = None

    self._init_state["init_params"] = self._policy.get_weights()
    self._init_state["obs_norm_buffer_data"] = (
        self._observation_normalizer.buffer.data
    )

  def work(  # pytype: disable=signature-mismatch  # overriding-default-value-checks
      self,
      params_to_eval: np.ndarray,
      obs_norm_state: Optional[Mapping[str, np.ndarray]] = None,
      update_obs_norm_buffer: bool = False,
      partial_rollout_length: Optional[int] = None,
      env_seed: Optional[int] = None,
      enable_logging: bool = False,
      record_video: bool = False,
      video_framerate: Optional[int] = None,
      video_path: Optional[str] = None,
      training_state: Optional[Dict[str, Any]] = None,
      iteration: Optional[int] = None,
  ) -> worker_util.EvaluationResult:
    """Runs one episode of the env with given policy parameters.

    Args:
      params_to_eval: Weight vector for policy to evaluate.
      obs_norm_state: Observation normalizer state (mean and std).
      update_obs_norm_buffer: Whether to update observation normalizer buffer.
      partial_rollout_length: Partial environment rollout length.
      env_seed: Environment random seed.
      enable_logging: Whether to log intermediate data.
      record_video: Whether to record video.
      video_framerate: Framerate of video recording.
      video_path: Path for saving video.
      training_state: Metrics related to training.
      iteration: Coordinator iteration number.

    Returns:
      A tuple with modified suggestion and total episode reward.

    Raises:
      RolloutRetryError: if retry_rollout := True and environment info object
      contains _VALID_ROLLOUT key, and a valid rollout cannot be generated
      within `max_rollout_retries` retries.
    """
    workerlock.acquire()

    self._policy.update_weights(params_to_eval)
    self._policy.set_iteration(iteration)

    valid_rollout, reward, metrics, video = False, 0.0, {}, None
    num_rollouts = 0
    while not valid_rollout and num_rollouts < self._max_rollout_retries:
      valid_rollout, reward, metrics, video = self._run_rollout(
          obs_norm_state,
          update_obs_norm_buffer,
          partial_rollout_length,
          env_seed,
          enable_logging,
          record_video,
          video_framerate,
          video_path,
          training_state,
      )
      num_rollouts += 1

    if self._retry_rollout and not valid_rollout:
      raise RolloutRetryError(
          f"Rollout error: max_rollout_retries: {self._max_rollout_retries}")

    metrics["current_step"] = self._step
    if record_video and video is not None:
      video.end_video()
    if enable_logging:
      logging.info("Total Reward: %f, Num invalid rollouts: %i", reward,
                   num_rollouts - 1)

    buffer_data = self._observation_normalizer.buffer.data
    evaluation_result = worker_util.EvaluationResult(
        params_evaluated=params_to_eval,
        value=reward,
        obs_norm_buffer_data=buffer_data,
        metrics=dict(metrics))
    workerlock.release()
    return evaluation_result

  def _run_rollout(self,
                   obs_norm_state: Optional[Mapping[str, np.ndarray]] = None,
                   update_obs_norm_buffer: bool = False,
                   partial_rollout_length: Optional[int] = None,
                   env_seed: Optional[int] = None,
                   enable_logging: bool = False,
                   record_video: bool = False,
                   video_framerate: Optional[int] = None,
                   video_path: Optional[str] = None,
                   training_state: Optional[Dict[str, Any]] = None):
    """Runs a rollout."""
    self._observation_normalizer.buffer.reset()

    if obs_norm_state is not None:
      self._observation_normalizer.state = obs_norm_state

    video = None
    if record_video:
      video = mp4_video_recorder.Mp4VideoRecorder(video_path, video_framerate)

    rewards = []
    metrics = collections.defaultdict(list)

    if self._send_training_state_to_env and hasattr(self._env,
                                                    "set_training_state"):
      self._env.set_training_state(training_state)

    if self._step == 0:
      self._policy.reset()
      if env_seed is not None:
        self._env.seed(env_seed)
      self._obs = self._env.reset()

    if partial_rollout_length is None:
      partial_rollout_length = self._rollout_length
    valid_rollout = True
    for _ in range(partial_rollout_length):
      normalized_obs = self._observation_normalizer(self._obs,
                                                    update_obs_norm_buffer)
      action = self._action_denormalizer(self._policy.act(normalized_obs))
      next_obs, r, done, info = self._env.step(action)
      rewards.append(r)

      step_output = {
          "steps": self._step,
          "observation": self._obs,
          "action": action,
          "next_observation": next_obs,
          "reward": r,
          "done": done,
          INFO: info
      }
      mdict = self._metrics_fn(self._env, step_output)
      for metric_name, metric_value in mdict.items():
        metrics[metric_name].append(metric_value)

      self._obs = next_obs

      if record_video:
        video.add_frame(self._env.render(mode="rgb_array"))
      if enable_logging:
        logging.info("Step: %d, Reward: %f, Done: %d", self._step, r, done)
        if mdict:
          logging.info("Metrics: %s", mdict)

      self._step += 1
      if done or (self._step >= self._rollout_length):
        if self._retry_rollout and _VALID_ROLLOUT in info:
          valid_rollout = info[_VALID_ROLLOUT]
        self._step = 0
        break

    aggregate_reward, reward_stats = self._stats_fn(
        "reward", rewards, **self._stats_fn_args
    )
    aggregate_metrics_with_stats = {}
    for metric_name, metric_values in metrics.items():
      metric, metric_stats = self._stats_fn(
          metric_name, metric_values, **self._stats_fn_args
      )
      aggregate_metrics_with_stats[metric_name] = metric
      aggregate_metrics_with_stats.update(metric_stats)
    aggregate_metrics_with_stats.update(reward_stats)

    return valid_rollout, aggregate_reward, aggregate_metrics_with_stats, video


class MultiAgentRLWorker(RLWorker):
  """Worker class that evaluates a multi-agent policy on a multi-agent RL env.

  The only difference compared with RLWorker is MultiAgentRLWorker expects the
  environment to return a dictionary of rewards, one per agent, instead of a
  float.
  """

  def work(  # pytype: disable=signature-mismatch  # overriding-default-value-checks
      self,
      params_to_eval: np.ndarray,
      obs_norm_state: Optional[Mapping[str, np.ndarray]] = None,
      update_obs_norm_buffer: bool = False,
      partial_rollout_length: Optional[int] = None,
      env_seed: Optional[int] = None,
      enable_logging: bool = False,
      record_video: bool = False,
      video_framerate: Optional[int] = None,
      video_path: Optional[str] = None,
      training_state: Optional[Dict[str, Any]] = None,
  ) -> worker_util.EvaluationResult:
    """Runs one episode of the env with given policy parameters.

    Args:
      params_to_eval: Weight vector for policy to evaluate.
      obs_norm_state: Observation normalizer state (mean and std).
      update_obs_norm_buffer: Whether to update observation normalizer buffer.
      partial_rollout_length: Partial environment rollout length.
      env_seed: Environment random seed.
      enable_logging: Whether to log intermediate data.
      record_video: Whether to record video.
      video_framerate: Framerate of video recording.
      video_path: Path for saving video.
      training_state: Metrics related to training.

    Returns:
      A tuple with modified suggestion and total episode reward.
    """
    workerlock.acquire()

    self._policy.update_weights(params_to_eval)
    self._observation_normalizer.buffer.reset()

    if obs_norm_state is not None:
      self._observation_normalizer.state = obs_norm_state

    if record_video:
      video = mp4_video_recorder.Mp4VideoRecorder(video_path, video_framerate)

    reward_dict = collections.defaultdict(float)
    agent_1 = None
    metrics = collections.defaultdict(float)

    if self._send_training_state_to_env and hasattr(self._env,
                                                    "set_training_state"):
      self._env.set_training_state(training_state)

    if self._step == 0:
      self._policy.reset()
      if env_seed is not None:
        self._env.seed(env_seed)
      self._obs = self._env.reset()

    if partial_rollout_length is None:
      partial_rollout_length = self._rollout_length
    for _ in range(partial_rollout_length):
      normalized_obs = self._observation_normalizer(self._obs,
                                                    update_obs_norm_buffer)
      action = self._action_denormalizer(self._policy.act(normalized_obs))
      next_obs, r, done, info = self._env.step(action)

      for rkey, rval in r.items():
        if rkey not in reward_dict:
          reward_dict[rkey] = rval
        else:
          reward_dict[rkey] += rval
      if agent_1 is None:
        agent_1 = list(r.keys())[0]

      step_output = {
          "steps": self._step,
          "observation": self._obs,
          "action": action,
          "next_observation": next_obs,
          "reward": r[agent_1],
          "done": done,
          INFO: info
      }
      mdict = self._metrics_fn(self._env, step_output)
      for metric_name, metric_value in mdict.items():
        metrics[metric_name] += metric_value
      for rkey, rval in reward_dict.items():
        metrics[f"reward_{rkey}"] = rval

      self._obs = next_obs

      if record_video:
        video.add_frame(self._env.render(mode="rgb_array"))
      if enable_logging:
        logging.info("Step: %d, Reward: %s, Done: %d", self._step, r, done)
        if mdict:
          logging.info("Metrics: %s", mdict)

      self._step += 1
      if done or (self._step >= self._rollout_length):
        self._step = 0
        break

    metrics["current_step"] = self._step
    if record_video:
      video.end_video()
    if enable_logging:
      logging.info("Total Reward: %s", reward_dict)

    buffer_data = None
    if self._observation_normalizer.buffer is not None:
      buffer_data = self._observation_normalizer.buffer.data
    evaluation_result = worker_util.EvaluationResult(
        params_evaluated=params_to_eval,
        value=reward_dict[agent_1],
        obs_norm_buffer_data=buffer_data,
        metrics=dict(metrics))
    workerlock.release()
    return evaluation_result


class RLProxyWorker(RLWorker):
  """RL Proxy Worker.

  RLProxyWorker bridges the gap between training in simulation and real when
  the reinforcement learning environment consists of multiple agents, and if
  the real agents are exposed via independent Gym APIs (e.g., Toaster table
  tennis real agents).

  RLProxyWorker registers with the coordinator and functions similarly to
  RLWorker. Therefore, it loads a simulated environment and policy, which is
  required by the coordinator at the initialization.

  RLProxyWorker receives rollout weights which demultiplexes to the helper
  workers. The rollout results are multiplexed and communicated back to the
  coordinator.

  NOTE: the current implementation returns the rollout results of the first
  helper worker assuming homogeneous results.

  NOTE: to extend to a multi-agent environment with other specifications
  (b/271358434).

  Example extension for table tennis agent-vs-agent with
  cooperative/competitive/mixed agents trained with MultiAgentRLWorker.

  RLProxyWorker initializes with MultiAgentRLWorker configuration. The helper
  workers initialize with agents with a specific configuration: PLAYER is
  cooperative and PLAYER2 is adversarial. This is configured from
  the config flag to the launch script.

  RLProxyWorker returns the results of the agents (similar to MultiAgentRLWorker
  returns).

  Design doc: http://shortn/_46gpheZmP9
  """

  def __init__(
      self,
      workers: Sequence[courier.Client],
      **kwargs,
  ):
    """Initiates RL Proxy Worker.

    Args:
      workers: Workers for evaluating RL function.
      **kwargs: Other keyword arguments for base class.
    """
    super().__init__(**kwargs)
    self._workers = workers
    if self._workers is None:
      raise ValueError("A set of workers must be available.")

    logging.info("RLProxyWorker: env type: %s", type(self._env))
    logging.info("RLProxyWorker: obs space: %s", self._env.observation_space)
    logging.info("RLProxyWorker: action space: %s", self._env.action_space)

  # pylint: disable=arguments-renamed
  def work(  # pytype: disable=signature-mismatch  # overriding-default-value-checks
      self,
      params_to_eval: np.ndarray,
      obs_norm_state: Optional[Mapping[str, np.ndarray]] = None,
      update_obs_norm_buffer: bool = False,
      partial_rollout_length: Optional[int] = None,
      env_seed: Optional[int] = None,
      enable_logging: bool = False,
      record_video: bool = False,
      video_framerate: Optional[int] = None,
      video_path: Optional[str] = None,
      training_state: Optional[Dict[str, Any]] = None,
  ) -> worker_util.EvaluationResult:
    """Multiplexer and a demultiplexer for the policy parameters.

    Args:
      params_to_eval: Weight vector for policy to evaluate.
      obs_norm_state: Observation normalizer state (mean and std).
      update_obs_norm_buffer: Whether to update observation normalizer buffer.
      partial_rollout_length: Partial environment rollout length.
      env_seed: Environment random seed.
      enable_logging: Whether to log intermediate data.
      record_video: Whether to record video.
      video_framerate: Framerate of video recording.
      video_path: Path for saving video.
      training_state: Metrics related to training.

    Returns:
      A tuple with modified suggestion and total episode reward.

    Raises:
      ValueError: if a worker returns an error.
    """
    results = dict()
    with workerlock:
      # TODO: update to support competitive cases.
      futures = []
      for index, worker in enumerate(self._workers):
        logging.info("Worker index: %s", index)
        future = worker.futures.work(
            params_to_eval,
            obs_norm_state,
            update_obs_norm_buffer,
            partial_rollout_length,
            env_seed,
            enable_logging,
            record_video,
            video_framerate,
            video_path,
            training_state,
        )
        futures.append((index, future))

      for index, future in futures:
        if future.cancelled():
          raise ValueError("Helper worker cancelled.")
        if future.exception():
          raise ValueError(future.exception())
        result = future.result()
        logging.info("Result: %s", result.value)
        results[index] = result

    logging.info("Multiplexed result: %s", results[0].value)
    return results[0]

  # pylint: enable=arguments-renamed


class RLRepresentationWorker(RLWorker):
  """Worker that evaluates a policy with learned representation on a RL env."""

  def __init__(self,
               latent_rollout_length: int = 5,
               write_to_replay: bool = False,
               reverb_client: Optional[reverb.client.Client] = None,
               **kwargs) -> None:
    """Initiates RL with Representation Learning Worker.

    Args:
      latent_rollout_length: Latent environment rollout length.
      write_to_replay: Whether to write to replay buffer.
      reverb_client: A `reverb.client.Client`.
      **kwargs: Other keyword arguments for base class.
    """
    super().__init__(**kwargs)

    representation_params = self._policy.get_representation_weights()
    self._init_state["init_representation_params"] = representation_params
    if reverb_client is not None:
      self._init_state["reverb_server_addr"] = reverb_client.server_address

    obs_spec = gym_wrapper.spec_from_gym_space(self._env.observation_space)
    action_spec = gym_wrapper.spec_from_gym_space(self._env.action_space)
    time_step_spec = ts.time_step_spec(observation_spec=obs_spec)
    policy_step_spec = policy_step.PolicyStep(action=action_spec)
    collect_data_spec = trajectory.from_transition(time_step_spec,
                                                   policy_step_spec,
                                                   time_step_spec)
    collect_data_spec = tensor_spec.from_spec(collect_data_spec)

    self._write_to_replay = write_to_replay
    if write_to_replay:
      self.reverb_observer = reverb_utils.ReverbAddTrajectoryObserver(
          reverb_client,
          "uniform_table",
          sequence_length=(latent_rollout_length+1),
          stride_length=1)

  def work(  # pytype: disable=signature-mismatch  # overriding-default-value-checks
      self,
      params_to_eval: np.ndarray,
      representation_params: Optional[np.ndarray] = None,
      obs_norm_state: Optional[Mapping[str, np.ndarray]] = None,
      update_obs_norm_buffer: bool = False,
      env_seed: Optional[int] = None,
      enable_logging: bool = True,
      record_video: bool = False,
      video_framerate: Optional[int] = None,
      video_path: Optional[str] = None,
  ) -> worker_util.EvaluationResult:
    """Runs one episode of the env with given policy parameters.

    Args:
      params_to_eval: Weight vector for policy to evaluate.
      representation_params: Representation weight vector.
      obs_norm_state: Observation normalizer state (mean and std).
      update_obs_norm_buffer: Whether to update observation normalizer buffer.
      env_seed: Environment random seed.
      enable_logging: Whether to log intermediate data.
      record_video: Whether to record video.
      video_framerate: Framerate of video recording.
      video_path: Path for saving video.

    Returns:
      A tuple with modified suggestion and total episode reward.
    """
    workerlock.acquire()

    self._policy.update_weights(params_to_eval)
    if representation_params is not None:
      self._policy.update_representation_weights(representation_params)
    self._policy.reset()
    self._observation_normalizer.buffer.reset()

    if obs_norm_state is not None:
      self._observation_normalizer.state = obs_norm_state

    if env_seed is not None:
      self._env.seed(env_seed)

    if record_video:
      video = mp4_video_recorder.Mp4VideoRecorder(video_path, video_framerate)

    reward = 0.
    metrics = collections.defaultdict(float)
    obs = self._env.reset()
    time_step = ts.TimeStep(step_type=ts.StepType.FIRST,
                            reward=np.array(0.0, dtype=np.float32),
                            discount=np.array(1.0, dtype=np.float32),
                            observation=obs)
    obs = self._observation_normalizer(obs, update_obs_norm_buffer)
    done = False
    for st in range(self._rollout_length):
      normalized_obs = self._observation_normalizer(obs, update_obs_norm_buffer)
      action = self._policy.act(normalized_obs)
      action_step = policy_step.PolicyStep(action)
      action = self._action_denormalizer(action)
      next_obs, r, done, info = self._env.step(action)
      reward += r
      next_time_step = ts.TimeStep(
          step_type=ts.StepType.LAST if done else ts.StepType.MID,
          reward=np.array(r, dtype=np.float32),
          discount=np.array(0.0 if done else 1.0, dtype=np.float32),
          observation=next_obs)
      if self._write_to_replay:
        self.reverb_observer(
            trajectory.from_transition(time_step, action_step, next_time_step))

      step_output = {
          "steps": st,
          "observation": obs,
          "action": action,
          "next_observation": next_obs,
          "reward": r,
          "done": done,
          INFO: info
      }
      mdict = self._metrics_fn(self._env, step_output)
      for metric_name, metric_value in mdict.items():
        metrics[metric_name] += metric_value

      obs = next_obs
      time_step = next_time_step

      if record_video:
        video.add_frame(self._env.render(mode="rgb_array"))
      if enable_logging:
        logging.info("Step: %d, Reward: %f, Done: %d", st, r, done)
        if mdict:
          logging.info("Metrics: %s", mdict)
      if done:
        next_obs, r, _, _ = self._env.step(action)
        next_time_step = ts.TimeStep(
            step_type=ts.StepType.FIRST,
            reward=np.array(0.0, dtype=np.float32),
            discount=np.array(1.0, dtype=np.float32),
            observation=next_obs)
        if self._write_to_replay:
          self.reverb_observer(trajectory.from_transition(
              time_step, action_step, next_time_step))
        break
    if not done and self._write_to_replay:
      self.reverb_observer.reset(write_cached_steps=False)

    if record_video:
      video.end_video()
    if enable_logging:
      logging.info("Total Reward: %f", reward)

    buffer_data = None
    if self._observation_normalizer.buffer is not None:
      buffer_data = self._observation_normalizer.buffer.data
    evaluation_result = worker_util.EvaluationResult(
        params_evaluated=params_to_eval,
        value=reward,
        obs_norm_buffer_data=buffer_data,
        metrics=dict(metrics))
    workerlock.release()
    return evaluation_result


# TODO: Consider combining w/ PyGloveWorker.
class PyGloveRLWorker(RLWorker):
  """Subclass of RLWorker that allows PyGlove DNAs.

  NOTE: This only works with ES-ENAS.
  """

  def __init__(self, policy: Union[nas_policy.PyGlovePolicy,
                                   Callable[..., nas_policy.PyGlovePolicy]],
               **kwargs) -> None:
    super().__init__(policy=policy, **kwargs)
    self._init_state["serialized_dna_spec"] = pg.to_json_str(
        self._policy.dna_spec)  # pytype: disable=attribute-error

  def work(self,
           metadata: Optional[str] = None,
           **kwargs) -> worker_util.EvaluationResult:
    if metadata:
      dna = pg.from_json_str(metadata)
      self._policy.update_dna(dna)  # pytype: disable=attribute-error
    vanilla_evaluation_result = super().work(**kwargs)
    evaluation_result = worker_util.EvaluationResult(
        params_evaluated=vanilla_evaluation_result.params_evaluated,
        value=vanilla_evaluation_result.value,
        obs_norm_buffer_data=vanilla_evaluation_result.obs_norm_buffer_data,
        metadata=metadata)
    return evaluation_result


@enum.unique
class AdaptationType(enum.Enum):
  HILLCLIMB = 1
  GRADIENT = 2


class MAMLWorker(Worker):
  """Makes multiple calls of another worker's work() function for adaptation.

  `worker_kwargs` will be passed into the worker's constructor, but it's best
  practice to externally wrap the constructor with functools.partial.
  """

  def __init__(self, worker_constructor: Callable[[], Worker],
               adaptation_type: AdaptationType,
               adaptation_kwargs: Mapping[str, Any], **worker_kwargs) -> None:
    super().__init__(**worker_kwargs)
    self._worker = worker_constructor(**worker_kwargs)
    self._adaptation_type = adaptation_type
    self._init_state = self._worker._init_state

    if self._adaptation_type is AdaptationType.HILLCLIMB:
      self._adaptation_optimizer = adaptation_optimizers.HillClimbAdaptation(
          **adaptation_kwargs)
    elif self._adaptation_type is AdaptationType.GRADIENT:
      self._adaptation_optimizer = adaptation_optimizers.GradientAdaptation(
          **adaptation_kwargs)

  def work(self, params_to_eval, **work_kwargs) -> worker_util.EvaluationResult:  # pytype: disable=signature-mismatch  # overriding-parameter-count-checks
    """Uses another Worker's work() function for adaptation.

    Please note to make sure that `work_fn` represents the same objective
    function throughout this entire call.

    Args:
      params_to_eval: Starting parameter of the inner loop, or AKA "meta-point".
      **work_kwargs: Extra keyword arguments for freezing the worker's work()
        function.

    Returns:
      Evaluation of the adapted parameter.
    """

    work_fn = functools.partial(self._worker.work, **work_kwargs)

    adapted_value, total_results = self._adaptation_optimizer.run_adaptation(
        params_to_eval=params_to_eval, work_fn=work_fn)
    merged_result = worker_util.merge_eval_results(
        total_results)  # collecting obs buffer

    return worker_util.EvaluationResult(  # pytype: disable=wrong-arg-types  # numpy-scalars
        params_evaluated=params_to_eval,
        value=adapted_value,
        obs_norm_buffer_data=merged_result.obs_norm_buffer_data)
