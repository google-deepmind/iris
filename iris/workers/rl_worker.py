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

"""Worker class that evaluates a policy on an RL environment."""

import collections
import threading
from typing import Any, Callable, Dict, Mapping, Optional, Union

from absl import logging
import gym
from iris import normalizer
from iris.policies import base_policy
from iris.workers import worker
from iris.workers import worker_util
import numpy as np
from tf_agents.google.utils import mp4_video_recorder


workerlock = threading.Lock()

INFO = "info"
_VALID_ROLLOUT = "valid"
_MAX_ROLLOUT_RETRIES = 100


class RolloutRetryError(Exception):
  """If the environment cannot produce a valid rollout."""


class RLWorker(worker.Worker):
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
          Callable[[gym.Env, Dict[str, Any]], Dict[str, worker.FloatLike]]
      ] = lambda a, b: {},
      stats_fn: Optional[
          Callable[
              ...,
              tuple[worker.FloatLike, Dict[str, worker.FloatLike]],
          ]
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
      retry_rollout: Optionally, retry a rollout, if the rollout ends with an
        invalid state. The environment informs rollout state via the info object
        using the key _VALID_ROLLOUT.
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
          **policy_args,
      )
    else:
      self._policy = policy

    if not isinstance(observation_normalizer, normalizer.Normalizer):
      self._observation_normalizer = observation_normalizer(
          self._env.observation_space
      )
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
    self._obs = {}

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
          f"Rollout error: max_rollout_retries: {self._max_rollout_retries}"
      )

    metrics["current_step"] = self._step
    if record_video and video is not None:
      video.end_video()
    if enable_logging:
      logging.info(
          "Total Reward: %f, Num invalid rollouts: %i", reward, num_rollouts - 1
      )

    buffer_data = self._observation_normalizer.buffer.data
    evaluation_result = worker_util.EvaluationResult(
        params_evaluated=params_to_eval,
        value=reward,
        obs_norm_buffer_data=buffer_data,
        metrics=dict(metrics),
    )
    workerlock.release()
    return evaluation_result

  def _run_rollout(
      self,
      obs_norm_state: Optional[Mapping[str, np.ndarray]] = None,
      update_obs_norm_buffer: bool = False,
      partial_rollout_length: Optional[int] = None,
      env_seed: Optional[int] = None,
      enable_logging: bool = False,
      record_video: bool = False,
      video_framerate: Optional[int] = None,
      video_path: Optional[str] = None,
      training_state: Optional[Dict[str, Any]] = None,
  ):
    """Runs a rollout."""
    self._observation_normalizer.buffer.reset()

    if obs_norm_state is not None:
      self._observation_normalizer.state = obs_norm_state

    video = None
    if record_video:
      video = mp4_video_recorder.Mp4VideoRecorder(video_path, video_framerate)

    rewards = []
    metrics = collections.defaultdict(list)

    if self._send_training_state_to_env and hasattr(
        self._env, "set_training_state"
    ):
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
      normalized_obs = self._observation_normalizer(
          self._obs, update_obs_norm_buffer
      )
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
          INFO: info,
      }
      mdict = self._metrics_fn(self._env, step_output)
      for metric_name, metric_value in mdict.items():
        metrics[metric_name].append(metric_value)

      self._obs = next_obs

      if record_video and video is not None:
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
