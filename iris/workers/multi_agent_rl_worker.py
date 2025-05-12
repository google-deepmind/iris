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

"""Worker class that evaluates a multi-agent policy on a multi-agent RL env."""

import collections
from typing import Any, Dict, Mapping, Optional

from absl import logging
from iris import video_recorder
from iris.workers import rl_worker
from iris.workers import worker_util
import numpy as np


class MultiAgentRLWorker(rl_worker.RLWorker):
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
    rl_worker.workerlock.acquire()

    self._policy.update_weights(params_to_eval)
    self._observation_normalizer.buffer.reset()

    if obs_norm_state is not None:
      self._observation_normalizer.state = obs_norm_state

    video = None
    if record_video:
      video = video_recorder.VideoRecorder(video_path, video_framerate)

    reward_dict = collections.defaultdict(float)
    agent_1 = None
    metrics = collections.defaultdict(float)

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
    for _ in range(partial_rollout_length):
      normalized_obs = self._observation_normalizer(
          self._obs, update_obs_norm_buffer
      )
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
          rl_worker.INFO: info,
      }
      mdict = self._metrics_fn(self._env, step_output)
      for metric_name, metric_value in mdict.items():
        metrics[metric_name] += metric_value
      for rkey, rval in reward_dict.items():
        metrics[f"reward_{rkey}"] = rval

      self._obs = next_obs

      if record_video and video is not None:
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
    if record_video and video is not None:
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
        metrics=dict(metrics),
    )
    rl_worker.workerlock.release()
    return evaluation_result
