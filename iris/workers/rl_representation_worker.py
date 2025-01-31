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

"""Worker that evaluates a policy with learned representation on a RL env."""

import collections
from typing import Mapping, Optional

from absl import logging
from iris.workers import rl_worker
from iris.workers import worker_util
import numpy as np
import reverb
from tf_agents.environments import gym_wrapper
from tf_agents.google.utils import mp4_video_recorder
from tf_agents.replay_buffers import reverb_utils
from tf_agents.specs import tensor_spec
from tf_agents.trajectories import policy_step
from tf_agents.trajectories import time_step as ts
from tf_agents.trajectories import trajectory


class RLRepresentationWorker(rl_worker.RLWorker):
  """Worker that evaluates a policy with learned representation on a RL env."""

  def __init__(
      self,
      latent_rollout_length: int = 5,
      write_to_replay: bool = False,
      reverb_client: Optional[reverb.client.Client] = None,
      **kwargs
  ) -> None:
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
    collect_data_spec = trajectory.from_transition(
        time_step_spec, policy_step_spec, time_step_spec
    )
    collect_data_spec = tensor_spec.from_spec(collect_data_spec)

    self._write_to_replay = write_to_replay
    if write_to_replay:
      self.reverb_observer = reverb_utils.ReverbAddTrajectoryObserver(
          reverb_client,
          "uniform_table",
          sequence_length=(latent_rollout_length + 1),
          stride_length=1,
      )

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
    rl_worker.workerlock.acquire()

    self._policy.update_weights(params_to_eval)
    if representation_params is not None:
      self._policy.update_representation_weights(representation_params)
    self._policy.reset()
    self._observation_normalizer.buffer.reset()

    if obs_norm_state is not None:
      self._observation_normalizer.state = obs_norm_state

    if env_seed is not None:
      self._env.seed(env_seed)

    video = None
    if record_video:
      video = mp4_video_recorder.Mp4VideoRecorder(video_path, video_framerate)

    reward = 0.0
    metrics = collections.defaultdict(float)
    obs = self._env.reset()
    time_step = ts.TimeStep(
        step_type=ts.StepType.FIRST,
        reward=np.array(0.0, dtype=np.float32),
        discount=np.array(1.0, dtype=np.float32),
        observation=obs,
    )
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
          observation=next_obs,
      )
      if self._write_to_replay:
        self.reverb_observer(
            trajectory.from_transition(time_step, action_step, next_time_step)
        )

      step_output = {
          "steps": st,
          "observation": obs,
          "action": action,
          "next_observation": next_obs,
          "reward": r,
          "done": done,
          rl_worker.INFO: info,
      }
      mdict = self._metrics_fn(self._env, step_output)
      for metric_name, metric_value in mdict.items():
        metrics[metric_name] += metric_value

      obs = next_obs
      time_step = next_time_step

      if record_video and video is not None:
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
            observation=next_obs,
        )
        if self._write_to_replay:
          self.reverb_observer(
              trajectory.from_transition(time_step, action_step, next_time_step)
          )
        break
    if not done and self._write_to_replay:
      self.reverb_observer.reset(write_cached_steps=False)

    if record_video and video is not None:
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
        metrics=dict(metrics),
    )
    rl_worker.workerlock.release()
    return evaluation_result
