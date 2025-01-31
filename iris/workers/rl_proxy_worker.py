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

"""RL Proxy Worker."""

from typing import Any, Dict, Mapping, Optional, Sequence

from absl import logging
import courier
from iris.workers import rl_worker
from iris.workers import worker_util
import numpy as np


class RLProxyWorker(rl_worker.RLWorker):
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
    with rl_worker.workerlock:
      # TODO: update to support competitive cases.
      futures = []
      for index, w in enumerate(self._workers):
        logging.info("Worker index: %s", index)
        future = w.futures.work(
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
