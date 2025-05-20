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

"""Coordinator class for distributed blackbox optimization library."""

from collections.abc import Mapping, Sequence
from concurrent import futures
import dataclasses
import os
import pathlib
import pickle as pkl
import threading
import time
from typing import Any, Callable

from absl import flags
from absl import logging
import courier
from iris import checkpoint_util
from iris import logger
from iris.algorithms import algorithm
from iris.workers import worker_util
import launchpad as lp
import numpy as np
from tensorflow.io import gfile


# Needed to avoid bullet error (b/196750790)
if "lp_termination_notice_secs" in flags.FLAGS:
  flags.FLAGS.lp_termination_notice_secs = 0

DEFAULT_SAVE_RATE = 20
DEFAULT_EVAL_RATE = 20


@dataclasses.dataclass(frozen=True, kw_only=True)
class CallbackInfo:
  """A class for holding information to be used while collecting evals.

  Attributes:
    callback: Callback function that should be called after collecting the eval.
    future: The future that was used to send the eval request.
    worker_id: ID of the worker the request was sent to.
    suggestion_id: ID of the suggestion sent.
    iteration: Training iteration when the request was sent.
  """
  callback: Callable[[futures.Future[Any]], None]
  future: futures.Future[Any]
  worker_id: int
  suggestion_id: int
  iteration: int


class Coordinator:
  """Runs distributed blackbox optimization."""

  def __init__(
      self,
      *,
      num_iterations: int,
      num_evals_per_suggestion: int,
      workers: Sequence[courier.Client],
      algo: algorithm.BlackboxAlgorithm,
      logdir: pathlib.Path,
      save_rate: int = DEFAULT_SAVE_RATE,
      eval_rate: int = DEFAULT_EVAL_RATE,
      evaluator: courier.Client | None = None,
      warmstartdir: pathlib.Path | None = None,
      record_video_during_eval: bool = False,
      num_videos: int = 1,
      video_framerate: int = 10,
      video_extension: str = "mp4",
      metrics: Sequence[str] = (),
      send_training_state_to_workers: bool = False,
      send_iteration_to_workers: bool = False,
      reschedule_failed_request: bool = True,
      merge_fn: Callable[
          [Sequence[worker_util.EvaluationResult]], worker_util.EvaluationResult
      ] = worker_util.merge_eval_results,
  ) -> None:
    """Initiates Coordinator.

    Args:
      num_iterations: Number of iterations for Blackbox optimization.
      num_evals_per_suggestion: Number of times each suggestion should be
        evaluated.
      workers: Workers for evaluating blackbox function.
      algo: Blackbox algorithm to perform optimization and suggest points for
        evaluation.
      logdir: Logging directory path. Checkpoints and videos are saved here.
      save_rate: Number of iterations between saving checkpoints.
      eval_rate: Number of iterations between evaluation.
      evaluator: Node for evaluating current params and saving checkpoints.
      warmstartdir: Directory path for fetching the warmstart policy. The last
        chckpoint will automatically be used.
      record_video_during_eval: Whether to save video while reporting
        performance of current checkpoint. Video can only be recorded if the
        worker supports it.
      num_videos: Number of videos to record. More than 1 videos might have to
        be recorded for randomized blackbox function. Eg. in case of randomized
        RL environments.
      video_framerate: Framerate of videos to record.
      video_extension: Extension of video files to save.
      metrics: Labels of metrics to report to Xmanager.
      send_training_state_to_workers: Whether to send training state to workers.
      send_iteration_to_workers: Whether to send current iteration number to
        workers.
      reschedule_failed_request: Whether to requeue a failed request after error
        callback on a worker.
      merge_fn: Function that merges a sequence of evaluation results from the
        workers into a single evaluation result.
    """
    self._workers = workers
    self._num_workers = len(workers)
    self._algorithm = algo
    self._evaluator = evaluator
    self._num_iterations = num_iterations
    self._num_evals_per_suggestion = num_evals_per_suggestion
    self._save_rate = save_rate
    self._eval_rate = eval_rate
    self._record_video_during_eval = record_video_during_eval
    self._num_videos = num_videos
    self._video_framerate = video_framerate
    self._video_extension = video_extension
    self._metrics = metrics or ["value"]
    self._send_training_state_to_workers = send_training_state_to_workers
    self._send_iteration_to_workers = send_iteration_to_workers
    self._reschedule_failed_request = reschedule_failed_request
    self._merge_fn = merge_fn
    self._training_state = None

    self._warmstartdir = warmstartdir
    user_datatable_name = ""
    self._logdir = logdir
    if not gfile.exists(self._logdir):
      gfile.makedirs(self._logdir)

    self._logger = logger.make_logger(
        label="iris",
        user_datatable_name=user_datatable_name,
        time_delta=0.2,
    )
    self.reset([], -1)
    self._evaluator_lock = threading.Lock()
    self._evaluation_in_progress = False
    self._evaluator_lock = threading.Lock()
    self._evaluation_in_progress = False
    logging.set_verbosity(10)

  def reset(self, suggestions: Sequence[Mapping[str, Any]], iteration: int):
    self._iteration = iteration
    self._suggestions = suggestions
    num_suggestions = len(suggestions)
    self._futures = []
    self._evaluations = {}
    self._aggregate_evaluations = [
        worker_util.EvaluationResult(np.empty(0), 0)  # pytype: disable=wrong-arg-types  # numpy-scalars
    ] * num_suggestions
    self._requests_to_send = []

  def _send_first_round_requests(self) -> None:
    """Send one evaluation request to each worker."""

    # Queue all evaluation requests.
    self._requests_to_send.extend(
        list(enumerate(self._suggestions)) * self._num_evals_per_suggestion
    )

    logging.debug("%d workers available to send requests.", len(self._workers))

    # send a request from queue to each worker
    for i in range(self._num_workers):
      self._send_next_request_to_worker(i)

  def _send_next_request_to_worker(self, worker_id: int) -> None:
    """Send the next eval request in queue to given worker."""

    next_request = self._get_next_request()
    if next_request is None:
      logging.debug("No more requests left to send")
      return
    suggestion_id, suggestion = next_request
    worker_kwargs = {**suggestion}
    if self._send_training_state_to_workers:
      worker_kwargs["training_state"] = self._training_state
    if self._send_iteration_to_workers:
      worker_kwargs["iteration"] = self._iteration
    worker_kwargs = {k: pkl.dumps(v) for k, v in worker_kwargs.items()}
    future = self._workers[worker_id].futures.work_with_serialized_inputs(
        **worker_kwargs
    )
    callback = self._create_future_callback(
        worker_id, suggestion_id, self._iteration
    )
    self._futures.append(
        CallbackInfo(
            callback=callback,
            future=future,
            worker_id=worker_id,
            suggestion_id=suggestion_id,
            iteration=self._iteration
        )
    )

  def _get_next_request(self) -> tuple[int, Mapping[str, Any]] | None:
    """Get next request from the queue."""

    if not self._requests_to_send:
      return None
    return self._requests_to_send.pop()

  def _create_future_callback(
      self, worker_id: int, suggestion_id: int, iteration: int
  ) -> Callable[[futures.Future[Any]], None]:
    """Returns callback function for workers.

    NOTE: The callback function will not create `suggestion_id` as a key in
    self._evaluations if a worker fails to return a correct evaluation, which
    can break `self._aggregate_results()`. The ways to mitigate this issue are:

    1. Increase config's num_evals_per_suggestion.
    2. Add redundant workers. Currently # workers = # batchsize of suggestions.
    3. Manually stop and restart the run on XManager via "Stop" and "Play"
    buttons.
    4. Use workers in PROD priority (to reduce preemption risk).

    Trying to fix this issue 100% would require the coordinator to infinite-loop
    in waiting for a result, which may be bad for debugging workers.

    Args:
      worker_id: ID of the worker.
      suggestion_id: ID of the suggestion.
      iteration: Current iteration counter.

    Returns:
      Callback function.
    """

    def callback(future: futures.Future[Any]) -> None:
      if iteration != self._iteration:
        logging.info("Received result from an old iteration.")
        return
      if future.cancelled():
        logging.info(
            "Cancelled request for suggestion id %d on worker %d",
            suggestion_id,
            worker_id,
        )
        return
      if future.exception():
        logging.debug(
            "Error occurred while evaluating suggestion id %d on worker %d",
            suggestion_id,
            worker_id,
        )
        logging.error(future.exception())
        if self._reschedule_failed_request:
          logging.debug(
              "Putting suggestion id %d back in request queue.", suggestion_id
          )
          self._requests_to_send.append(
              (suggestion_id, self._suggestions[suggestion_id])
          )
        return

      # collect evaluation result from worker
      if suggestion_id not in self._evaluations:
        self._evaluations[suggestion_id] = []
      self._evaluations[suggestion_id].append(future.result())

      logging.debug(
          "1 eval received for suggestion ID %d from worker %d.",
          suggestion_id,
          worker_id,
      )

      # send a new request to this worker
      self._send_next_request_to_worker(worker_id)

    return callback

  def _aggregate_results(self, suggestion_id: int) -> int:
    """Collect and merge evaluations for the given suggestion ID.

    Modifies self._aggregate_evaluations at the suggestion ID by merging all its
    evaluations using self._merge_fn.

    Arguments:
      suggestion_id: ID of the suggestion.

    Returns:
      Number of evaluations that were aggregated for this suggestion.
    """
    if suggestion_id not in self._evaluations:
      logging.warning(
          "Evaluation for suggestion ID %d was not returned by the "
          "worker. Skipping this suggestion for iteration %d",
          suggestion_id,
          self._iteration,
      )
      return 0
    num_evals = len(self._evaluations[suggestion_id])
    result_list = self._evaluations[suggestion_id]
    merged_result = self._merge_fn(result_list)
    self._aggregate_evaluations[suggestion_id] = merged_result
    return num_evals

  def _get_evals(self) -> None:
    """Send eval requests run future callbacks and collect results."""
    stime = time.time()
    self._send_first_round_requests()
    logging.debug(
        "Sent first round of req in %f sec for iteration %d.",
        time.time() - stime,
        self._iteration,
    )
    stime = time.time()

    for callback_info in self._futures:
      logging.debug(
          "Starting callback for suggestion ID %d on worker %d for "
          "iteration %d.",
          callback_info.suggestion_id,
          callback_info.worker_id,
          callback_info.iteration,
      )
      st = time.time()
      callback_info.callback(callback_info.future)
      t = time.time() - st
      logging.debug(
          "Callback for suggestion ID %d on worker %d for iteration "
          "%d finished in %f seconds.",
          callback_info.suggestion_id,
          callback_info.worker_id,
          callback_info.iteration,
          t,
      )

    logging.debug(
        "Ran all futures in %f sec for iteration %d.",
        time.time() - stime,
        self._iteration,
    )
    stime = time.time()

    failed_suggestions = 0
    for i in range(len(self._suggestions)):
      st = time.time()
      num_evals = self._aggregate_results(i)
      if not num_evals:
        failed_suggestions += 1
      logging.debug(
          "Aggregated results for suggestion %d in %f sec for iteration %d.",
          i,
          time.time() - st,
          self._iteration,
      )
      logging.debug(
          "Got %d evals for suggestion %d for iteration %d.",
          num_evals,
          i,
          self._iteration,
      )

    logging.debug(
        "%d out of %d suggestions failed for iteration %d.",
        failed_suggestions,
        len(self._suggestions),
        self._iteration,
    )
    logging.debug(
        "Aggregated all results in %f sec for iteration %d.",
        time.time() - stime,
        self._iteration,
    )

  def _log_progress(self) -> None:
    """Log evaluation metrics to XManager measurements."""
    measurement = {"steps": self._iteration}
    for metric_name in self._metrics:
      metric_values = []
      for evaluation in self._aggregate_evaluations:
        if hasattr(evaluation, metric_name):
          metric_values.append(getattr(evaluation, metric_name))
        elif (
            evaluation.metrics is not None and metric_name in evaluation.metrics
        ):
          metric_values.append(evaluation.metrics[metric_name])
      if not metric_values:
        logging.warning(
            "Metric %s not found in evaluation result.", metric_name
        )
      else:
        measurement[metric_name] = np.mean(metric_values)
    self._training_state = measurement
    self._logger.write(measurement)

  def training_state(self):
    return self._training_state

  def _save_checkpoint(self, state) -> None:
    checkpoint_path = self._logdir.joinpath(
        "iteration_{:04d}".format(self._iteration)
    )
    logging.info("Saving checkpoints to %s...", checkpoint_path)
    checkpoint_util.save_checkpoint(checkpoint_path, state)
    self._algorithm.maybe_save_custom_checkpoint(
        state=state, checkpoint_path=checkpoint_path
    )

  def _restore_checkpoint(
      self, logdir: pathlib.Path, max_allowed_iteration_for_restart: int = 50
  ) -> tuple[dict[str, Any] | None, int]:
    """Load the state from the latest checkpoint found in logdir.

    Args:
      logdir: Logging directory path to be checked.
      max_allowed_iteration_for_restart: if the checkpoint has fewer total
        iteration than this value and all checkpoints fail to load, restart the
        training. Otherwise through an error.

    Returns:
      algorithm state, iteration number
    """
    # If specific checkpoint is provided
    try:
      state = checkpoint_util.load_checkpoint_state(logdir.as_posix())
      iteration = 0  # No iteration information is extracted
      return state, iteration
    except (ValueError, FileNotFoundError):
      logging.warning(
          "Failed to load directly as a checkpoint, try searching subfolders"
          " with checkpoints."
      )

    # If the logging directory is provided
    checkpoint_paths = gfile.glob(str(logdir.joinpath("iteration_*")))
    logging.debug(
        "restore_checkpoint from logdir: %s, checkpoint_paths: %s",
        logdir,
        checkpoint_paths,
    )
    if checkpoint_paths:
      checkpoint_load_error = None
      checkpoint_paths_sorted = sorted(
          checkpoint_paths, key=lambda x: int(x.split("_")[-1]), reverse=True
      )
      for checkpoint_path in checkpoint_paths_sorted:
        try:
          logging.info("Found checkpoint %s", checkpoint_path)
          state = checkpoint_util.load_checkpoint_state(checkpoint_path)
          iteration = int(checkpoint_path.split("_")[-1]) + 1
          return state, iteration
        except ValueError as e:
          logging.warning(
              "Found a policy weights directory %s, but failed to load, try an"
              " earlier one.",
              (checkpoint_path),
          )
          checkpoint_load_error = e
      # If all checkpoints fail to load and training has been done for more than
      # max_allowed_iteration_for_restart iterations, throw error and stop the
      # training.
      latest_checkpoint_num_iterations = (
          int(checkpoint_paths_sorted[0].split("_")[-1]) + 1
      )
      if latest_checkpoint_num_iterations > max_allowed_iteration_for_restart:
        raise checkpoint_load_error
    return None, 0

  def evaluate(
      self, iteration: int, suggestions: Sequence[Mapping[str, Any]] | bytes
  ) -> None:
    """Evaluate given state."""
    with self._evaluator_lock:
      self._evaluation_in_progress = True
      if isinstance(suggestions, bytes):
        suggestions = pkl.loads(suggestions)
      self.reset(suggestions, iteration)
      self._get_evals()
      self._log_progress()
      if self._record_video_during_eval:
        video_path = self._logdir.joinpath(
            "iteration_{:04d}".format(self._iteration)
        )
        for i in range(min(self._num_videos, len(self._workers))):
          worker_kwargs = dict(
              **suggestions[0],
              record_video=True,
              video_framerate=self._video_framerate,
              video_path=str(
                  video_path.joinpath(f"video_{i}.{self._video_extension}")
              ),
          )
          worker_kwargs = {k: pkl.dumps(v) for k, v in worker_kwargs.items()}
          self._workers[i].work_with_serialized_inputs(**worker_kwargs)
      self._evaluation_in_progress = False

  def save(self, iteration: int, state: dict[str, Any] | bytes) -> None:
    """Save checkpoint."""
    with self._evaluator_lock:
      self._evaluation_in_progress = True
      self.reset([], iteration)
      state = pkl.loads(state) if isinstance(state, bytes) else state
      self._save_checkpoint(state)
      self._evaluation_in_progress = False

  def step(self, iteration: int) -> None:
    """Single iteration step of blackbox optimization.

    For each iteration, parameter suggestions are obtained from the blackbox
    optimization algorithm. The suggestions are sent to the workers for
    evaluation and the algorithm processes the results.

    Args:
      iteration: Iteration number.
    """
    stime0 = time.time()
    if self._evaluator is not None:
      self._training_state = self._evaluator.training_state()
    logging.debug("Retrieved training state in %s sec.", time.time() - stime0)
    stime = time.time()
    if iteration % self._save_rate == 0:
      if self._evaluator is not None:
        self._evaluator.futures.save(
            iteration=iteration, state=pkl.dumps(self._algorithm.state)
        )
      else:
        self.save(iteration=iteration, state=self._algorithm.state)
    logging.debug("Send save request in %s sec.", time.time() - stime)
    stime = time.time()
    if iteration % self._eval_rate == 0:
      suggestions = self._algorithm.get_param_suggestions(evaluate=True)
      logging.debug(
          "Evaluate iteration #: %s, suggestions: %s",
          iteration,
          len(suggestions),
      )
      # The "coordinator" uses an "evaluator" CourierNode to offload the eval
      # computation. The "evaluator" has its own set of "eval_workers". When the
      # evaluator is None, the "coordinator" uses its own workers to evaluate
      # as well. Thus the coordinator uses its workers to get the returns for
      # the suggestions as well as to evaluate the optimum suggestion. This is a
      # requirement for on robot training/ evaluating, as both the training and
      # evaluating phases need to share the same environment that binds with the
      # hardware.
      if self._evaluator is not None:
        self._evaluator.futures.evaluate(
            iteration=iteration, suggestions=pkl.dumps(suggestions)
        )
      else:
        self.evaluate(iteration=iteration, suggestions=suggestions)

    logging.debug("Sent evaluation request in %s sec.", time.time() - stime)
    stime = time.time()

    suggestions = self._algorithm.get_param_suggestions()

    logging.debug("Retrieved suggestions in %s sec.", time.time() - stime)
    stime = time.time()
    logging.debug(
        "iteration #: %s, suggestions: %s", iteration, len(suggestions)
    )

    self.reset(suggestions, iteration)

    logging.debug("Reset in %s sec.", time.time() - stime)
    stime = time.time()

    self._get_evals()

    logging.debug("Get worker evals in %s sec.", time.time() - stime)
    stime = time.time()

    try:
      self._algorithm.process_evaluations(self._aggregate_evaluations.copy())
    except ValueError:
      logging.exception("An error occurred in process evaluations.")

    logging.debug("Process evaluations in %s sec.", time.time() - stime)
    logging.debug(
        "Done iteration #: %d in %s sec.",
        self._iteration,
        time.time() - stime0,
    )

  def _get_init_state(self) -> dict[str, Any]:
    """Retrieve initial state from the first responding worker.

    Returns:
      Initial state dictionary fom the worker.
    """
    init_state = None
    i = 0
    while init_state is None:
      try:
        init_state = self._workers[i].get_init_state()
      except ValueError as e:
        logging.warning(
            "Worker %s initialization failed with error %s. Trying the next "
            "worker.", i, e
        )
        i = (i + 1) % self._num_workers
    return init_state  # pytype: disable=bad-return-type

  def initialize_algorithm_state(self) -> int:
    """Initialize algorithm state, potentially loading an existing checkpoint.

    Returns:
      iteration number of the restored state

    Raises:
      ValueError: If a warmstart was requested, but no state was found.
    """
    self._algorithm.initialize(self._get_init_state())

    state, iteration = self._restore_checkpoint(self._logdir)
    if iteration == 0 and self._warmstartdir:
      state, _ = self._restore_checkpoint(self._warmstartdir)
      if state is None:
        raise ValueError(
            f"Specified warmstartdir {self._warmstartdir}, but no checkpoint"
            " was loaded."
        )
    if state:
      self._algorithm.restore_state_from_checkpoint(state)
    return iteration

  def run(self) -> None:
    """Main iteration loop for blackbox optimization."""

    iteration = self.initialize_algorithm_state()

    for i in range(iteration, self._num_iterations):
      self.step(i)

    if self._evaluator is not None:
      while self._evaluator.evaluation_in_progress():
        time.sleep(1)
    else:
      while self.evaluation_in_progress():
        time.sleep(1)

    self._logger.close()
    lp.stop()

  def evaluation_in_progress(self):
    return self._evaluation_in_progress
