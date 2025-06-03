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

"""Makes multiple calls of another worker's work() function for adaptation."""

import functools
from typing import Any, Callable, Literal, Sequence, Tuple, Union

from iris.workers import worker
from iris.workers import worker_util
import numpy as np


FloatLike = Union[float, np.float32, np.float64]


def _multiple_eval(
    params_to_eval: np.ndarray,
    num_evals: int,
    work_fn: Callable[[np.ndarray], worker_util.EvaluationResult],
    **work_kwargs,
) -> Tuple[float, Sequence[worker_util.EvaluationResult]]:
  """Evaluates parameters multiple times and averages results."""
  results = [work_fn(params_to_eval, **work_kwargs) for _ in range(num_evals)]
  return np.mean([r.value for r in results]), results


# TODO: Potentially make this a subclass of BlackboxAlgorithm.
class Adaptation(object):
  """Base class for all adaptation methods."""

  def __init__(self, random_seed: int = 0) -> None:
    self._np_random_state = np.random.RandomState(random_seed)

  def run_adaptation(
      self,
      params_to_eval: np.ndarray,
      work_fn: Callable[[np.ndarray], worker_util.EvaluationResult],
  ) -> Tuple[float, Sequence[worker_util.EvaluationResult]]:
    """Runs adaptation method using "work" function given a starting input.

    Args:
      params_to_eval: Initial input for adaptation.
      work_fn: Objective function.

    Returns:
      Final value of adapted parameter, along with all results collected.
    """
    raise NotImplementedError('Abstract method')


class GradientAdaptation(Adaptation):
  """Performs gradient-based adaptation techniques."""

  def __init__(
      self,
      num_iterations: int = 1,
      num_iteration_suggestions: int = 20,
      num_adapted_evals: int = 1,
      std: float = 0.01,
      step_size: float = 0.05,
      top_percentage: float = 1.0,
      **kwargs,
  ) -> None:
    """Performs a mini version of AugmentedRandomSearch algorithm.

    Args:
      num_iterations: How many gradient steps to run.
      num_iteration_suggestions: How many evaluations to use per gradient step.
      num_adapted_evals: How many evaluations of adapted parameter to average.
      std: Standard deviation for normal perturbations around current
        optimization parameter vector.
      step_size: Step size for gradient ascent.
      top_percentage: Fraction of top performing perturbations to use for
        gradient estimation.
      **kwargs: Other keyword arguments for base class.
    """

    super().__init__(**kwargs)
    self._num_iterations = num_iterations
    self._num_iteration_suggestions = num_iteration_suggestions
    self._num_adapted_evals = num_adapted_evals
    self._std = std
    self._step_size = step_size
    self._top_percentage = top_percentage
    self._num_top = int(self._top_percentage * self._num_iteration_suggestions)

  def run_adaptation(
      self,
      params_to_eval: np.ndarray,
      work_fn: Callable[[np.ndarray], worker_util.EvaluationResult],
  ) -> Tuple[float, Sequence[worker_util.EvaluationResult]]:
    """Runs Gradient-based adaptation method."""
    total_results = []
    params_so_far = params_to_eval

    for _ in range(self._num_iterations):
      params_so_far, step_results = self._iteration_step(
          params_so_far, work_fn=work_fn
      )
      total_results += step_results

    adapted_value, adapted_result_list = _multiple_eval(
        params_so_far, self._num_adapted_evals, work_fn
    )
    total_results += adapted_result_list

    return adapted_value, total_results

  def _iteration_step(
      self,
      params_to_eval: np.ndarray,
      work_fn: Callable[[np.ndarray], worker_util.EvaluationResult],
  ) -> Tuple[np.ndarray, Sequence[worker_util.EvaluationResult]]:
    """Performs standard ES-gradient estimation."""
    dimensions = params_to_eval.shape[0]
    directions = self._np_random_state.normal(
        0, 1, (self._num_iteration_suggestions, dimensions)
    )

    param_suggestions = np.vstack([
        params_to_eval + self._std * directions,
        params_to_eval - self._std * directions,
    ])
    eval_results = [work_fn(params) for params in param_suggestions]

    # Get top evaluation results
    evals = np.array([r.value for r in eval_results])
    pos_evals = evals[: self._num_iteration_suggestions]
    neg_evals = evals[self._num_iteration_suggestions :]
    max_evals = np.max(np.vstack([pos_evals, neg_evals]), axis=0)
    idx = (-max_evals).argsort()[: self._num_top]
    pos_evals = pos_evals[idx]
    neg_evals = neg_evals[idx]
    all_top_evals = np.hstack([pos_evals, neg_evals])
    evals = pos_evals - neg_evals

    # Get delta directions corresponding to top evals
    directions = directions[idx, :]

    # Estimate gradients
    gradient = np.dot(evals, directions) / evals.shape[0]
    if not np.isclose(np.std(all_top_evals), 0.0):
      gradient /= np.std(all_top_evals)

    # Apply gradients
    return params_to_eval + self._step_size * gradient, eval_results


class HillClimbAdaptation(Adaptation):
  """Performs variations of Hill-Climbing."""

  def __init__(
      self,
      parallel_alg: Literal['batch', 'average'] = 'batch',
      num_iterations: int = 20,
      std: float = 0.05,
      num_iteration_suggestions: int = 1,
      num_adapted_evals: int = 1,
      num_meta_evals: int = 1,
      **kwargs,
  ) -> None:
    """Initializes parameters for Hill-Climbing algorithm.

    Args:
      parallel_alg: Which algorithm to use.
      num_iterations: How many parameter updates throughout algorithm.
      std: Standard deviation for normal perturbations around current
        optimization parameter vector.
      num_iteration_suggestions: How many evaluations before updating parameter.
      num_adapted_evals: How many evaluations of adapted parameter to average.
      num_meta_evals: How many evaluations of initial meta parameters.
      **kwargs: Other keyword arguments for base class.
    """

    super().__init__(**kwargs)
    self._parallel_alg = parallel_alg
    self._num_iterations = num_iterations
    self._std = std
    self._num_iteration_suggestions = num_iteration_suggestions
    self._num_adapted_evals = num_adapted_evals
    self._num_meta_evals = num_meta_evals

  def run_adaptation(
      self,
      params_to_eval: np.ndarray,
      work_fn: Callable[[np.ndarray], worker_util.EvaluationResult],
      meta_value: Union[FloatLike, None] = None,
  ) -> Tuple[float, Sequence[worker_util.EvaluationResult]]:
    """Runs Hill-Climb-based adaptation method."""

    total_results = []
    best_params = params_to_eval

    if meta_value:
      pivot_value = meta_value
    else:
      meta_value_list = []
      for _ in range(self._num_meta_evals):
        meta_result = work_fn(params_to_eval)
        total_results.append(meta_result)
        meta_value_list.append(meta_result.value)
      pivot_value = np.mean(meta_value_list)

    for _ in range(self._num_iterations):
      if self._parallel_alg == 'average':
        potential_best_params, potential_pivot_value, eval_results = (
            self._average_iteration_step(
                params_to_eval=best_params, work_fn=work_fn
            )
        )
      elif self._parallel_alg == 'batch':
        potential_best_params, potential_pivot_value, eval_results = (
            self._batch_iteration_step(
                params_to_eval=best_params, work_fn=work_fn
            )
        )
      else:
        raise ValueError(f'Unknown parallel algorithm: {self._parallel_alg}')

      total_results += eval_results
      if potential_pivot_value > pivot_value:
        best_params = potential_best_params
        pivot_value = potential_pivot_value

    adapted_value, adapted_result_list = _multiple_eval(
        best_params, self._num_adapted_evals, work_fn
    )
    total_results += adapted_result_list

    return adapted_value, total_results

  def _batch_iteration_step(
      self,
      params_to_eval: np.ndarray,
      work_fn: Callable[[np.ndarray], worker_util.EvaluationResult],
  ) -> Tuple[np.ndarray, float, Sequence[worker_util.EvaluationResult]]:
    """Takes a batch of perturbations and returns the (noisy) argmax."""

    dimensions = params_to_eval.shape[0]
    batch_params_list = []
    iteration_value_list = []
    eval_results = []
    for _ in range(self._num_iteration_suggestions):
      perturbation = self._np_random_state.normal(size=(dimensions)) * self._std
      temp_params = params_to_eval + perturbation
      batch_params_list.append(temp_params)

      iteration_result = work_fn(temp_params)
      eval_results.append(iteration_result)
      iteration_value_list.append(iteration_result.value)

    best_index = np.argmax(iteration_value_list)
    potential_pivot_value = iteration_value_list[best_index]
    potential_best_params = batch_params_list[best_index]
    return potential_best_params, potential_pivot_value, eval_results

  def _average_iteration_step(
      self,
      params_to_eval: np.ndarray,
      work_fn: Callable[[np.ndarray], worker_util.EvaluationResult],
  ) -> Tuple[np.ndarray, float, Sequence[worker_util.EvaluationResult]]:
    """Takes a random perturbation and averages multiple evaluations."""

    dimensions = params_to_eval.shape[0]
    perturbation = self._np_random_state.normal(size=(dimensions)) * self._std
    potential_best_params = params_to_eval + perturbation

    potential_pivot_value, eval_results = _multiple_eval(
        params_to_eval=potential_best_params,
        num_evals=self._num_iteration_suggestions,
        work_fn=work_fn,
    )

    return potential_best_params, potential_pivot_value, eval_results


class MAMLWorker(worker.Worker):
  """Makes multiple calls of another worker's work() function for adaptation."""

  def __init__(
      self,
      worker_constructor: Callable[..., worker.Worker],
      adaptation_constructor: Callable[[], Adaptation],
      **worker_kwargs,
  ) -> None:
    super().__init__(**worker_kwargs)
    self._worker = worker_constructor(**worker_kwargs)
    self._adaptation_optimizer = adaptation_constructor()
    self._init_state = self._worker._init_state

  def work(
      self, params_to_eval: Any, **work_kwargs  # pytype: disable=signature-mismatch  # overriding-parameter-count-checks
  ) -> worker_util.EvaluationResult:
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
        params_to_eval=params_to_eval, work_fn=work_fn
    )

    merged_result = worker_util.merge_eval_results(total_results)
    return worker_util.EvaluationResult(
        params_evaluated=params_to_eval,  # original meta-point
        value=adapted_value,  # adapted point value
        obs_norm_buffer_data=merged_result.obs_norm_buffer_data,
        metadata=merged_result.metadata,
        metrics=merged_result.metrics,
    )
