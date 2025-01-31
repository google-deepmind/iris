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

import enum
import functools
from typing import Any, Callable, Mapping

from iris.maml import adaptation_optimizers
from iris.workers import worker
from iris.workers import worker_util


@enum.unique
class AdaptationType(enum.Enum):
  HILLCLIMB = 1
  GRADIENT = 2


class MAMLWorker(worker.Worker):
  """Makes multiple calls of another worker's work() function for adaptation.

  `worker_kwargs` will be passed into the worker's constructor, but it's best
  practice to externally wrap the constructor with functools.partial.
  """

  def __init__(
      self,
      worker_constructor: Callable[[], worker.Worker],
      adaptation_type: AdaptationType,
      adaptation_kwargs: Mapping[str, Any],
      **worker_kwargs
  ) -> None:
    super().__init__(**worker_kwargs)
    self._worker = worker_constructor(**worker_kwargs)
    self._adaptation_type = adaptation_type
    self._init_state = self._worker._init_state

    if self._adaptation_type is AdaptationType.HILLCLIMB:
      self._adaptation_optimizer = adaptation_optimizers.HillClimbAdaptation(
          **adaptation_kwargs
      )
    elif self._adaptation_type is AdaptationType.GRADIENT:
      self._adaptation_optimizer = adaptation_optimizers.GradientAdaptation(
          **adaptation_kwargs
      )

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
        params_to_eval=params_to_eval, work_fn=work_fn
    )
    merged_result = worker_util.merge_eval_results(
        total_results
    )  # collecting obs buffer

    return worker_util.EvaluationResult(  # pytype: disable=wrong-arg-types  # numpy-scalars
        params_evaluated=params_to_eval,
        value=adapted_value,
        obs_norm_buffer_data=merged_result.obs_norm_buffer_data,
    )
