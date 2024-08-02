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

"""Algorithm class for the Robust Blackbox Optimization (RBO) algorithm.

Algorithm class for the Robust Blackbox Optimization (RBO) algorithm from the
paper: Provably Robust Blackbox Optimization for Reinforcement Learning
(https://arxiv.org/abs/1903.02993) (CoRL 2021).
"""

from typing import Sequence

from iris import worker_util
from iris.algorithms import ars_algorithm
from iris.algorithms import optimizers
import numpy as np


class RBO(ars_algorithm.AugmentedRandomSearch):
  """Robust Blackbox Optimization Algorithm."""

  def __init__(self,
               regression_method: str = "ridge",
               regularizer: float = 0.01,
               **kwargs) -> None:
    """Initializes the augmented random search algorithm.

    Args:
      regression_method: type of the regression method used for grad retrieval.
        Currently supported methods include: LP-decoding ("lp"), Lasso
        regression ("lasso") and ridge regression ("ridge").
      regularizer: regression regularizer used for gradient retrieval.
      **kwargs: Other keyword arguments for base class.
    """
    super().__init__(**kwargs)
    if regression_method == "lasso":
      self._regression_method = optimizers.lasso_regression_jacobian_decoder
    elif regression_method == "lp":
      self._regression_method = optimizers.l1_jacobian_decoder
    elif regression_method == "ridge":
      self._regression_method = optimizers.ridge_regression_jacobian_decoder
    else:
      raise ValueError("Invalid regression_method")
    self._regularizer = regularizer

  def process_evaluations(
      self, eval_results: Sequence[worker_util.EvaluationResult]) -> None:
    """Processes the list of Blackbox function evaluations return from workers.

    Gradient is computed by applying a particular regression procedure. The
    current parameter vector is then updated in the gradient direction with
    specified step size.

    Args:
      eval_results: List containing Blackbox function evaluations based on the
        order in which the suggestions were sent. RBO performs gradient-based
        update, where gradient is retrieved via a regression procedure. The
        particular type of the regression procedure applied is specified in the
        constructor.
    """

    # Retrieve delta direction from the param suggestion sent for evaluation.
    pos_eval_results = eval_results[:self._num_suggestions]
    neg_eval_results = eval_results[self._num_suggestions:]
    filtered_pos_eval_results = []
    filtered_neg_eval_results = []
    for i in range(len(pos_eval_results)):
      if (pos_eval_results[i].params_evaluated.size) and (
          neg_eval_results[i].params_evaluated.size):
        filtered_pos_eval_results.append(pos_eval_results[i])
        filtered_neg_eval_results.append(neg_eval_results[i])
    params = np.array([r.params_evaluated for r in filtered_pos_eval_results])
    perturbations = params - self._opt_params
    eval_results = filtered_pos_eval_results + filtered_neg_eval_results
    pos_evals = np.array([r.value for r in filtered_pos_eval_results])
    neg_evals = np.array([r.value for r in filtered_neg_eval_results])
    evals = (pos_evals - neg_evals) / 2.0

    # Estimate gradients via regression.
    gradient = self._regression_method(
        np.transpose(perturbations), np.expand_dims(evals, 1),
        self._regularizer)
    gradient = np.reshape(gradient, (len(gradient[0])))

    # Apply gradients
    self._opt_params += self._step_size * gradient

    # Update the observation buffer
    if self._obs_norm_data_buffer is not None:
      for r in eval_results:
        self._obs_norm_data_buffer.merge(r.obs_norm_buffer_data)
