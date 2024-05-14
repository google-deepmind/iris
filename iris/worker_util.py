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

"""Contains utilities for Worker classes."""

import dataclasses
import itertools
from typing import Dict, Optional, Sequence
from iris import normalizer
import numpy as np


@dataclasses.dataclass()
class EvaluationResult():
  """A class for holding blackbox function evaluation result."""
  params_evaluated: np.ndarray
  value: np.float64
  obs_norm_buffer_data: Optional[Dict[str, np.ndarray]] = None
  metadata: Optional[str] = None
  metrics: Optional[Dict[str, np.float64]] = None


def merge_eval_results(results: Sequence[EvaluationResult]) -> EvaluationResult:
  """Merges evaluation results when a parameter is evaluated multiple times."""

  if not results:
    raise ValueError("Cannot merge an empty list of EvaluationResults.")

  if len(results) == 1:
    return results[0]

  merged_value = np.mean([r.value for r in results])

  merged_obs_norm_buffer_data = None
  if results[0].obs_norm_buffer_data is not None:
    merged_buffer = normalizer.MeanStdBuffer()
    merged_buffer.data = results[0].obs_norm_buffer_data
    for result in itertools.islice(results, 1, None):
      merged_buffer.merge(result.obs_norm_buffer_data)
    merged_obs_norm_buffer_data = merged_buffer.data

  if results[0].metrics is not None:
    merged_metrics = {}
    for metric_name in results[0].metrics:
      merged_metrics[metric_name] = np.mean(
          [result.metrics[metric_name] for result in results])
  else:
    merged_metrics = {}

  return EvaluationResult(results[0].params_evaluated,
                          merged_value,
                          merged_obs_norm_buffer_data,
                          results[0].metadata,
                          merged_metrics)
