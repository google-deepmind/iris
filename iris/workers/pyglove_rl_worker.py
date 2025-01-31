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

"""Subclass of RLWorker that allows PyGlove DNAs."""

from typing import Callable, Optional, Union

from iris.policies import nas_policy
from iris.workers import rl_worker
from iris.workers import worker_util
import pyglove as pg


class PyGloveRLWorker(rl_worker.RLWorker):
  """Subclass of RLWorker that allows PyGlove DNAs.

  NOTE: This only works with ES-ENAS.
  """

  def __init__(
      self,
      policy: Union[
          nas_policy.PyGlovePolicy, Callable[..., nas_policy.PyGlovePolicy]
      ],
      **kwargs
  ) -> None:
    super().__init__(policy=policy, **kwargs)
    self._init_state["serialized_dna_spec"] = pg.to_json_str(
        self._policy.dna_spec  # pytype: disable=attribute-error
    )

  def work(
      self, metadata: Optional[str] = None, **kwargs
  ) -> worker_util.EvaluationResult:
    if metadata:
      dna = pg.from_json_str(metadata)
      self._policy.update_dna(dna)  # pytype: disable=attribute-error
    vanilla_evaluation_result = super().work(**kwargs)
    evaluation_result = worker_util.EvaluationResult(
        params_evaluated=vanilla_evaluation_result.params_evaluated,
        value=vanilla_evaluation_result.value,
        obs_norm_buffer_data=vanilla_evaluation_result.obs_norm_buffer_data,
        metadata=metadata,
    )
    return evaluation_result
