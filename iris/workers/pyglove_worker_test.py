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

from iris.workers import pyglove_worker
import numpy as np
import pyglove as pg

from absl.testing import absltest


class PygloveWorkerTest(absltest.TestCase):

  def test_pyglove_worker(self):
    dna_spec = pg.dna_spec(pg.floatv(-1.0, 1.0))
    worker_obj = pyglove_worker.PyGloveWorker(
        dna_spec=dna_spec, blackbox_function=lambda dna: dna.value, worker_id=0
    )
    dna = pg.random_dna(dna_spec)
    serialized_dna = pg.to_json_str(dna)

    result = worker_obj.work(
        metadata=serialized_dna, params_to_eval=np.empty((), dtype=np.float64)
    )
    self.assertIsInstance(result.value, float)


if __name__ == "__main__":
  absltest.main()
