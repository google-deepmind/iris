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

from iris import checkpoint_util
from absl.testing import absltest

_TEST_CHECKPOINT = "./testdata/test_checkpoint.pkl"


class CheckpointUtilTest(absltest.TestCase):

  def test_load_checkpoint(self):
    state = checkpoint_util.load_checkpoint_state(_TEST_CHECKPOINT)
    self.assertIsNotNone(state["params_to_eval"])


if __name__ == "__main__":
  absltest.main()
