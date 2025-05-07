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

from iris import buffer as buffer_lib
import numpy as np
from absl.testing import absltest


class BufferTest(absltest.TestCase):

  def test_meanstdbuffer(self):
    buffer = buffer_lib.MeanStdBuffer((1,))
    buffer.push(np.asarray(10.0))
    buffer.push(np.asarray(11.0))

    new_buffer = buffer_lib.MeanStdBuffer((1,))
    new_buffer.data = buffer.data

    self.assertEqual(new_buffer._std, buffer._std)
    self.assertEqual(new_buffer._data['n'], buffer._data['n'])


if __name__ == '__main__':
  absltest.main()
