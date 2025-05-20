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

"""Utility for recording videos."""

import os
import numpy as np


class VideoRecorder:
  """Utility for recording videos."""

  def __init__(self, video_path: str, fps: int = 30):
    self._video_path = video_path
    video_dir = os.path.dirname(self._video_path)
    if not os.path.exists(video_dir):
      os.makedirs(video_dir, exist_ok=True)
    self._fps = fps
    self._recorder = []

  def add_frame(self, data: np.ndarray):
    self._recorder.append(data)

  def end_video(self):
    np.save(self._video_path, self._recorder)
