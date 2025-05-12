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

import imageio  # pylint: disable=unused-import
import numpy as np


class VideoRecorder:
  """Utility for recording videos."""

  def __init__(self, video_path: str, fps: int = 30):
    self._video_path = video_path
    self._fps = fps
    self._recorder = imageio.get_writer(
        video_path, fps=fps, format="FFMPEG", codec="libx264"
    )

  def add_frame(self, data: np.ndarray):
    self._recorder.append_data(data)

  def end_video(self):
    self._recorder.close()
