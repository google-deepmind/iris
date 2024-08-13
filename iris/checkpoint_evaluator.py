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

"""Evaluates the learned blackbox parameters from a saved checkpoint."""

import collections
import os
import time

from absl import app
from absl import flags
from absl import logging
from iris import checkpoint_util
from ml_collections import config_flags
import numpy as np
from tensorflow.io import gfile


FLAGS = flags.FLAGS
_CHECKPOINT_FILE = flags.DEFINE_string("checkpoint_file", None,
                                       "The file name of the checkpoint.")
_NUM_EVALUATIONS = flags.DEFINE_integer("num_evaluations", 1,
                                        "The number of evaluations.")
_RECORD_VIDEO = flags.DEFINE_bool("record_video", False,
                                  "Whether to record video.")
_VIDEO_PATH = flags.DEFINE_string("video_path", None,
                                  "The path for saving recorded video.")
_VIDEO_FRAMERATE = flags.DEFINE_integer("video_framerate", 10,
                                        "The video framerate.")
config_flags.DEFINE_config_file("config", "path/to/config",
                                "Configuration file.")


def main(argv):
  del argv
  worker_config = FLAGS.config.worker
  if "write_to_replay" in worker_config.worker_args:
    worker_config.worker_args.write_to_replay = False
  worker = worker_config["worker_class"](
      worker_id=0, **worker_config["worker_args"])
  state = checkpoint_util.load_checkpoint_state(_CHECKPOINT_FILE.value)
  returns = []
  times = []
  metric_dict = collections.defaultdict(list)
  for i in range(_NUM_EVALUATIONS.value):
    logging.info("Evaluation #: %d", i)
    st = time.time()
    if _RECORD_VIDEO.value:
      if not gfile.Exists(_VIDEO_PATH.value):
        gfile.MakeDirs(
            _VIDEO_PATH.value, mode=gfile.LEGACY_GROUP_WRITABLE_WORLD_READABLE
        )
      video_path = os.path.join(_VIDEO_PATH.value, "video_" + str(i) + ".mp4")
      result = worker.work(
          **state,
          enable_logging=True,
          record_video=_RECORD_VIDEO.value,
          video_path=video_path,
          video_framerate=_VIDEO_FRAMERATE.value)
    else:
      result = worker.work(**state, enable_logging=True)
    ep_time = time.time() - st
    times.append(ep_time)
    logging.info("Episode time: %f sec", ep_time)
    returns.append(result.value)
    if result.metrics is not None:
      for metric_name, metric_value in result.metrics.items():
        metric_dict[metric_name].append(metric_value)

  logging.info("Mean return: %f", np.mean(returns))
  logging.info("Std return: %f", np.std(returns))
  logging.info("Mean time: %f", np.mean(times))

  for metric_name, metric_values in metric_dict.items():
    logging.info("Mean %s: %f", metric_name, np.mean(metric_values))
    logging.info("Std %s: %f", metric_name, np.std(metric_values))


if __name__ == "__main__":
  app.run(main)
