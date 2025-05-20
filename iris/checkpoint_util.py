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

"""Utilities for interacting with Iris checkpoints."""

import datetime
import pathlib
import pickle as pkl
from typing import Any

from absl import logging


def load_checkpoint_state_oss(path: str | pathlib.Path) -> dict[str, Any]:
  """Loads a checkpoint state from a given path."""
  path = pathlib.Path(path)
  if path.is_dir():
    path = path.joinpath("checkpoint.pkl")
  try:
    with open(path, "rb") as f:
      state = pkl.load(f)
    return state
  except IsADirectoryError as e:
    raise ValueError(f"Path {path} is not a valid checkpoint file.") from e


load_checkpoint_state = load_checkpoint_state_oss


def save_checkpoint_oss(path: str | pathlib.Path, state: dict[str, Any]):
  """Saves a checkpoint state to a given path."""
  path = pathlib.Path(path)
  path.mkdir(parents=True, exist_ok=True)
  path = path.joinpath("checkpoint.pkl")
  with open(path, "wb") as f:
    pkl.dump(state, f)


save_checkpoint = save_checkpoint_oss
