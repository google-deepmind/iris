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

"""Iris training logger."""

from collections.abc import Mapping

import time
from typing import Any
import weakref

import jax
import numpy as np
import tree


def tensor_to_numpy(value: Any):
  if hasattr(value, 'numpy'):
    return value.numpy()  # tf.Tensor (TF2).
  if isinstance(value, jax.Array):
    return np.asarray(value)
  return value


def to_numpy(values: Any):
  """Converts tensors in a nested structure to numpy.

  Converts tensors from TensorFlow to Numpy if needed without importing TF
  dependency.

  Args:
    values: nested structure with numpy and / or TF tensors.

  Returns:
    Same nested structure as values, but with numpy tensors.
  """
  return tree.map_structure(tensor_to_numpy, values)


def _format_key(key: str) -> str:
  """Internal function for formatting keys."""
  return key.replace('_', ' ').title()


def _format_value(value: Any) -> str:
  """Internal function for formatting values."""
  value = to_numpy(value)
  if isinstance(value, (float, np.number)):
    return f'{value:0.3f}'
  return f'{value}'


def serialize(values: Mapping[str, Any]) -> str:
  """Converts `values` to a pretty-printed string.

  This takes a dictionary `values` whose keys are strings and returns
  a formatted string such that each [key, value] pair is separated by ' = ' and
  each entry is separated by ' | '. The keys are sorted alphabetically to ensure
  a consistent order, and snake case is split into words.

  For example:

      values = {'a': 1, 'b' = 2.33333333, 'c': 'hello', 'big_value': 10}
      # Returns 'A = 1 | B = 2.333 | Big Value = 10 | C = hello'
      values_string = serialize(values)

  Args:
    values: A dictionary with string keys.

  Returns:
    A formatted string.
  """
  return ' | '.join(
      f'{_format_key(k)} = {_format_value(v)}'
      for k, v in sorted(values.items())
  )


class TerminalLogger:
  """Logs to terminal."""

  def __init__(
      self,
      label: str = '',
      time_delta: float = 0.0,
  ):
    """Initializes the logger.

    Args:
      label: label string to use when logging.
      time_delta: How often (in seconds) to write values. This can be used to
        minimize terminal spam, but is 0 by default---ie everything is written.
    """

    self._label = label and f'[{_format_key(label)}] '
    self._time = time.time()
    self._time_delta = time_delta

  def write(self, values: Mapping[str, Any]):
    now = time.time()
    values = dict(values)
    values['timestep'] = now
    if (now - self._time) > self._time_delta:
      print(f'{self._label}{serialize(values)}')
      self._time = now

  def close(self):
    pass


class AutoCloseLogger:
  """Logger which auto closes itself on exit if not already closed."""

  def __init__(self, logger):
    self._logger = logger
    # The finalizer "logger.close" is invoked in one of the following scenario:
    # 1) the current logger is GC
    # 2) from the python doc, when the program exits, each remaining live
    #    finalizer is called.
    # Note that in the normal flow, where "close" is explicitly called,
    # the finalizer is marked as dead using the detach function so that
    # the underlying logger is not closed twice (once explicitly and once
    # implicitly when the object is GC or when the program exits).
    self._finalizer = weakref.finalize(self, logger.close)

  def write(self, values: Mapping[str, Any]):
    if self._logger is None:
      raise ValueError('init not called')
    self._logger.write(values)

  def close(self):
    if self._finalizer.detach():
      self._logger.close()
    self._logger = None


def make_logger_oss(
    label: str,
    user_datatable_name: str = '',
    time_delta: float = 0.2,
):
  """Make an Acme or XData logger for BBV2.

  Args:
    label: Name to give to the logger.
    user_datatable_name: User datatable name. If set, also log to this
      datatable.
    time_delta: Time (in seconds) between logging events.

  Returns:
    A logger object that responds to logger.write(some_dict).
  """
  del user_datatable_name
  terminal_logger = TerminalLogger(label=label, time_delta=time_delta)
  return AutoCloseLogger(terminal_logger)


make_logger = make_logger_oss
