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

import enum
import getpass
from acme.utils import loggers


_READERS = ('all-users', 'all-groups')
MAX_FLUSH_DELAY_SECONDS = 10
_CAPACITY = 5


def make_logger_acme(label: str,
                     user_datatable_name: str = '',
                     time_delta: float = 0.2,) -> loggers.Logger:
  """Make an Acme logger for Iris.

  Args:
    label: Name to give to the logger.
    user_datatable_name: User datatable name. If set, also log to this
      datatable.
    time_delta: Time (in seconds) between logging events.

  Returns:
    A logger object that responds to logger.write(some_dict).
  """
  additional_loggers = []
  return loggers.make_default_logger(
      label=label,
      additional_loggers=additional_loggers,
      time_delta=time_delta,
  )


def make_logger_oss(
    label: str,
    user_datatable_name: str = '',
    time_delta: float = 0.2,
) -> xdata.bt.WriterInterface | loggers.Logger:
  """Make an Acme or XData logger for BBV2.

  Args:
    label: Name to give to the logger.
    user_datatable_name: User datatable name. If set, also log to this
      datatable.
    time_delta: Time (in seconds) between logging events.

  Returns:
    A logger object that responds to logger.write(some_dict).
  """
  return make_logger_acme(label=label,
                          user_datatable_name=user_datatable_name,
                          time_delta=time_delta)

make_logger = make_logger_oss
