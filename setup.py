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

"""Setup for pip package."""

import itertools
import setuptools


def _get_version():
  with open('iris/__init__.py') as fp:
    for line in fp:
      if line.startswith('__version__'):
        g = {}
        exec(line, g)  # pylint: disable=exec-used
        return g['__version__']
    raise ValueError('`__version__` not defined in `iris/__init__.py`')


def _strip_comments_from_line(s: str) -> str:
  """Parses a line of a requirements.txt file."""
  requirement, *_ = s.split('#')
  return requirement.strip()


def _parse_requirements(requirements_txt_path: str) -> list[str]:
  """Returns a list of dependencies for setup() from requirements.txt."""

  with open(requirements_txt_path) as fp:
    # Parse comments.
    lines = [_strip_comments_from_line(line) for line in fp.read().splitlines()]
    # Remove empty lines and direct github repos (not allowed in setup.py)
    return [l for l in lines if (l and 'github.com' not in l)]


extras_require = {
    'rl': _parse_requirements('requirements-rl.txt'),
    'extras': _parse_requirements('requirements-extras.txt'),
}

extras_require['all'] = list(
    itertools.chain.from_iterable(extras_require.values())
)

setuptools.setup(
    name='google-iris',
    version=_get_version(),
    description='Iris',
    author='Iris Team',
    author_email='jaindeepali@google.com',
    install_requires=_parse_requirements('requirements.txt'),
    packages=setuptools.find_packages(),
    extras_require=extras_require,
    python_requires='>=3.10',
)
