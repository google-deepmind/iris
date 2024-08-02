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

"""Loads all controllers."""
from iris.algorithms.controllers import hill_climb_controller
from iris.algorithms.controllers import neat_controller
from iris.algorithms.controllers import policy_gradient_controller
from iris.algorithms.controllers import random_controller
from iris.algorithms.controllers import regularized_evolution_controller

CONTROLLER_DICT = {
    "hill_climb":
        hill_climb_controller.HillClimbController,
    "neat":
        neat_controller.NEATController,
    "policy_gradient":
        policy_gradient_controller.PolicyGradientController,
    "random_search":
        random_controller.RandomController,
    "regularized_evolution":
        regularized_evolution_controller.RegularizedEvolutionController
}
