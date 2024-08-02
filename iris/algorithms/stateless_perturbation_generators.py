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

r"""Library for generating ensembles of perturbations for Blackbox training.

A library for generating ensembles of perturbations for Blackbox training.
The generator is stateless, e.g. does not need a state in the current point of
the optimization to calculatee the directions. Thus the library supports in
particular a rich class of algorithms generating structured (orthogonal or
quasi-orthogonal, QMC) ensembles.
"""

import abc
import math

import numpy as np
import scipy.stats as ss


class MatrixGenerator(metaclass=abc.ABCMeta):
  r"""Abstract class for generting matrices with rows encoding perturbations.

  Class is responsible for constructing matrices with rows encoding
  perturbations for the Blackbox training. The matrices are of the shape [m,d],
  where m stands for number of perturbations and d for perturbations'
  dimensionality.
  """

  @abc.abstractmethod
  def generate_matrix(self):
    r"""Returns the generated 2D matrix.

    Creates a 2D matrix.

    Args:

    Returns:
      Generated 2D matrix.
    """
    raise NotImplementedError('Abstract method')


class GaussianUnstructuredMatrixGenerator(MatrixGenerator):
  r"""Derives from MatrixGenerator and creates a Gaussian matrix.

  Class responsible for constructing unstructured Gaussian matrix with entries
  taken independently at random from N(0,1).
  """

  def __init__(self, num_suggestions, dim):
    self.num_suggestions = num_suggestions
    self.dim = dim
    super().__init__()

  def generate_matrix(self):
    return np.random.normal(size=(self.num_suggestions, self.dim))


class GaussianOrthogonalMatrixGenerator(MatrixGenerator):
  r"""Derives from MatrixGenerator and creates Gaussian orthogonal matrix.

  Class responsible for constructing block-orthogonal Gaussian matrix with:
  different blocks constructed independently, orthogonal rows within a fixed
  d x d block and marginal distributions of rows N(0,I_d).
  """

  def __init__(self, num_suggestions, dim, deterministic_lengths):
    self.num_suggestions = num_suggestions
    self.dim = dim
    self.deterministic_lengths = deterministic_lengths
    super().__init__()

  def generate_matrix(self):
    nb_full_blocks = int(self.num_suggestions / self.dim)
    block_list = []
    for _ in range(nb_full_blocks):
      unstructured_block = np.random.normal(size=(self.dim, self.dim))
      q, _ = np.linalg.qr(unstructured_block)
      q = np.transpose(q)
      block_list.append(q)
    remaining_rows = self.num_suggestions - nb_full_blocks * self.dim
    if remaining_rows > 0:
      unstructured_block = np.random.normal(size=(self.dim, self.dim))
      q, _ = np.linalg.qr(unstructured_block)
      q = np.transpose(q)
      block_list.append(q[0:remaining_rows])
    final_matrix = np.vstack(block_list)

    if not self.deterministic_lengths:
      multiplier = np.linalg.norm(
          np.random.normal(size=(self.num_suggestions, self.dim)), axis=1)
    else:
      multiplier = np.sqrt(float(self.dim)) * np.ones((self.num_suggestions))

    return np.matmul(np.diag(multiplier), final_matrix)


class SignMatrixGenerator(MatrixGenerator):
  r"""Derives from MatrixGenerator and creates Sign matrix.

  Class responsible for constructing a matrix with random entries from {-1,+1}.
  """

  def __init__(self, num_suggestions, dim):
    self.num_suggestions = num_suggestions
    self.dim = dim
    super().__init__()

  def generate_matrix(self):
    return np.sign(np.random.normal(size=(self.num_suggestions, self.dim)))


class SphereMatrixGenerator(MatrixGenerator):
  r"""Derives from MatrixGenerator and creates normalized Gaussian matrix.

  Class responsible for constructing a normalized Gaussian matrix with rows of
  length sqrt{d}.
  """

  def __init__(self, num_suggestions, dim):
    self.num_suggestions = num_suggestions
    self.dim = dim
    super().__init__()

  def generate_matrix(self):
    gaussian_unnormalized = np.random.normal(
        size=(self.num_suggestions, self.dim))
    lengths = np.linalg.norm(gaussian_unnormalized, axis=1, keepdims=True)
    return np.sqrt(self.dim) * (gaussian_unnormalized / lengths)


class RandomHadamardMatrixGenerator(MatrixGenerator):
  r"""Derives from MatrixGenerator and creates random Hadamard matrix.

  Class responsible for constructing a random Hadamard matrix HD, where
  H is a Kronecker-product Hadamard and D is a random diagonal matrix with
  entries on the diagonal taken independently and uniformly at random from the
  two-element discrete set {-1,+1}.
  Since H is a Kronecker-product Hadamard matrix, it is assumed that dim is
  a power of two (if necessary, by padding extra zeros).
  """

  def __init__(self, num_suggestions, dim):
    self.num_suggestions = num_suggestions
    self.dim = dim
    full_matrix_size = 1
    while full_matrix_size < dim:
      full_matrix_size = 2 * full_matrix_size
    nph = np.tile(1.0, (full_matrix_size, full_matrix_size))
    i = 1
    while i < full_matrix_size:
      for j in range(i):
        for k in range(i):
          nph[j + i][k] = nph[j][k]
          nph[j][k + i] = nph[j][k]
          nph[j + i][k + i] = -nph[j][k]
      i += i

    self.core_hadamard = nph
    self.extended_dim = full_matrix_size
    super().__init__()

  def generate_matrix(self):
    ones = np.ones((self.extended_dim))
    minus_ones = np.negative(ones)
    diagonal = np.where(
        np.random.uniform(size=(self.extended_dim)) < 0.5, ones, minus_ones)
    final_list = []
    for i in range(min(self.num_suggestions, self.extended_dim)):
      pointwise_product = np.multiply(self.core_hadamard[i], diagonal)
      final_list.append(pointwise_product)
    return np.array(final_list)


def create_rect_kac_matrix(num_rows, dim, number_of_blocks, angles,
                           indices_pairs):
  r"""Creates a rectangular Kac's random walk matrix for given rotations.

  Outputs a submatrix of the  Kac's random walk matrix truncated to its first
  <num_rows> rows for a given list of angles and indices defining
  low-dimensional rotations. The Kac's random walk matrix is a product of
  <number_of_blocks> Givens rotations. Each Givens rotation is characterized by
  its angle and a pair of indices characterizing 2-dimensional space spanned by
  two canonical vectors, where the rotation occurs.

  Args:
    num_rows: number of first rows output
    dim: number of rows/columns of the full Kac's random walk matrix
    number_of_blocks: number of Givens random rotations used to create full
      Kac's random walk matrix
    angles: list of angles used to construct Givens random rotations
    indices_pairs: list of pairs of indices used to construct Givens random
      rotations

  Returns:
    Kac's random walk matrix.
  """
  matrix_as_list = []
  for index in range(min(num_rows, dim)):
    base_vector = np.zeros(dim)
    np.put(base_vector, index, 1.0)
    for j in range(number_of_blocks):
      angle = angles[j]
      p = indices_pairs[j][0]
      q = indices_pairs[j][1]
      if p > q:
        u = p
        p = q
        q = u
      base_vector[p] = math.cos(angle) * base_vector[p] - math.sin(
          angle) * base_vector[q]
      base_vector[q] = math.sin(angle) * base_vector[p] + math.cos(
          angle) * base_vector[q]
    matrix_as_list.append(base_vector)
  return math.sqrt(float(dim)) * np.array(matrix_as_list)


class KacMatrixGenerator(MatrixGenerator):
  r"""Derives from MatrixGenerator and creates Kac's random walk matrix.

  Class responsible for constructing a submatrix of the Kac's random walk
  matrix obtained from the full Kac's random walk matrix by truncating it to its
  first x rows.
  """

  def __init__(self, num_suggestions, dim, number_of_blocks):
    r"""Constructor of the Kac's random walk matrix generator.

    Args:
      num_suggestions: number of the rows of the constructed matrix
      dim: the number of the columns of the constructed matrix
      number_of_blocks: number of blocks of the applied Kac's random walk matrix
    """
    self.num_suggestions = num_suggestions
    self.dim = dim
    self.number_of_blocks = number_of_blocks
    super().__init__()

  def generate_matrix(self):
    angles = np.random.uniform(
        low=0.0, high=2.0 * np.pi, size=(self.number_of_blocks))
    indices_pairs = np.random.choice(
        np.arange(self.dim), size=(self.number_of_blocks, 2))
    return create_rect_kac_matrix(self.num_suggestions, self.dim,
                                  self.number_of_blocks, angles, indices_pairs)


def phi_reflection(b, x):
  r"""Outputs \phi function used in the computation of Halton sequences.

  Outputs a function \phi_{b} defined as follows:

    \phi_{b}(y_{k}y_{k-1}...y_{0}_{b}) = y_{0}/b + y_{1}/b^{2} + ...,

  where y_{k}y_{k-1}...y_{0}_{b} stands for the representation of the number
  using base  b.

  Args:
    b: base used by the \phi function
    x: input to the \phi function

  Returns:
    \phi_{b}(x)
  """
  b_f = float(b)
  x_f = float(x)
  coefficients = []
  while x_f >= b_f:
    w = math.floor(x_f / b_f) * b_f
    r = x_f - w
    coefficients.append(r)
    x_f = math.floor(x_f / b_f)
  coefficients.append(x_f)
  x_f_reflection = 0.0
  power_of_b = b_f
  for i in range(len(coefficients)):
    x_f_reflection += coefficients[i] / power_of_b
    power_of_b *= b_f
  return x_f_reflection


def create_rect_hal_matrix(b_set, r_array):
  r"""Creates a Halton matrix.

  Creates a Halton matrix using inputs from the list <r_array> and base values
  for \phi function parametrization from the list <b_set>.

  Args:
    b_set: set of base values
    r_array: inputs to the \phi function

  Returns:
    corresponding Halton matrix
  """
  rows = []
  for i in range(len(b_set)):
    next_row = []
    for j in range(len(r_array)):
      next_row.append(ss.norm.ppf(phi_reflection(b_set[i], r_array[j])))
    rows.append(next_row)
  return np.array(rows)


class HaltonMatrixGenerator(MatrixGenerator):
  r"""Derives from MatrixGenerator and creates Halton matrix.

  Class responsible for constructing a subsampled version of the Halton matrix H
  with rows defined as follows:

  h_{j} = (\phi_{b_{1}}(j),..., \phi_{b_{dim}}(j))^{T}, where

  b_{1},...,b_{dim} is a set of numbers such that gcd(b_{i},b_{k}) = 1 for
  i != k and \phi_{y} is defined as follows:

  \phi_{y}(y_{0} + y_{1}*y + y_{2}*y^{2} + ...) = y_{0}/y + y_{1}/y^{2} + ...

  for y_{0},y_{1},...., < y.
  """

  def __init__(self, num_suggestions, dim, b_set):
    r"""Constructor of the Halton matrix generator.

    Args:
      num_suggestions: number of the rows of the constructed matrix
      dim: the number of the columns of the constructed matrix
      b_set: the list of bases: [b_{1},...,b_{dim}] (see: explanation above)
    """
    self.num_suggestions = num_suggestions
    self.dim = dim
    self.b_set = b_set
    super().__init__()

  def generate_matrix(self):
    return np.transpose(
        create_rect_hal_matrix(
            np.array(self.b_set),
            np.arange(min(self.num_suggestions + 1, self.dim))))[1:]
