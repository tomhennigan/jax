# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import itertools
import operator
import textwrap

import scipy.ndimage

from ..numpy import lax_numpy as np
from ..numpy.lax_numpy import _wraps
from ..util import safe_zip as zip


_nonempty_prod = functools.partial(functools.reduce, operator.mul)
_nonempty_sum = functools.partial(functools.reduce, operator.add)

_INDEX_FIXERS = {
    'constant': lambda index, size: index,
    'nearest': lambda index, size: np.clip(index, 0, size - 1),
    'wrap': lambda index, size: index % size,
}


def _nearest_indices_and_weights(coordinate):
  index = np.around(coordinate).astype(np.int32)
  weight = coordinate.dtype.type(1)
  return [(index, weight)]


def _linear_indices_and_weights(coordinate):
  lower = np.floor(coordinate)
  upper = np.ceil(coordinate)
  l_index = lower.astype(np.int32)
  u_index = upper.astype(np.int32)
  one = coordinate.dtype.type(1)
  l_weight = one - (coordinate - lower)
  u_weight = one - l_weight  # handles the edge case lower==upper
  return [(l_index, l_weight), (u_index, u_weight)]


@_wraps(scipy.ndimage.map_coordinates, lax_description=textwrap.dedent("""\
    Only linear interpolation (``order=1``) and modes ``'constant'``,
    ``'nearest'`` and ``'wrap'`` are currently supported. Note that
    interpolation near boundaries differs from the scipy function, because we
    fixed an outstanding bug (https://github.com/scipy/scipy/issues/2640);
    this function interprets the ``mode`` argument as documented by SciPy, but
    not as implemented by SciPy.
    """))
def map_coordinates(
    input, coordinates, order, mode='constant', cval=0.0,
):
  input = np.asarray(input)
  coordinates = [np.asarray(c, input.dtype) for c in coordinates]
  cval = np.asarray(cval, input.dtype)

  if len(coordinates) != input.ndim:
    raise ValueError('coordinates must be a sequence of length input.ndim, but '
                     '{} != {}'.format(len(coordinates), input.ndim))

  index_fixer = _INDEX_FIXERS.get(mode)
  if index_fixer is None:
    raise NotImplementedError(
        'jax.scipy.ndimage.map_coordinates does not yet support mode {}. '
        'Currently supported modes are {}.'.format(mode, set(_INDEX_FIXERS)))

  if mode == 'constant':
    is_valid = lambda index, size: (0 <= index) & (index < size)
  else:
    is_valid = lambda index, size: True

  if order == 0:
    interp_fun = _nearest_indices_and_weights
  elif order == 1:
    interp_fun = _linear_indices_and_weights
  else:
    raise NotImplementedError(
        'jax.scipy.ndimage.map_coordinates currently requires order<=1')

  valid_1d_interpolations = []
  for coordinate, size in zip(coordinates, input.shape):
    interp_nodes = interp_fun(coordinate)
    valid_interp = []
    for index, weight in interp_nodes:
      fixed_index = index_fixer(index, size)
      valid = is_valid(index, size)
      valid_interp.append((fixed_index, valid, weight))
    valid_1d_interpolations.append(valid_interp)

  outputs = []
  for items in itertools.product(*valid_1d_interpolations):
    indices, validities, weights = zip(*items)
    if any(valid is not True for valid in validities):
      all_valid = functools.reduce(operator.and_, validities)
      contribution = np.where(all_valid, input[indices], cval)
    else:
      contribution = input[indices]
    outputs.append(_nonempty_prod(weights) * contribution)
  result = _nonempty_sum(outputs)
  return result
