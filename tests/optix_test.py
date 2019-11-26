# Copyright 2018 Google LLC
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

"""Tests for the optix module."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
from jax import numpy as jnp
from jax.experimental import optimizers
from jax.experimental import optix
import jax.test_util  # imported only for flags
from jax.tree_util import tree_leaves
import numpy as onp

from jax.config import config
config.parse_flags_with_absl()


STEPS = 50
LR = 1e-2


class OptixTest(absltest.TestCase):

  def setUp(self):
    super(OptixTest, self).setUp()
    self.init_params = (jnp.array([1., 2.]), jnp.array([3., 4.]))
    self.per_step_updates = (jnp.array([500., 5.]), jnp.array([300., 3.]))

  def test_sgd(self):

    # experimental/optimizers.py
    jax_params = self.init_params
    opt_init, opt_update, get_params = optimizers.sgd(LR)
    state = opt_init(jax_params)
    for i in range(STEPS):
      state = opt_update(i, self.per_step_updates, state)
      jax_params = get_params(state)

    # experimental/optix.py
    optix_params = self.init_params
    opt_init, opt_update = optix.sgd(LR, 0.0)
    state = opt_init(optix_params)
    for _ in range(STEPS):
      updates, state = opt_update(self.per_step_updates, state)
      optix_params = optix.apply_updates(optix_params, updates)

    # Check equivalence.
    for x, y in zip(tree_leaves(jax_params), tree_leaves(optix_params)):
      onp.testing.assert_allclose(x, y, rtol=1e-5)

  def test_adam(self):
    b1, b2, eps = 0.9, 0.999, 1e-8

    # experimental/optimizers.py
    jax_params = self.init_params
    opt_init, opt_update, get_params = optimizers.adam(LR, b1, b2, eps)
    state = opt_init(jax_params)
    for i in range(STEPS):
      state = opt_update(i, self.per_step_updates, state)
      jax_params = get_params(state)

    # experimental/optix.py
    optix_params = self.init_params
    opt_init, opt_update = optix.adam(LR, b1, b2, eps)
    state = opt_init(optix_params)
    for _ in range(STEPS):
      updates, state = opt_update(self.per_step_updates, state)
      optix_params = optix.apply_updates(optix_params, updates)

    # Check equivalence.
    for x, y in zip(tree_leaves(jax_params), tree_leaves(optix_params)):
      onp.testing.assert_allclose(x, y, rtol=1e-4)

  def test_rmsprop(self):
    decay, eps = .9, 0.1

    # experimental/optimizers.py
    jax_params = self.init_params
    opt_init, opt_update, get_params = optimizers.rmsprop(LR, decay, eps)
    state = opt_init(jax_params)
    for i in range(STEPS):
      state = opt_update(i, self.per_step_updates, state)
      jax_params = get_params(state)

    # experimental/optix.py
    optix_params = self.init_params
    opt_init, opt_update = optix.rmsprop(LR, decay, eps)
    state = opt_init(optix_params)
    for _ in range(STEPS):
      updates, state = opt_update(self.per_step_updates, state)
      optix_params = optix.apply_updates(optix_params, updates)

    # Check equivalence.
    for x, y in zip(tree_leaves(jax_params), tree_leaves(optix_params)):
      onp.testing.assert_allclose(x, y, rtol=1e-5)


if __name__ == '__main__':
  absltest.main()
