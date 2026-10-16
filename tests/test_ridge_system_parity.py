# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Paired runtime derivatives, gradients and dtype behavior for ridge syzygies."""

from fractions import Fraction

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import torch
from omnibias.core.ridge_system import RidgeSyzygy
from omnibias.jax.ridge_system import ridge_solution as jridge
from omnibias.torch.ridge_system import ridge_solution as tridge


@pytest.mark.parametrize("activation", ["tanh", "sigmoid", "softplus", "gaussian", "exp",
                                         "sin", "cos", "sinh", "cosh"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_values_and_parameter_gradients(activation, dtype):
    # The scalar equation u_xxx + v = 0 has Q=(-8/27, s^3) at direction 3/2.
    pack = RidgeSyzygy((Fraction(3, 2),), ((Fraction(-8, 27), 0), (0, 0), (0, 0), (0, 1)))
    xx = np.linspace(-0.7, 0.6, 7, dtype=dtype).reshape(-1, 1)
    tx, jx = torch.tensor(xx), jnp.asarray(xx)
    tb = torch.tensor(0.13, dtype=tx.dtype, requires_grad=True)
    tw = torch.tensor(-0.7, dtype=tx.dtype, requires_grad=True)
    tv = tw * tridge(tx, pack, tb, activation=activation)

    def forward(x, bias, weight):
        return weight * jridge(x, pack, bias, activation=activation)

    jb, jw = jnp.asarray(0.13, dtype=jx.dtype), jnp.asarray(-0.7, dtype=jx.dtype)
    jv = jax.jit(forward)(jx, jb, jw)
    tolerance = 3e-5 if dtype is np.float32 else 2e-11
    np.testing.assert_allclose(tv.detach(), jv, rtol=tolerance, atol=tolerance)
    tg = torch.autograd.grad(tv.square().sum(), (tb, tw))
    jg = jax.jit(jax.grad(lambda b, w: jnp.sum(forward(jx, b, w)**2), argnums=(0, 1)))(jb, jw)
    for torch_grad, jax_grad in zip(tg, jg, strict=True):
        np.testing.assert_allclose(torch_grad.detach(), jax_grad, rtol=tolerance, atol=tolerance)
