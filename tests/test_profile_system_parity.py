# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Torch/JAX companion-profile evaluation and live-parameter parity."""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import torch
from omnibias.core.profile_system import ProfilePack
from omnibias.jax.profile_system import profile_solution as jprofile
from omnibias.torch.profile_system import profile_solution as tprofile


@pytest.mark.parametrize("annihilator,initial", [((-1, 1), [0.7]), ((1, 0, 1), [0.2, 0.7]),
                                               ((1, -2, 1), [0.2, 0.7]),
                                               ((1, 0, 2, 0, 1), [0.0, 0.0, 0.0, 1.0])])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_values_and_parameter_gradients(annihilator, initial, dtype):
    order = len(initial)
    pack = ProfilePack((0, 1), (1, 0), annihilator,
                       tuple(tuple(int(i == j) for j in range(order)) for i in range(order)))
    xx = np.array([[0.2, -0.3], [-0.1, 0.4], [0.3, 0.6]], dtype=dtype)
    tx, jx = torch.tensor(xx), jnp.asarray(xx)
    ti = torch.tensor(initial, dtype=tx.dtype, requires_grad=True)
    tb = torch.tensor(0.13, dtype=tx.dtype, requires_grad=True)
    tv = tprofile(tx, pack, ti, tb)

    def forward(x, initial_jet, bias):
        return jprofile(x, pack, initial_jet, bias)

    ji, jb = jnp.asarray(initial, dtype=jx.dtype), jnp.asarray(0.13, dtype=jx.dtype)
    jv = jax.jit(forward)(jx, ji, jb)
    tolerance = 5e-5 if dtype is np.float32 else 3e-11
    np.testing.assert_allclose(tv.detach(), jv, rtol=tolerance, atol=tolerance)
    tg = torch.autograd.grad(tv.square().sum(), (ti, tb))
    jg = jax.jit(jax.grad(lambda i, b: jnp.sum(forward(jx, i, b)**2), argnums=(0, 1)))(ji, jb)
    for torch_grad, jax_grad in zip(tg, jg, strict=True):
        np.testing.assert_allclose(torch_grad.detach(), jax_grad, rtol=tolerance, atol=tolerance)
