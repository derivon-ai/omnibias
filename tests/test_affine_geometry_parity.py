# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Torch/JAX parity for affine geometry values and first/second derivatives."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from omnibias.jax.affine_geometry import affine_project as jax_project
from omnibias.jax.affine_geometry import linearized_equality_step as jax_step
from omnibias.torch.affine_geometry import affine_project as torch_project
from omnibias.torch.affine_geometry import linearized_equality_step as torch_step

jax.config.update("jax_enable_x64", True)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_values_gradients_hessians_and_nonlinear_iteration_parity(dtype):
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        packed = np.array([.4, -.7, 1.2, 1.1, 2.3, .4, .2, -.5, 1.3, .1, -.2], dtype=dtype)
        t = torch.tensor(packed, requires_grad=True)
        j = jnp.array(packed)
        def tf(z):
            result = torch_project(z[:3], z[3:9].reshape(2, 3), z[9:])
            return (result.projected * z.new_tensor([.7, -.3, .2])).sum() + .3 * result.half_squared_distance
        def jf(z):
            result = jax_project(z[:3], z[3:9].reshape(2, 3), z[9:])
            return result.projected @ jnp.array([.7, -.3, .2], dtype=z.dtype) + .3 * result.half_squared_distance
        tr = torch_project(t[:3], t[3:9].reshape(2, 3), t[9:])
        jr = jax.jit(jax_project)(j[:3], j[3:9].reshape(2, 3), j[9:])
        tolerance = 2e-5 if dtype == np.float32 else 3e-12
        for tv, jv in zip(tr, jr, strict=True):
            np.testing.assert_allclose(tv.detach(), jv, rtol=tolerance, atol=tolerance)
        np.testing.assert_allclose(torch.autograd.functional.jacobian(tf, t).detach(), jax.jit(jax.grad(jf))(j),
                                   rtol=tolerance, atol=tolerance)
        np.testing.assert_allclose(torch.autograd.functional.hessian(tf, t).detach(), jax.jit(jax.hessian(jf))(j),
                                   rtol=tolerance, atol=tolerance)
        tx, jx = t[:3], j[:3]
        ts = torch_step(tx, (tx.square().sum() - 1)[None], (2*tx)[None])
        js = jax.jit(jax_step)(jx, (jnp.sum(jx**2) - 1)[None], (2*jx)[None])
        for tv, jv in zip(ts, js, strict=True):
            np.testing.assert_allclose(tv.detach(), jv, rtol=tolerance, atol=tolerance)
    finally:
        torch.set_num_threads(old_threads)
