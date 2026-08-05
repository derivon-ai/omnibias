# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""torch <-> jax parity of the finite-difference stencil twins.

Both twins are materialised from the *same* pure-Python stencil numbers, so they
agree up to the base activation's per-backend libm rounding. For the numerical
``finite_difference_tower`` that per-element floor is amplified by the sign
magnitude ``~1/delta^order``, so the tolerance is calibrated to it rather than a
single flat constant.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import omnibias.difference.jax as DJ  # noqa: E402
import omnibias.difference.torch as DT  # noqa: E402
from omnibias.difference import stencil_signs  # noqa: E402

NAMES = ["sigmoid", "tanh", "gaussian", "softplus", "silu", "gelu"]
_XS = np.linspace(-1.6, 1.6, 9)


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("order", [0, 1, 2, 3])
def test_collapse_to_derivative_parity(name: str, order: int) -> None:
    zt = torch.tensor(_XS, dtype=torch.float64)
    zj = jnp.asarray(_XS)
    ct = DT.collapse_to_derivative(name, zt, order).detach().numpy()
    cj = np.asarray(DJ.collapse_to_derivative(name, zj, order))
    scale = 1.0 + float(np.max(np.abs(ct)))
    assert np.max(np.abs(ct - cj)) < 1e-8 * scale


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("order", [1, 2, 3])
@pytest.mark.parametrize("stencil", ["forward", "central"])
def test_finite_difference_tower_parity(name: str, order: int, stencil: str) -> None:
    delta = 1e-2
    zt = torch.tensor(_XS, dtype=torch.float64)
    zj = jnp.asarray(_XS)
    ft = DT.finite_difference_tower(name, zt, order, delta, stencil).detach().numpy()
    fj = np.asarray(DJ.finite_difference_tower(name, zj, order, delta, stencil))
    sign_mag = sum(abs(s) for s in stencil_signs(order, delta, stencil))
    tol = 1e-12 * sign_mag + 1e-12  # calibrated to the 1/delta^order amplification
    assert np.max(np.abs(ft - fj)) < tol


@pytest.mark.parametrize("name", NAMES)
def test_residual_parity(name: str) -> None:
    delta = 1e-2
    zt = torch.tensor(_XS, dtype=torch.float64)
    zj = jnp.asarray(_XS)
    rt = DT.finite_difference_residual(name, zt, 2, delta, "central").detach().numpy()
    rj = np.asarray(DJ.finite_difference_residual(name, zj, 2, delta, "central"))
    sign_mag = sum(abs(s) for s in stencil_signs(2, delta, "central"))
    assert np.max(np.abs(rt - rj)) < 1e-12 * sign_mag + 1e-12


def test_twins_are_jit_and_grad_safe() -> None:
    zj = jnp.asarray(_XS)
    # jit through the numerical tower
    val = jax.jit(lambda z: DJ.finite_difference_tower("sigmoid", z, 1, 1e-2).sum())(zj)
    assert np.isfinite(float(val))
    # grad through the closed-form collapse
    grad = jax.grad(lambda z: DJ.collapse_to_derivative("tanh", z, 1).sum())(zj)
    assert grad.shape == zj.shape
    # torch autograd through the stencil operator
    zt = torch.tensor(_XS, dtype=torch.float64, requires_grad=True)
    DT.finite_difference_tower("sigmoid", zt, 1, 1e-2).sum().backward()
    assert zt.grad is not None and torch.isfinite(zt.grad).all()
