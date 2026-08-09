# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Bit-parity: jax vs torch compactified CCF residual."""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
torch = pytest.importorskip("torch")
torch.set_default_dtype(torch.float64)

import jax.numpy as jnp  # noqa: E402
from omnibias.pinn.jax.equations import ccf_compactified as jcc  # noqa: E402
from omnibias.pinn.torch.equations import ccf_compactified as tcc  # noqa: E402


def test_compactify_parity() -> None:
    y = np.linspace(-5.0, 5.0, 40)
    qj = np.asarray(jcc.compactify_y_rational(jnp.asarray(y)))
    qt = tcc.compactify_y_rational(torch.tensor(y)).numpy()
    np.testing.assert_allclose(qj, qt, atol=1e-15, rtol=0.0)
    lam = 0.6057
    qj2 = np.asarray(jcc.compactify_y_lambda(jnp.asarray(y), lam))
    qt2 = tcc.compactify_y_lambda(torch.tensor(y), lam).numpy()
    np.testing.assert_allclose(qj2, qt2, atol=1e-15, rtol=0.0)


def test_hardy_profile_parity_and_symbolic() -> None:
    from omnibias.symbolic.ccf import hardy_profile_numpy

    y = np.linspace(-4.0, 4.0, 80)
    coeffs = np.array([1.0, -0.4, 0.15])
    scales = np.array([0.7, 1.4, 2.2])
    lam = 0.6057
    alpha = 1.0 / (1.0 + lam)
    th_j, thp_j, h_j, hp_j = jcc.hardy_profile(
        jnp.asarray(y), jnp.asarray(coeffs), jnp.asarray(scales), alpha
    )
    th_t, thp_t, h_t, hp_t = tcc.hardy_profile(
        torch.tensor(y), torch.tensor(coeffs), torch.tensor(scales), alpha
    )
    th_n, thp_n, h_n, hp_n = hardy_profile_numpy(y, coeffs, scales, alpha)
    for a, b, c in (
        (th_j, th_t, th_n),
        (thp_j, thp_t, thp_n),
        (h_j, h_t, h_n),
        (hp_j, hp_t, hp_n),
    ):
        np.testing.assert_allclose(np.asarray(a), b.detach().numpy(), atol=1e-14)
        np.testing.assert_allclose(np.asarray(a), c, atol=1e-14)
    eq, factored, _, fields = jcc.ccf_hardy_residual_samples(
        jnp.asarray(y), jnp.asarray(coeffs), jnp.asarray(scales), lam
    )
    assert np.isfinite(float(jnp.max(jnp.abs(eq))))
    assert fields["theta"].shape == y.shape
    _ = factored


def test_ccf_compactified_residual_parity() -> None:
    _, y = jcc.compactified_grid(64, q_max=0.97)
    y_np = np.asarray(y)
    theta = np.exp(-0.3 * y_np * y_np)
    theta_y = -0.6 * y_np * theta
    lam = 0.5
    eq_j, r_j, w_j = jcc.ccf_compactified_residual_samples(
        jnp.asarray(y_np), jnp.asarray(theta), jnp.asarray(theta_y), lam
    )
    eq_t, r_t, w_t = tcc.ccf_compactified_residual_samples(
        torch.tensor(y_np), torch.tensor(theta), torch.tensor(theta_y), lam
    )
    np.testing.assert_allclose(np.asarray(w_j), w_t.numpy(), atol=1e-14)
    np.testing.assert_allclose(np.asarray(eq_j), eq_t.numpy(), atol=1e-10)
    np.testing.assert_allclose(np.asarray(r_j), r_t.numpy(), atol=1e-10)
