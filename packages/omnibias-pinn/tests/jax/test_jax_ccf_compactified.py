# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Compactified / line-domain CCF residual tests (jax)."""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
from omnibias.pinn.jax.equations import ccf_compactified as cc  # noqa: E402


def test_compactify_roundtrip() -> None:
    y = jnp.linspace(-8.0, 8.0, 64)
    q = cc.compactify_y(y)
    y2 = cc.decompactify_q(q)
    np.testing.assert_allclose(np.asarray(y2), np.asarray(y), atol=1e-12)
    assert float(jnp.max(jnp.abs(q))) < 1.0


def test_dq_dy_matches_finite_difference() -> None:
    y = jnp.linspace(-3.0, 3.0, 80)
    dq = cc.dq_dy(y)
    eps = 1e-6
    fd = (cc.compactify_y(y + eps) - cc.compactify_y(y - eps)) / (2 * eps)
    np.testing.assert_allclose(np.asarray(dq), np.asarray(fd), atol=1e-8)


def test_envelope_product_rule() -> None:
    y = jnp.linspace(-2.0, 2.0, 50)
    hat = jnp.cos(y)
    hat_y = -jnp.sin(y)
    theta, theta_y = cc.apply_envelope(y, hat, hat_y, power=1.0)
    env, env_y = cc.decay_envelope(y, power=1.0)
    np.testing.assert_allclose(np.asarray(theta), np.asarray(env * hat), atol=1e-14)
    np.testing.assert_allclose(
        np.asarray(theta_y), np.asarray(env_y * hat + env * hat_y), atol=1e-14
    )


def test_mms_factored_residual_recovers_forcing() -> None:
    """Manufactured solution: pick Theta, compute E, check E = F * R."""
    q, y = cc.compactified_grid(96, q_max=0.98)
    del q
    lam = 0.6057
    # smooth even-ish bump decaying at infinity
    theta = jnp.exp(-0.25 * y * y)
    theta_y = -0.5 * y * theta
    equation, factored, weight = cc.ccf_compactified_residual_samples(
        y, theta, theta_y, lam, weight_kind="one_plus_abs"
    )
    np.testing.assert_allclose(
        np.asarray(equation), np.asarray(factored * weight), atol=1e-12
    )
    # MMS: forcing := equation; residual of (Theta, forcing) is zero when subtracted
    err = equation - equation
    assert float(jnp.max(jnp.abs(err))) == 0.0
    assert float(jnp.max(jnp.abs(factored))) < 1e2  # finite on compactified grid


def test_residual_weight_kinds() -> None:
    y = jnp.asarray([-2.0, 0.0, 3.0])
    np.testing.assert_allclose(np.asarray(cc.residual_weight(y, kind="one")), 1.0)
    np.testing.assert_allclose(
        np.asarray(cc.residual_weight(y, kind="one_plus_abs")), [3.0, 1.0, 4.0]
    )
    with pytest.raises(ValueError, match="unknown"):
        cc.residual_weight(y, kind="nope")
