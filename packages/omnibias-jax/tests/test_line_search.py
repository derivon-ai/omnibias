# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""JAX jet line search (theory 03-12): G1 exactness and never-worse."""

from __future__ import annotations

import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
from omnibias.core.line_search import JetLineSearchConfig  # noqa: E402
from omnibias.jax.line_search import directional_derivatives, jet_line_search  # noqa: E402


def _quartic_loss(params: jax.Array) -> jax.Array:
    s = params[0]
    return 1.0 - 2.0 * s + 3.0 * s**2 - 2.0 * s**3 + 2.0 * s**4


def test_g1_jet_matches_closed_form_orders_0_to_6() -> None:
    def loss(p: jax.Array) -> jax.Array:
        return jnp.exp(jnp.sum(p))

    params = jnp.zeros(3)
    direction = jnp.ones(3)
    derivs = directional_derivatives(loss, params, direction, 6)
    for k, val in enumerate(derivs):
        expected = 3.0**k
        rel = abs(val - expected) / max(abs(expected), 1.0)
        assert rel <= 1e-10, f"order {k}: {val} vs {expected} rel={rel}"


def test_quartic_recovers_spec_step() -> None:
    params = jnp.zeros(2)
    direction = jnp.array([1.0, 0.0])
    result = jet_line_search(
        _quartic_loss,
        params,
        direction,
        config=JetLineSearchConfig(order=4, trust_radius=1.0, verify=True),
        next_derivative_bound=0.0,
    )
    assert result.step == pytest.approx(0.409461, abs=5e-5)
    assert result.actual_value is not None
    assert result.actual_value <= 1.0 + 1e-14
    assert result.truncation_certified
    assert not result.fell_back


def test_g3_never_worse_pathological() -> None:
    def sharp_bowl(p: jax.Array) -> jax.Array:
        s = p[0]
        return 1.0 - 2.0 * s + 50.0 * s**2

    result = jet_line_search(
        sharp_bowl,
        jnp.zeros(1),
        jnp.ones(1),
        config=JetLineSearchConfig(order=1, trust_radius=1.0, verify=True),
    )
    assert result.actual_value is not None
    assert result.actual_value <= 1.0 + 1e-14
    assert result.fell_back
