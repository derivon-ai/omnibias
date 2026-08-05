# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Unit tests for the diagnostic helpers (jax backend)."""

from __future__ import annotations

import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

jax.config.update("jax_enable_x64", True)

from omnibias.pinn._core.coords import CoordinateSpec
from omnibias.pinn.jax.fields.one_layer import make_one_layer_vector_field
from omnibias.qpinn import make_psi_components
from omnibias.qpinn.jax.diagnostics import (
    continuity_residual,
    current_divergence,
    energy_variance,
    expectation_value,
    expected_energy,
    norm_drift,
    norm_squared,
    probability_current,
)


@pytest.fixture
def psi_field_1d():
    coord = CoordinateSpec(axes=("x",))
    spec = make_psi_components(name="psi")
    return make_one_layer_vector_field(
        coordinate_spec=coord, components=spec, hidden=8,
        base="gaussian", dtype=jnp.float64, seed=0,
    )


@pytest.fixture
def psi_field_xt_local():
    coord = CoordinateSpec(axes=("x", "t"))
    spec = make_psi_components(name="psi")
    return make_one_layer_vector_field(
        coordinate_spec=coord, components=spec, hidden=8,
        base="gaussian", dtype=jnp.float64, seed=0,
    )


class TestNormDiagnostics:
    def test_norm_squared_finite(self, psi_field_1d):
        coords = jnp.linspace(-2.0, 2.0, 17, dtype=jnp.float64)[:, None]
        state = psi_field_1d(coords)
        n = norm_squared(state)
        assert jnp.isfinite(n) and float(n) > 0

    def test_norm_drift_nonnegative(self, psi_field_1d):
        coords = jnp.linspace(-2.0, 2.0, 17, dtype=jnp.float64)[:, None]
        state = psi_field_1d(coords)
        d = norm_drift(state, target_norm=1.0)
        assert float(d) >= 0


class TestEnergyDiagnostics:
    def test_expected_energy_finite(self, psi_field_1d):
        coords = jnp.linspace(-2.0, 2.0, 31, dtype=jnp.float64)[:, None]
        state = psi_field_1d(coords)
        E = expected_energy(
            state,
            potential=lambda s: 0.5 * s.coords[..., 0] ** 2,
        )
        assert jnp.isfinite(E)

    def test_expectation_value_decomposition(self, psi_field_1d):
        coords = jnp.linspace(-2.0, 2.0, 31, dtype=jnp.float64)[:, None]
        state = psi_field_1d(coords)
        from omnibias.qpinn._core.complex import apply_kinetic
        E_T = expectation_value(
            state, operator_action=lambda s: apply_kinetic(s, hbar=1.0, mass=1.0),
        )
        E_V = expectation_value(
            state,
            operator_action=lambda s: (
                0.5 * s.coords[..., 0] ** 2 * s.ops.value(s, "psi_re"),
                0.5 * s.coords[..., 0] ** 2 * s.ops.value(s, "psi_im"),
            ),
        )
        E_full = expected_energy(
            state, potential=lambda s: 0.5 * s.coords[..., 0] ** 2,
        )
        assert jnp.allclose(E_T + E_V, E_full)

    def test_variance_nonnegative_within_tolerance(self, psi_field_1d):
        coords = jnp.linspace(-2.0, 2.0, 31, dtype=jnp.float64)[:, None]
        state = psi_field_1d(coords)
        var = energy_variance(
            state, potential=lambda s: 0.5 * s.coords[..., 0] ** 2,
        )
        assert float(var) >= -1e-10


class TestCurrentDiagnostics:
    def test_probability_current_shape(self, psi_field_xt_local):
        coords = jax.random.normal(jax.random.PRNGKey(1), (16, 2), dtype=jnp.float64)
        state = psi_field_xt_local(coords)
        j = probability_current(state)
        assert j.shape == (16, 1)

    def test_current_divergence_shape(self, psi_field_xt_local):
        coords = jax.random.normal(jax.random.PRNGKey(1), (16, 2), dtype=jnp.float64)
        state = psi_field_xt_local(coords)
        d = current_divergence(state)
        assert d.shape == (16,)

    def test_continuity_residual_requires_time(self, psi_field_1d):
        coords = jnp.linspace(-1.0, 1.0, 5, dtype=jnp.float64)[:, None]
        state = psi_field_1d(coords)
        with pytest.raises(ValueError, match="time axis"):
            continuity_residual(state)

    def test_continuity_residual_finite(self, psi_field_xt_local):
        coords = jax.random.normal(jax.random.PRNGKey(1), (16, 2), dtype=jnp.float64)
        state = psi_field_xt_local(coords)
        r = continuity_residual(state)
        assert r.shape == (16,)
        assert jnp.all(jnp.isfinite(r))

    def test_zero_mass_rejected(self, psi_field_xt_local):
        coords = jax.random.normal(jax.random.PRNGKey(1), (8, 2), dtype=jnp.float64)
        state = psi_field_xt_local(coords)
        with pytest.raises(ValueError, match="mass"):
            probability_current(state, mass=0.0)
