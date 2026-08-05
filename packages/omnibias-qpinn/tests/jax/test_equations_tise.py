# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Unit tests for the TISE residual (jax backend)."""

from __future__ import annotations

import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from omnibias.qpinn.jax.equations import TISE, TISEOutput, tise


class TestTISEShape:
    def test_returns_named_tuple(self, psi_field_x, coords_x):
        state = psi_field_x(coords_x)
        out = TISE(energy=0.5)(state)
        assert isinstance(out, TISEOutput)
        assert out.residual.shape == (coords_x.shape[0], 2)
        assert out.energy_estimate is None
        assert "mean_sq_residual" in out.diag

    def test_function_form_matches_class(self, psi_field_x, coords_x):
        state = psi_field_x(coords_x)
        out_cls = TISE(energy=0.5, hbar=1.0, mass=1.0)(state)
        out_fn = tise(state, energy=0.5, hbar=1.0, mass=1.0)
        assert jnp.allclose(out_cls.residual, out_fn.residual)


class TestTISEEnergyEstimate:
    def test_with_quadrature_weights(self, psi_field_x, coords_x):
        state = psi_field_x(coords_x)
        B = coords_x.shape[0]
        weights = jnp.full((B,), 4.0 / B, dtype=jnp.float64)
        out = TISE(
            energy=0.0,
            potential=lambda s: 0.5 * s.coords[..., 0] ** 2,
            quadrature_weights=weights,
        )(state)
        assert out.energy_estimate is not None
        assert out.energy_estimate.ndim == 0
        assert "energy_estimate" in out.diag
        assert "norm_squared" in out.diag
