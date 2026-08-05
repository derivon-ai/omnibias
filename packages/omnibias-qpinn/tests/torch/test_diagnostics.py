# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Unit tests for the diagnostic helpers (torch backend)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from omnibias.pinn._core.coords import CoordinateSpec
from omnibias.pinn.torch.fields.one_layer import OneLayerVectorField
from omnibias.qpinn import make_psi_components
from omnibias.qpinn.torch.diagnostics import (
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
    torch.manual_seed(0)
    coord = CoordinateSpec(axes=("x",))
    spec = make_psi_components(name="psi")
    return OneLayerVectorField(
        coordinate_spec=coord, components=spec, hidden=8,
        base="gaussian", dtype=torch.float64,
    )


@pytest.fixture
def psi_field_xt():
    torch.manual_seed(0)
    coord = CoordinateSpec(axes=("x", "t"))
    spec = make_psi_components(name="psi")
    return OneLayerVectorField(
        coordinate_spec=coord, components=spec, hidden=8,
        base="gaussian", dtype=torch.float64,
    )


class TestNormDiagnostics:
    def test_norm_squared_uniform(self, psi_field_1d):
        coords = torch.linspace(-2.0, 2.0, 17, dtype=torch.float64).unsqueeze(-1)
        state = psi_field_1d(coords)
        n = norm_squared(state)
        assert torch.isfinite(n) and float(n.detach()) > 0

    def test_norm_squared_with_weights(self, psi_field_1d):
        coords = torch.linspace(-2.0, 2.0, 17, dtype=torch.float64).unsqueeze(-1)
        state = psi_field_1d(coords)
        w = torch.full((17,), 4.0 / 17, dtype=torch.float64)
        n = norm_squared(state, quadrature_weights=w)
        assert torch.isfinite(n) and float(n.detach()) > 0

    def test_norm_drift_nonnegative(self, psi_field_1d):
        coords = torch.linspace(-2.0, 2.0, 17, dtype=torch.float64).unsqueeze(-1)
        state = psi_field_1d(coords)
        d = norm_drift(state, target_norm=1.0)
        assert float(d.detach()) >= 0


class TestEnergyDiagnostics:
    def test_expected_energy_finite(self, psi_field_1d):
        coords = torch.linspace(-2.0, 2.0, 31, dtype=torch.float64).unsqueeze(-1)
        state = psi_field_1d(coords)
        E = expected_energy(
            state,
            potential=lambda s: 0.5 * s.coords[..., 0] ** 2,
        )
        assert torch.isfinite(E)

    def test_expectation_value_decomposition(self, psi_field_1d):
        coords = torch.linspace(-2.0, 2.0, 31, dtype=torch.float64).unsqueeze(-1)
        state = psi_field_1d(coords)
        # Pure-kinetic operator
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
        torch.testing.assert_close(E_T + E_V, E_full, atol=1e-12, rtol=1e-12)

    def test_energy_variance_nonnegative_within_tolerance(self, psi_field_1d):
        coords = torch.linspace(-2.0, 2.0, 31, dtype=torch.float64).unsqueeze(-1)
        state = psi_field_1d(coords)
        var = energy_variance(
            state,
            potential=lambda s: 0.5 * s.coords[..., 0] ** 2,
        )
        # Variance can be slightly negative numerically; ensure it's
        # essentially zero or positive.
        assert float(var.detach()) >= -1e-10


class TestCurrentDiagnostics:
    def test_probability_current_shape(self, psi_field_xt):
        torch.manual_seed(1)
        coords = torch.randn(16, 2, dtype=torch.float64)
        state = psi_field_xt(coords)
        j = probability_current(state)
        assert j.shape == (16, 1)

    def test_current_divergence_shape(self, psi_field_xt):
        torch.manual_seed(1)
        coords = torch.randn(16, 2, dtype=torch.float64)
        state = psi_field_xt(coords)
        d = current_divergence(state)
        assert d.shape == (16,)

    def test_continuity_residual_requires_time(self, psi_field_1d):
        coords = torch.linspace(-1.0, 1.0, 5, dtype=torch.float64).unsqueeze(-1)
        state = psi_field_1d(coords)
        with pytest.raises(ValueError, match="time axis"):
            continuity_residual(state)

    def test_continuity_residual_finite(self, psi_field_xt):
        torch.manual_seed(1)
        coords = torch.randn(16, 2, dtype=torch.float64)
        state = psi_field_xt(coords)
        r = continuity_residual(state)
        assert r.shape == (16,)
        assert torch.isfinite(r).all()

    def test_zero_mass_rejected(self, psi_field_xt):
        torch.manual_seed(1)
        coords = torch.randn(8, 2, dtype=torch.float64)
        state = psi_field_xt(coords)
        with pytest.raises(ValueError, match="mass"):
            probability_current(state, mass=0.0)
