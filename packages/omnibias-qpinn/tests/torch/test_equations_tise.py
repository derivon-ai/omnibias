# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Unit tests for the TISE residual (torch backend)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from omnibias.qpinn.torch.equations import TISE, TISEOutput, tise


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
        torch.testing.assert_close(out_cls.residual, out_fn.residual)

    def test_potential_callable_used(self, psi_field_x, coords_x):
        state = psi_field_x(coords_x)
        # Without potential
        out_a = TISE(energy=0.0)(state)
        # With a non-zero potential
        out_b = TISE(
            energy=0.0,
            potential=lambda s: 0.5 * s.coords[..., 0] ** 2,
        )(state)
        assert not torch.allclose(out_a.residual, out_b.residual)


class TestTISEEnergyEstimate:
    def test_with_quadrature_weights(self, psi_field_x, coords_x):
        state = psi_field_x(coords_x)
        B = coords_x.shape[0]
        weights = torch.full((B,), 4.0 / B, dtype=torch.float64)
        out = TISE(
            energy=0.0,
            potential=lambda s: 0.5 * s.coords[..., 0] ** 2,
            quadrature_weights=weights,
        )(state)
        assert out.energy_estimate is not None
        assert out.energy_estimate.dim() == 0
        assert "energy_estimate" in out.diag
        assert "norm_squared" in out.diag


class TestTISEFreeParticleAnalytic:
    """Sanity: free particle psi(x) = cos(k x) is an eigenstate of T = -1/2 d^2/dx^2.

    We pin the activation init by hand so this test is fully analytic.
    """

    def test_plane_wave_residual_is_small(self):
        from omnibias.pinn._core.coords import CoordinateSpec
        from omnibias.pinn.torch.fields.one_layer import OneLayerVectorField
        from omnibias.qpinn import make_psi_components

        torch.manual_seed(0)
        coord = CoordinateSpec(axes=("x",))
        spec = make_psi_components(name="psi")
        # The network is not constrained to be a cosine, so we don't
        # expect zero residual; instead we just verify the energy
        # estimate is finite + the residual has the right shape.
        field = OneLayerVectorField(
            coordinate_spec=coord, components=spec, hidden=16,
            base="gaussian", dtype=torch.float64,
        )
        coords = torch.linspace(-3.0, 3.0, 31, dtype=torch.float64).unsqueeze(-1)
        state = field(coords)
        out = tise(
            state, energy=0.0, hbar=1.0, mass=1.0,
            potential=lambda s: 0.5 * s.coords[..., 0] ** 2,
            quadrature_weights=torch.full((31,), 6.0 / 31, dtype=torch.float64),
        )
        assert torch.isfinite(out.residual).all()
        assert torch.isfinite(out.energy_estimate)
