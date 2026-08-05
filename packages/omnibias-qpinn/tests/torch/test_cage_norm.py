# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Unit tests for the L^2-norm-conservation cage (torch backend)."""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from omnibias.pinn._core.coords import CoordinateSpec
from omnibias.pinn.torch.fields.one_layer import OneLayerVectorField
from omnibias.qpinn import make_psi_components
from omnibias.qpinn.torch.cage import (
    NormConservationField,
    make_norm_conservation_field,
    norm_loss,
)
from omnibias.qpinn.torch.diagnostics import norm_squared


def _build_base_field_1d():
    torch.manual_seed(0)
    coord = CoordinateSpec(axes=("x",))
    spec = make_psi_components(name="psi")
    return OneLayerVectorField(
        coordinate_spec=coord, components=spec, hidden=16,
        base="gaussian", dtype=torch.float64,
    ), coord


class TestNormCageConstructor:
    def test_builds_with_uniform_quadrature(self):
        base, _ = _build_base_field_1d()
        x = torch.linspace(-3.0, 3.0, 401, dtype=torch.float64).unsqueeze(-1)
        w = torch.full((401,), 6.0 / 401, dtype=torch.float64)
        cage = make_norm_conservation_field(
            base=base, quadrature_coords=x, quadrature_weights=w,
        )
        assert isinstance(cage, NormConservationField)
        assert cage.components.names == ("psi_re", "psi_im")

    def test_rejects_2d_weights(self):
        base, _ = _build_base_field_1d()
        x = torch.linspace(-3.0, 3.0, 5, dtype=torch.float64).unsqueeze(-1)
        w = torch.full((5, 1), 1.0, dtype=torch.float64)
        with pytest.raises(ValueError, match="1D"):
            make_norm_conservation_field(
                base=base, quadrature_coords=x, quadrature_weights=w,
            )

    def test_rejects_negative_weights(self):
        base, _ = _build_base_field_1d()
        x = torch.linspace(-3.0, 3.0, 5, dtype=torch.float64).unsqueeze(-1)
        w = torch.tensor([1.0, -0.1, 1.0, 1.0, 1.0], dtype=torch.float64)
        with pytest.raises(ValueError, match="non-negative"):
            make_norm_conservation_field(
                base=base, quadrature_coords=x, quadrature_weights=w,
            )

    def test_rejects_mismatched_ndim(self):
        base, _ = _build_base_field_1d()
        x = torch.zeros((5, 2), dtype=torch.float64)
        w = torch.ones((5,), dtype=torch.float64)
        with pytest.raises(ValueError, match="last dim"):
            make_norm_conservation_field(
                base=base, quadrature_coords=x, quadrature_weights=w,
            )


class TestNormCageNumerics:
    def test_caged_field_has_unit_norm(self):
        """The cage divides by sqrt(integral |psi|^2). On the same grid the
        integrated density of the caged field must be ~1."""
        base, _ = _build_base_field_1d()
        x = torch.linspace(-3.0, 3.0, 401, dtype=torch.float64).unsqueeze(-1)
        w = torch.full((401,), 6.0 / 401, dtype=torch.float64)
        cage = make_norm_conservation_field(
            base=base, quadrature_coords=x, quadrature_weights=w,
        )
        state = cage(x)
        nrm = norm_squared(state, quadrature_weights=w)
        assert math.isclose(float(nrm.detach()), 1.0, rel_tol=1e-12, abs_tol=1e-12)

    def test_derivatives_pass_through_division(self):
        """d/dx (psi_tilde / N) = (d psi_tilde / dx) / N because N is scalar.

        We can verify this by comparing cage derivative to base derivative / N
        at the same query points.
        """
        base, _ = _build_base_field_1d()
        x_grid = torch.linspace(-3.0, 3.0, 401, dtype=torch.float64).unsqueeze(-1)
        w = torch.full((401,), 6.0 / 401, dtype=torch.float64)
        cage = make_norm_conservation_field(
            base=base, quadrature_coords=x_grid, quadrature_weights=w,
        )
        query = torch.linspace(-2.0, 2.0, 11, dtype=torch.float64).unsqueeze(-1)
        cage_state = cage(query)
        base_state = base(query)
        # Recompute norm from grid for the comparison.
        grid_state = base(x_grid)
        psi_re_q = grid_state.ops.value(grid_state, "psi_re")
        psi_im_q = grid_state.ops.value(grid_state, "psi_im")
        N = torch.sqrt((w * (psi_re_q ** 2 + psi_im_q ** 2)).sum())

        d_re_cage = cage_state.ops.derivative(cage_state, "psi_re", axis=0, order=2)
        d_re_base = base_state.ops.derivative(base_state, "psi_re", axis=0, order=2)
        torch.testing.assert_close(d_re_cage * N, d_re_base, atol=1e-12, rtol=1e-12)


class TestSoftNormLoss:
    def test_zero_when_normalised(self):
        """When the field already has unit norm, the loss is ~0."""
        base, _ = _build_base_field_1d()
        x = torch.linspace(-3.0, 3.0, 401, dtype=torch.float64).unsqueeze(-1)
        w = torch.full((401,), 6.0 / 401, dtype=torch.float64)
        cage = make_norm_conservation_field(
            base=base, quadrature_coords=x, quadrature_weights=w,
        )
        state = cage(x)
        loss = norm_loss(state, quadrature_weights=w)
        assert float(loss.detach()) < 1e-20

    def test_nonzero_for_unnormalised_field(self):
        base, _ = _build_base_field_1d()
        x = torch.linspace(-3.0, 3.0, 401, dtype=torch.float64).unsqueeze(-1)
        w = torch.full((401,), 6.0 / 401, dtype=torch.float64)
        state = base(x)
        loss = norm_loss(state, quadrature_weights=w, target_norm=1.0)
        # No guarantee on direction, but it should be > 0 (random init).
        assert float(loss.detach()) > 1e-12
