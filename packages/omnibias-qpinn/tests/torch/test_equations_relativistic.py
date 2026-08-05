# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Unit tests for Helmholtz / Klein-Gordon / Dirac residuals (torch)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from omnibias.pinn._core.components import ComponentSpec
from omnibias.pinn._core.coords import CoordinateSpec
from omnibias.pinn.torch.fields.one_layer import OneLayerVectorField
from omnibias.qpinn import (
    make_psi_components,
    make_spinor_components,
)
from omnibias.qpinn.torch.equations import (
    Dirac,
    DiracOutput,
    Helmholtz,
    HelmholtzOutput,
    KleinGordon,
    KleinGordonOutput,
    dirac,
    helmholtz,
    klein_gordon,
)

# ----- Helmholtz ---------------------------------------------------

class TestHelmholtz:
    def _build_field(self):
        torch.manual_seed(0)
        coord = CoordinateSpec(axes=("x", "y"))
        spec = make_psi_components(name="psi")
        return OneLayerVectorField(
            coordinate_spec=coord, components=spec,
            hidden=8, base="gaussian", dtype=torch.float64,
        )

    def test_shape(self):
        field = self._build_field()
        coords = torch.randn(20, 2, dtype=torch.float64)
        state = field(coords)
        out = Helmholtz(k=2.0)(state)
        assert isinstance(out, HelmholtzOutput)
        assert out.residual.shape == (20, 2)

    def test_callable_k(self):
        """Position-dependent index of refraction."""
        field = self._build_field()
        coords = torch.randn(20, 2, dtype=torch.float64)
        state = field(coords)

        def k_callable(s):
            return 1.0 + 0.5 * s.coords[..., 0] ** 2

        out = Helmholtz(k=k_callable)(state)
        assert out.residual.shape == (20, 2)

    def test_function_form(self):
        field = self._build_field()
        coords = torch.randn(20, 2, dtype=torch.float64)
        state = field(coords)
        out_cls = Helmholtz(k=2.0)(state)
        out_fn = helmholtz(state, k=2.0)
        torch.testing.assert_close(out_cls.residual, out_fn.residual)


# ----- Klein-Gordon ------------------------------------------------

class TestKleinGordon:
    def _build_field(self, mass_term=False):
        torch.manual_seed(0)
        coord = CoordinateSpec(axes=("x", "t"))
        spec = ComponentSpec(("phi",))
        return OneLayerVectorField(
            coordinate_spec=coord, components=spec,
            hidden=8, base="gaussian", dtype=torch.float64,
        )

    def test_shape(self):
        field = self._build_field()
        coords = torch.randn(20, 2, dtype=torch.float64)
        state = field(coords)
        out = KleinGordon(mass=1.0)(state)
        assert isinstance(out, KleinGordonOutput)
        assert out.residual.shape == (20,)

    def test_phi4_changes_residual(self):
        field = self._build_field()
        coords = torch.randn(20, 2, dtype=torch.float64)
        state = field(coords)
        out_a = KleinGordon(mass=1.0, lambda_phi4=0.0)(state)
        out_b = KleinGordon(mass=1.0, lambda_phi4=0.5)(state)
        assert not torch.allclose(out_a.residual, out_b.residual)

    def test_requires_time(self):
        torch.manual_seed(0)
        coord = CoordinateSpec(axes=("x",))
        spec = ComponentSpec(("phi",))
        field = OneLayerVectorField(
            coordinate_spec=coord, components=spec,
            hidden=8, base="gaussian", dtype=torch.float64,
        )
        coords = torch.linspace(-1.0, 1.0, 5, dtype=torch.float64).unsqueeze(-1)
        state = field(coords)
        with pytest.raises(ValueError, match="time axis"):
            KleinGordon()(state)

    def test_function_form(self):
        field = self._build_field()
        coords = torch.randn(20, 2, dtype=torch.float64)
        state = field(coords)
        out_cls = KleinGordon(mass=1.5)(state)
        out_fn = klein_gordon(state, mass=1.5)
        torch.testing.assert_close(out_cls.residual, out_fn.residual)


# ----- Dirac -------------------------------------------------------

class TestDirac:
    def _build_field(self, axes=("x", "y", "z", "t")):
        torch.manual_seed(0)
        coord = CoordinateSpec(axes=axes)
        spec = make_spinor_components(name="spinor", n_components=4)
        return OneLayerVectorField(
            coordinate_spec=coord, components=spec,
            hidden=8, base="gaussian", dtype=torch.float64,
        )

    def test_shape(self):
        field = self._build_field()
        coords = torch.randn(16, 4, dtype=torch.float64)
        state = field(coords)
        out = Dirac(mass=1.0)(state)
        assert isinstance(out, DiracOutput)
        assert out.residual.shape == (16, 8)

    def test_function_form(self):
        field = self._build_field()
        coords = torch.randn(16, 4, dtype=torch.float64)
        state = field(coords)
        out_cls = Dirac(mass=0.5, representation="dirac")(state)
        out_fn = dirac(state, mass=0.5, representation="dirac")
        torch.testing.assert_close(out_cls.residual, out_fn.residual)

    def test_representation_changes_residual(self):
        """Dirac vs Weyl representations are unitarily equivalent but
        produce different split-real residual tensors."""
        field = self._build_field()
        coords = torch.randn(16, 4, dtype=torch.float64)
        state = field(coords)
        out_d = Dirac(mass=1.0, representation="dirac")(state)
        out_w = Dirac(mass=1.0, representation="weyl")(state)
        assert not torch.allclose(out_d.residual, out_w.residual)

    def test_works_in_lower_dimensions(self):
        """Dirac residual should work even when only x + t are provided
        (1+1 dimensional Dirac); the unused spatial gammas drop out."""
        field = self._build_field(axes=("x", "t"))
        coords = torch.randn(16, 2, dtype=torch.float64)
        state = field(coords)
        out = Dirac(mass=1.0)(state)
        assert out.residual.shape == (16, 8)
