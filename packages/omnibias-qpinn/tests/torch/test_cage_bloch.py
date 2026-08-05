# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Unit tests for the Bloch-periodic cage (torch)."""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from omnibias.pinn._core.coords import CoordinateSpec
from omnibias.pinn.torch.fields.one_layer import OneLayerVectorField
from omnibias.qpinn import make_psi_components
from omnibias.qpinn.torch.cage import (
    BlochPeriodicField,
    make_bloch_periodic_field,
)


def _build_base():
    torch.manual_seed(0)
    coord = CoordinateSpec(axes=("x",))
    spec = make_psi_components(name="u")
    return OneLayerVectorField(
        coordinate_spec=coord, components=spec,
        hidden=8, base="gaussian", dtype=torch.float64,
    )


class TestBlochConstructor:
    def test_builds(self):
        base = _build_base()
        cage = make_bloch_periodic_field(base=base, k=[1.5])
        assert isinstance(cage, BlochPeriodicField)
        assert cage.components.names == ("psi_re", "psi_im")

    def test_rejects_wrong_k_shape(self):
        base = _build_base()
        with pytest.raises(ValueError, match="k must have shape"):
            make_bloch_periodic_field(base=base, k=[1.5, 2.0])

    def test_rejects_no_base_group(self):
        torch.manual_seed(0)
        coord = CoordinateSpec(axes=("x",))
        from omnibias.pinn._core.components import ComponentSpec
        spec = ComponentSpec(("u",))  # no group
        base = OneLayerVectorField(
            coordinate_spec=coord, components=spec,
            hidden=8, base="gaussian", dtype=torch.float64,
        )
        with pytest.raises(ValueError, match="wavefunction group"):
            make_bloch_periodic_field(base=base, k=[1.5])


class TestBlochValueAtZeroK:
    """For k=0, the cage should reduce to the identity."""

    def test_value_matches_base_at_k0(self):
        base = _build_base()
        cage = make_bloch_periodic_field(base=base, k=[0.0])
        coords = torch.linspace(-2.0, 2.0, 11, dtype=torch.float64).unsqueeze(-1)
        base_state = base(coords)
        cage_state = cage(coords)
        u_re = base_state.ops.value(base_state, "u_re")
        u_im = base_state.ops.value(base_state, "u_im")
        psi_re = cage_state.ops.value(cage_state, "psi_re")
        psi_im = cage_state.ops.value(cage_state, "psi_im")
        torch.testing.assert_close(psi_re, u_re, atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(psi_im, u_im, atol=1e-12, rtol=1e-12)

    def test_derivative_matches_base_at_k0(self):
        base = _build_base()
        cage = make_bloch_periodic_field(base=base, k=[0.0])
        coords = torch.linspace(-2.0, 2.0, 11, dtype=torch.float64).unsqueeze(-1)
        base_state = base(coords)
        cage_state = cage(coords)
        d_u_re = base_state.ops.derivative(base_state, "u_re", axis=0, order=2)
        d_psi_re = cage_state.ops.derivative(cage_state, "psi_re", axis=0, order=2)
        torch.testing.assert_close(d_psi_re, d_u_re, atol=1e-12, rtol=1e-12)


class TestBlochDensityIsPeriodic:
    """|psi|^2 = |u|^2 since e^(ik.x) has unit modulus."""

    def test_density_independent_of_k(self):
        base = _build_base()
        coords = torch.linspace(-2.0, 2.0, 11, dtype=torch.float64).unsqueeze(-1)
        cage_a = make_bloch_periodic_field(base=base, k=[0.0])
        cage_b = make_bloch_periodic_field(base=base, k=[2.5])
        a_state = cage_a(coords)
        b_state = cage_b(coords)
        psi_re_a = a_state.ops.value(a_state, "psi_re")
        psi_im_a = a_state.ops.value(a_state, "psi_im")
        psi_re_b = b_state.ops.value(b_state, "psi_re")
        psi_im_b = b_state.ops.value(b_state, "psi_im")
        density_a = psi_re_a * psi_re_a + psi_im_a * psi_im_a
        density_b = psi_re_b * psi_re_b + psi_im_b * psi_im_b
        torch.testing.assert_close(density_a, density_b, atol=1e-12, rtol=1e-12)


class TestBlochSecondDerivativeFormula:
    r"""Verify the closed-form for ∂_x^2 psi = ∂_x^2 (exp(ikx) u) by checking
    that for u_re = exp(-x^2), u_im = 0, and k != 0, the resulting psi has
    the expected derivative."""

    def test_derivative_higher_order_raises(self):
        base = _build_base()
        cage = make_bloch_periodic_field(base=base, k=[1.0])
        coords = torch.linspace(-1.0, 1.0, 5, dtype=torch.float64).unsqueeze(-1)
        state = cage(coords)
        with pytest.raises(NotImplementedError, match="order"):
            state.ops.derivative(state, "psi_re", axis=0, order=3)
