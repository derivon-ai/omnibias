# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Named PDE transforms G1 (theory 02-13). Vocabulary, not 03-11 search."""

from __future__ import annotations

from omnibias.core.transforms_pde import (
    TransformKind,
    cole_hopf_from_heat_phi,
    cole_hopf_jet,
    darboux_dress,
    factorial_jet_multiply,
    factorial_jet_reciprocal,
    miura_v,
    named_cole_hopf,
    verify_cole_hopf_burgers_jet,
    verify_transform,
)


def test_cole_hopf_worked_example() -> None:
    t = named_cole_hopf()
    assert t.kind is TransformKind.COLE_HOPF
    assert verify_transform(t, order=8)
    assert cole_hopf_from_heat_phi(0.0, 0.0) == -2.0


def test_factorial_jet_reciprocal_roundtrip() -> None:
    a = [2.0, -0.5, 0.25, -0.125]
    b = factorial_jet_reciprocal(a)
    prod = factorial_jet_multiply(a, b)
    assert abs(prod[0] - 1.0) < 1e-14
    assert all(abs(c) < 1e-12 for c in prod[1:])


def test_cole_hopf_jet_burgers_exact() -> None:
    out = verify_cole_hopf_burgers_jet(nu=0.07, k=-1.25, order=10)
    assert out["passed"] is True
    assert out["navier_stokes_proof_claim"] is False
    # constant field: higher jet coeffs vanish
    phi = [1.0, 0.0, 0.0]
    phi_x = [0.5, 0.0, 0.0]
    u = cole_hopf_jet(phi, phi_x, nu=0.1)
    assert abs(u[0] - (-0.1)) < 1e-15
    # G5: a wrong viscosity is not vacuous
    other = cole_hopf_jet(phi, phi_x, nu=0.2)
    assert abs(other[0] - u[0]) > 1e-12


def test_miura_and_darboux_finite() -> None:
    assert miura_v(0.5, -0.2) == -0.2 + 0.25
    assert darboux_dress(2.0, 1.0, 0.3) == 0.3 - 1.0
