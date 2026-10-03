# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
"""Regression tests for the exact-Q Poincare-Lyapunov focal-value engine.

Includes an independent ``sympy`` cross-oracle re-derivation of the first
focal value via the classical Guckenheimer-Holmes closed-form formula
(*Nonlinear Oscillations, Dynamical Systems, and Bifurcations of Vector
Fields*, eq. 3.4.11): for ``x' = -y + f(x,y)``, ``y' = x + g(x,y)`` with
``f, g`` starting at total degree ``>= 2``,

    16*V_1 = f_xxx + f_xyy + g_xxy + g_yyy
             + f_xy*(f_xx+f_yy) - g_xy*(g_xx+g_yy) - f_xx*g_xx + f_yy*g_yy

with every partial evaluated at the origin. This is a genuinely independent
recomputation (a different library, a different closed-form route), not a
replay of :func:`~omnibias.dynamics.focal.lyapunov_quantities`.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from fractions import Fraction as Q

import pytest
import sympy as sp
from omnibias.core.realization.polynomial import SparsePolynomial as P
from omnibias.dynamics.bautin import hamiltonian_center_example, nonzero_focus_example
from omnibias.dynamics.compactify import PlanarPolynomialField
from omnibias.dynamics.focal import (
    certify_focal_values,
    focal_order,
    lyapunov_quantities,
    normalize_monodromic,
    verify_focal_values,
)
from omnibias.holonomic._core.poly_n import PolyN


def _polyn_to_sympy(poly: PolyN, x: sp.Symbol, y: sp.Symbol) -> sp.Expr:
    expr = sp.Integer(0)
    for (i, j), c in poly.terms.items():
        expr += sp.Rational(c.numerator, c.denominator) * x**i * y**j
    return expr


def _guckenheimer_holmes_v1(p: PolyN, q: PolyN) -> Q:
    """Independent sympy re-derivation of the first focal value (see module docstring)."""
    x, y = sp.symbols("x y")
    f = _polyn_to_sympy(p, x, y)
    g = _polyn_to_sympy(q, x, y)

    def d(expr: sp.Expr, *wrt: sp.Symbol) -> sp.Expr:
        out = expr
        for var in wrt:
            out = sp.diff(out, var)
        return out.subs({x: 0, y: 0})

    fxx, fxy, fyy = d(f, x, x), d(f, x, y), d(f, y, y)
    fxxx, fxyy = d(f, x, x, x), d(f, x, y, y)
    gxx, gxy, gyy = d(g, x, x), d(g, x, y), d(g, y, y)
    gxxy, gyyy = d(g, x, x, y), d(g, y, y, y)
    sixteen_v1 = (
        fxxx + fxyy + gxxy + gyyy + fxy * (fxx + fyy) - gxy * (gxx + gyy) - fxx * gxx + fyy * gyy
    )
    value = sp.nsimplify(sixteen_v1 / 16)
    assert value.is_rational
    r = sp.Rational(value)
    return Q(int(r.p), int(r.q))


def _field_from_perturbation(p_pert: PolyN, q_pert: PolyN, *, degree: int = 3) -> PlanarPolynomialField:
    x, y = P.variable(2, 0), P.variable(2, 1)
    p = -y + P(2, dict(p_pert.terms))
    q = x + P(2, dict(q_pert.terms))
    return PlanarPolynomialField(p, q, degree)


def test_nonzero_focus_matches_sympy_cross_oracle() -> None:
    p_pert, q_pert = nonzero_focus_example()
    quantities = lyapunov_quantities(p_pert, q_pert, order=4)
    assert focal_order(quantities) == 4
    ((degree, value),) = quantities.quantities
    assert degree == 4
    computed = value.terms[()]
    expected = _guckenheimer_holmes_v1(p_pert, q_pert)
    assert computed == expected == Q(3, 4)


def test_hamiltonian_center_matches_sympy_cross_oracle_at_zero() -> None:
    p_pert, q_pert = hamiltonian_center_example()
    quantities = lyapunov_quantities(p_pert, q_pert, order=6)
    assert focal_order(quantities) is None
    for _degree, value in quantities.quantities:
        assert value.is_zero()
    expected = _guckenheimer_holmes_v1(p_pert, q_pert)
    assert expected == Q(0)


def test_certify_and_verify_focal_values_round_trip() -> None:
    p_pert, q_pert = nonzero_focus_example()
    field = _field_from_perturbation(p_pert, q_pert)
    certificate = certify_focal_values(field, order=4)
    assert verify_focal_values(certificate)
    assert certificate.formal_seal["honesty"]["bautin_ideal_stabilization_proved"] is False
    assert certificate.formal_seal["honesty"]["physical_return_membership_proved"] is False


def test_focal_certificate_tamper_rejected() -> None:
    p_pert, q_pert = nonzero_focus_example()
    field = _field_from_perturbation(p_pert, q_pert)
    certificate = certify_focal_values(field, order=4)
    # A source-digest mismatch is rejected outright.
    assert not verify_focal_values(replace(certificate, source_digest="tampered"))
    # A seal whose stored digest no longer matches its payload is rejected.
    broken_seal = deepcopy(certificate.formal_seal)
    broken_seal["payload"]["equations"] = []
    assert not verify_focal_values(replace(certificate, formal_seal=broken_seal))
    # Re-sealing after tampering fixes the digest but not the underlying
    # quantities recomputation: swapping in a different (self-consistent)
    # LyapunovQuantities value must still be caught.
    zero_field = hamiltonian_center_example()
    fabricated = lyapunov_quantities(*zero_field, order=4)
    assert not verify_focal_values(replace(certificate, quantities=fabricated))


def test_normalize_monodromic_rejects_non_square_determinant() -> None:
    x, y = P.variable(2, 0), P.variable(2, 1)
    # trace 0, det = 2 (not a rational square) -> must refuse rather than
    # drift into an algebraic-number linear part.
    field = PlanarPolynomialField(-2 * y + x * x, x + y * y, 2)
    with pytest.raises(ValueError, match="strictly positive determinant|square"):
        normalize_monodromic(field)


def test_normalize_monodromic_rejects_nonzero_trace() -> None:
    x, y = P.variable(2, 0), P.variable(2, 1)
    field = PlanarPolynomialField(x - y + x * x, x + y + y * y, 2)
    with pytest.raises(ValueError, match="zero trace"):
        normalize_monodromic(field)


def test_normalize_monodromic_rejects_off_equilibrium_origin() -> None:
    x, y = P.variable(2, 0), P.variable(2, 1)
    field = PlanarPolynomialField(-y + P.constant(2, 1), x, 2)
    with pytest.raises(ValueError, match="equilibrium"):
        normalize_monodromic(field)


def test_lyapunov_quantities_rejects_low_order_and_linear_terms() -> None:
    p_pert, q_pert = nonzero_focus_example()
    with pytest.raises(ValueError, match="order"):
        lyapunov_quantities(p_pert, q_pert, order=2)
    x, y = PolyN.var(2, 0), PolyN.var(2, 1)
    with pytest.raises(ValueError, match="degree >= 2"):
        lyapunov_quantities(x, y * y, order=4)
