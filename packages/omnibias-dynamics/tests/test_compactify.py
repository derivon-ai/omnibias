# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
"""Exact Poincare compactification and equator-root regressions."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from fractions import Fraction as Q

import sympy as sp
from omnibias.core.proof.certificate import seal_certificate
from omnibias.core.realization.polynomial import SparsePolynomial as P
from omnibias.dynamics.compactify import (
    PlanarPolynomialField,
    certify_poincare_compactification,
    infinite_singular_points,
    poincare_chart,
    verify_poincare_compactification,
    verify_poincare_compactification_formally,
)


def _sympy(polynomial: P, variables: tuple[sp.Symbol, ...]) -> sp.Expr:
    return sp.Add(
        *(
            sp.Rational(coefficient.numerator, coefficient.denominator)
            * sp.prod(
                variable**power
                for variable, power in zip(variables, index, strict=True)
            )
            for index, coefficient in polynomial.terms
        )
    )


def test_chart_index_remap_matches_direct_symbolic_substitution() -> None:
    x, y = (P.variable(2, axis) for axis in range(2))
    p = Q(1, 3) + 2 * x - 3 * y + 5 * x**2 + 7 * x * y - 11 * y**2
    q = -2 + 13 * x + 17 * y - 19 * x**2 + 23 * x * y + 29 * y**2
    field = PlanarPolynomialField(p, q, 2)
    u1 = poincare_chart(field, "U1")
    u2 = poincare_chart(field, "U2")

    u, v = sp.symbols("u v")
    sx, sy = sp.symbols("x y")
    p_expr, q_expr = _sympy(p, (sx, sy)), _sympy(q, (sx, sy))
    assert sp.expand(_sympy(u1.tangential, (u, v)) - v**2 * (
        q_expr.subs({sx: 1 / v, sy: u / v})
        - u * p_expr.subs({sx: 1 / v, sy: u / v})
    )) == 0
    assert sp.expand(_sympy(u1.normal, (u, v)) + v**3 * p_expr.subs(
        {sx: 1 / v, sy: u / v}
    )) == 0
    assert sp.expand(_sympy(u2.tangential, (u, v)) - v**2 * (
        p_expr.subs({sx: u / v, sy: 1 / v})
        - u * q_expr.subs({sx: u / v, sy: 1 / v})
    )) == 0
    assert sp.expand(_sympy(u2.normal, (u, v)) + v**3 * q_expr.subs(
        {sx: u / v, sy: 1 / v}
    )) == 0
    assert u1.equator_invariant and u2.equator_invariant


def test_exact_and_interval_equator_singularities_are_complete() -> None:
    x, y = (P.variable(2, axis) for axis in range(2))
    field = PlanarPolynomialField(P.constant(2, 0), y**2 - 2 * x**2, 2)
    u1 = poincare_chart(field, "U1")
    roots = infinite_singular_points(u1, subdivisions=1024)
    assert len(roots) == 2
    assert all(root.exact_coordinate is None and root.multiplicity == 1 for root in roots)
    assert roots[0].enclosure.contains(-(2**0.5))
    assert roots[1].enclosure.contains(2**0.5)


def test_multiple_rational_root_records_its_multiplicity() -> None:
    x, y = (P.variable(2, axis) for axis in range(2))
    field = PlanarPolynomialField(y, x - x**2, 2)
    roots = infinite_singular_points(poincare_chart(field, "U2"))
    assert len(roots) == 1
    assert roots[0].exact_coordinate == 0
    assert roots[0].multiplicity == 3


def test_compactification_replays_tamper_rejects_and_kernel_checks() -> None:
    x, y = (P.variable(2, axis) for axis in range(2))
    field = PlanarPolynomialField(-2 * x + x**2, y + y**2, 2)
    certificate = certify_poincare_compactification(
        field,
        root_subdivisions=512,
    )
    assert certificate.source_digest == field.digest
    assert len(certificate.charts) == 3
    assert verify_poincare_compactification(certificate)
    assert not verify_poincare_compactification(
        replace(certificate, source_digest="tampered")
    )
    altered = deepcopy(certificate.formal_seal)
    altered["payload"]["equations"][0]["rhs"] += 1
    altered = seal_certificate(altered)
    assert not verify_poincare_compactification(
        replace(certificate, formal_seal=altered)
    )
    formal = verify_poincare_compactification_formally(certificate)
    if formal.result.available:
        assert formal.theorem_prover_verified
    else:
        assert not formal.theorem_prover_verified


def test_invalid_field_degree_and_chart_are_rejected() -> None:
    x, y = (P.variable(2, axis) for axis in range(2))
    try:
        PlanarPolynomialField(x**2, y, 1)
    except ValueError as exc:
        assert "degree" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("degree overflow accepted")
    try:
        poincare_chart(PlanarPolynomialField(x, y, 2), "U4")  # type: ignore[arg-type]
    except ValueError as exc:
        assert "U1" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("invalid chart accepted")
