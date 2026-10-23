# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Global polynomial identities distinguish realizable jets from point values."""

from __future__ import annotations

from fractions import Fraction as F

import pytest
from omnibias.core.jet_geometry import (
    JetGeometryBudgetExceeded,
    LinearJetEquation,
    PolynomialJetBoundary,
    PolynomialJetBudget,
    PolynomialJetCertificate,
    check_polynomial_jet,
    polynomial_jet,
)
from omnibias.core.realization.polynomial import AlgebraBudgetExceeded
from omnibias.core.realization.polynomial import SparsePolynomial as P


def test_multivariate_raw_derivative_jet_and_exact_wave_identity() -> None:
    x, t = P.variable(2, 0), P.variable(2, 1)
    # Right-moving cubic wave u=(x-2t)^3: u_tt-4u_xx=0.
    field = (x - 2 * t)**3
    section = polynomial_jet(field, 3)
    assert section[(1, 0)].terms == (3 * (x - 2 * t)**2).terms
    assert section[(2, 0)].terms == (6 * (x - 2 * t)).terms
    assert section[(3, 0)].terms == P.constant(2, 6).terms  # RAW, not coefficient 1.
    certificate = check_polynomial_jet(
        section,
        equations=(LinearJetEquation({(0, 2): 1, (2, 0): -4}, label="wave"),),
        boundaries=(PolynomialJetBoundary((0, 0), 1, 0, x**3, "initial displacement"),
                    PolynomialJetBoundary((0, 1), 1, 0, -6 * x**2, "initial velocity")),
    )
    assert certificate.valid and certificate.holonomic and not certificate.failures
    assert certificate.field.terms == field.terms
    assert certificate.dimension == 2 and certificate.order == 3
    assert len(certificate.fingerprint) == 64
    assert certificate.fingerprint == check_polynomial_jet(
        dict(reversed(list(section.items()))),
        equations=(LinearJetEquation({(0, 2): 1, (2, 0): -4}, label="wave"),),
        boundaries=(PolynomialJetBoundary((0, 0), 1, 0, x**3, "initial displacement"),
                    PolynomialJetBoundary((0, 1), 1, 0, -6 * x**2, "initial velocity")),
    ).fingerprint


def test_algebraic_pde_relation_without_contact_is_rejected() -> None:
    x = P.variable(1, 0)
    # Both coordinates satisfy p=u, but derivative(x)=1 differs from p=x.
    certificate = check_polynomial_jet({(0,): x, (1,): x},
                                      equations=(LinearJetEquation({(1,): 1, (0,): -1}),))
    assert not certificate.valid and not certificate.holonomic
    assert len(certificate.failures) == 1
    witness = certificate.failures[0]
    assert witness.kind == "contact" and witness.residual.terms == (P.constant(1, 1) - x).terms
    assert next(check for check in certificate.checks if check.kind == "equation").passed


def test_mixed_partial_incompatibility_and_finite_sample_false_pass() -> None:
    x = P.variable(2, 0)
    zero = P.constant(2, 0)
    certificate = check_polynomial_jet({(0, 0): zero, (1, 0): zero, (0, 1): x})
    assert not certificate.valid
    assert any(check.residual.terms == (-x).terms for check in certificate.failures)
    # A nonzero polynomial can vanish on every supplied finite sample. The
    # global checker retains its coefficient witness instead of sampling it.
    one_dimensional = P.variable(1, 0)
    fake = one_dimensional * (one_dimensional - 1) * (one_dimensional + 1)
    sampled = check_polynomial_jet({(0,): P.constant(1, 0), (1,): fake})
    assert all(fake.evaluate((point,)) == 0 for point in (-1, 0, 1))
    assert not sampled.holonomic and sampled.failures[0].residual.terms


def test_variable_coefficient_equations_and_boundary_failures() -> None:
    x, t = P.variable(2, 0), P.variable(2, 1)
    field = x**2 + 3 * t
    section = polynomial_jet(field, 2)
    certificate = check_polynomial_jet(
        section,
        equations=(LinearJetEquation({(1, 0): x}, 2 * x**2, "Euler identity"),
                   LinearJetEquation({(0, 1): 1}, 4, "wrong derivative")),
        boundaries=(PolynomialJetBoundary((0, 0), 0, F(1, 2), F(1, 4) + 3 * t, "correct boundary"),
                    PolynomialJetBoundary((1, 0), 0, 1, 3, "wrong Neumann data")),
    )
    assert certificate.holonomic and not certificate.valid
    assert [(failure.kind, failure.label) for failure in certificate.failures] == [
        ("equation", "wrong derivative"), ("boundary", "wrong Neumann data")]
    assert all(failure.residual.terms == P.constant(2, -1).terms for failure in certificate.failures)


def test_zero_order_and_zero_polynomial_are_valid() -> None:
    field = P.constant(3, 0)
    assert check_polynomial_jet(polynomial_jet(field, 0)).valid
    assert check_polynomial_jet(polynomial_jet(field, 4)).valid


@pytest.mark.parametrize("section", [
    {},
    {(0,): P.constant(1, 1), (2,): P.constant(1, 0)},
    {(0,): P.constant(1, 1), (-1,): P.constant(1, 0)},
    {(0,): P.constant(2, 1)},
    {(False,): P.constant(1, 1)},
    {(0,): 1},
    {(0,): P.constant(1, 0), (1,): P.constant(2, 0)},
])
def test_invalid_or_incomplete_sections_raise(section: dict) -> None:
    with pytest.raises((ValueError, TypeError)):
        check_polynomial_jet(section)


def test_invalid_equation_and_boundary_requests_raise() -> None:
    section = polynomial_jet(P.variable(1, 0), 1)
    for equation in (LinearJetEquation({(2,): 1}), LinearJetEquation({(0,): 1.0}),
                     LinearJetEquation({(0,): 1}, label="")):
        with pytest.raises((ValueError, TypeError)):
            check_polynomial_jet(section, equations=(equation,))
    for boundary in (PolynomialJetBoundary((2,), 0, 0), PolynomialJetBoundary((0,), 1, 0),
                     PolynomialJetBoundary((0,), True, 0), PolynomialJetBoundary((0,), 0, 0.1)):
        with pytest.raises((ValueError, TypeError)):
            check_polynomial_jet(section, boundaries=(boundary,))


@pytest.mark.parametrize("budget", [
    PolynomialJetBudget(max_dimension=1),
    PolynomialJetBudget(max_order=1),
    PolynomialJetBudget(max_coordinates=2),
    PolynomialJetBudget(max_terms=1),
    PolynomialJetBudget(max_total_terms=1),
    PolynomialJetBudget(max_degree=1),
    PolynomialJetBudget(max_bits=1),
    PolynomialJetBudget(max_work=1),
])
def test_cumulative_and_polynomial_budgets_refuse_without_false_verdict(budget: PolynomialJetBudget) -> None:
    x, y = P.variable(2, 0), P.variable(2, 1)
    with pytest.raises((JetGeometryBudgetExceeded, AlgebraBudgetExceeded)):
        check_polynomial_jet(polynomial_jet(x**2 + 3 * y, 2), budget=budget)


def test_checked_boundary_powers_refuse_large_rational_intermediates() -> None:
    x = P.variable(1, 0)
    with pytest.raises((JetGeometryBudgetExceeded, AlgebraBudgetExceeded)):
        check_polynomial_jet(polynomial_jet(x**6, 1),
                             boundaries=(PolynomialJetBoundary((0,), 0, 7),),
                             budget=PolynomialJetBudget(max_bits=8))


def test_nonlinear_multiplication_has_unrepresented_degree() -> None:
    x = P.variable(1, 0)
    # span{1,x} is derivative-closed, but x*x requires a new degree-two slot.
    assert (x * x).terms == (((2,), F(1)),)
    assert (x * x).terms != x.terms
    # Finite quotient arithmetic x^2=0 would give a different real function.
    assert (x * x).evaluate((2,)) == 4


def test_certificate_requires_checker_and_input_size_is_preflighted() -> None:
    from collections.abc import Mapping

    with pytest.raises(TypeError, match="check_polynomial_jet"):
        PolynomialJetCertificate()

    class OversizedMapping(Mapping):
        def __len__(self):
            return 100_000

        def __iter__(self):
            raise AssertionError("oversized section must refuse before traversal")

        def __getitem__(self, key):
            raise AssertionError("oversized section must refuse before lookup")

    with pytest.raises(JetGeometryBudgetExceeded, match="max_coordinates"):
        check_polynomial_jet(OversizedMapping())


def test_moving_plane_solves_variable_coefficient_shear_transport() -> None:
    x, y, t = (P.variable(3, axis) for axis in range(3))
    # v=(y,0), w(t)=(1,-t): w'=-A.T w for A=[[0,1],[0,0]].
    field = (x - t * y) ** 3
    certificate = check_polynomial_jet(
        polynomial_jet(field, 1),
        equations=(LinearJetEquation({(0, 0, 1): 1, (1, 0, 0): y}, label="shear transport"),),
        boundaries=(PolynomialJetBoundary((0, 0, 0), 2, 0, x ** 3, "initial field"),),
    )
    assert certificate.valid and certificate.holonomic
    # A stationary plane lacks the geometric evolution required by this PDE.
    stationary = check_polynomial_jet(
        polynomial_jet(x ** 3, 1),
        equations=(LinearJetEquation({(0, 0, 1): 1, (1, 0, 0): y}),),
    )
    assert not stationary.valid and stationary.holonomic
    assert stationary.failures[0].residual.terms == (3 * y * x ** 2).terms
