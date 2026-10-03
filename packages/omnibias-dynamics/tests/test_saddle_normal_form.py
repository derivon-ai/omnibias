# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
"""Regression tests for exact-Q Poincare-Dulac normalization at a hyperbolic
saddle, the derived corner expansion, and its composition with a regular-arc
return Jacobian.

The corner-expansion cross-check reproduces the module docstring's
hand-solvable Bernoulli example: ``lambda1=-1``, ``lambda2=1``,
``p_star = c*xi**2*eta`` (a single resonant monomial) has the *exact* closed
form ``y = x/(1+c*x*log(x))``; the first-order derivation this module
implements must match that closed form's own first-order Taylor expansion in
``c*x*log(x)`` exactly.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from fractions import Fraction as Q

import pytest
from omnibias.core.proof.certificate import seal_certificate
from omnibias.core.verified.interval import Interval
from omnibias.dynamics.graphic import (
    named_irrational_hyperbolic_graphic,
    named_rational_hyperbolic_graphic,
)
from omnibias.dynamics.return_maps import StoppedEventResult
from omnibias.dynamics.saddle_normal_form import (
    DiagonalSaddleField,
    certify_resonant_normal_form,
    derive_return_map,
    diagonalize_saddle,
    dulac_corner_expansion,
    resonant_normal_form,
    verify_resonant_normal_form,
)
from omnibias.holonomic._core.poly_n import PolyN


def _bernoulli_diagonal(c: Q) -> DiagonalSaddleField:
    xi, eta = PolyN.var(2, 0), PolyN.var(2, 1)
    p = c * xi * xi * eta
    q = PolyN.zero(2)
    return DiagonalSaddleField(Q(-1), Q(1), p, q, ((Q(1), Q(0)), (Q(0), Q(1))), (Q(0), Q(0)))


def test_diagonalize_saddle_on_the_named_rational_graphic() -> None:
    target = named_rational_hyperbolic_graphic()
    diag = diagonalize_saddle(target.field, target.saddle)
    assert diag.lambda1 == -2
    assert diag.lambda2 == 1
    assert all(sum(mon) >= 2 for mon in diag.p.terms)
    assert all(sum(mon) >= 2 for mon in diag.q.terms)


def test_diagonalize_saddle_refuses_irrational_eigenvalues() -> None:
    target = named_irrational_hyperbolic_graphic()
    assert not target.saddle.stable.exact
    assert not target.saddle.unstable.exact
    with pytest.raises(ValueError, match="exact rational eigenvalues"):
        diagonalize_saddle(target.field, target.saddle)


def test_diagonal_saddle_field_rejects_wrong_sign_eigenvalues() -> None:
    xi = PolyN.var(2, 0)
    with pytest.raises(ValueError, match="lambda1 < 0 < lambda2"):
        DiagonalSaddleField(
            Q(1), Q(2), xi * xi, PolyN.zero(2), ((Q(1), Q(0)), (Q(0), Q(1))), (Q(0), Q(0))
        )


def test_resonant_normal_form_on_the_rational_graphic_has_no_resonances() -> None:
    # xi^2 (eigenvalue -2) and eta^2 (eigenvalue +1 for lambda1=-2,lambda2=1)
    # are both non-resonant and must clear exactly, leaving a trivial x^r
    # corner map with no log correction.
    target = named_rational_hyperbolic_graphic()
    diag = diagonalize_saddle(target.field, target.saddle)
    normal_form = resonant_normal_form(diag, order=3)
    assert normal_form.resonant_monomials == ()
    assert normal_form.p_star.is_zero()
    assert normal_form.q_star.is_zero()
    derived = dulac_corner_expansion(normal_form, Q(2), order=2)
    terms = derived.expansion.terms
    assert len(terms) == 1
    assert terms[0].exponent.lo == terms[0].exponent.hi == Q(2)
    assert terms[0].log_power == 0
    assert terms[0].coefficient.lo == terms[0].coefficient.hi == Q(1)


def test_dulac_corner_expansion_matches_the_hand_solvable_bernoulli_case() -> None:
    c = Q(1, 3)
    diag = _bernoulli_diagonal(c)
    normal_form = resonant_normal_form(diag, order=3)
    assert normal_form.resonant_monomials == ((0, (2, 1)),)
    derived = dulac_corner_expansion(normal_form, Q(1), order=2)
    coefficients = {
        (term.exponent.lo, term.log_power): term.coefficient.lo for term in derived.expansion.terms
    }
    # y = x/(1+c*x*log(x)) = x - c*x^2*log(x) + O((x*log(x))^2) exactly to
    # first order in the nonlinearity.
    assert coefficients[(Q(1), 0)] == Q(1)
    assert coefficients[(Q(2), 1)] == -c
    assert derived.residual_verified is True


def test_dulac_corner_expansion_claims_no_remainder_bound() -> None:
    c = Q(1, 3)
    diag = _bernoulli_diagonal(c)
    normal_form = resonant_normal_form(diag, order=3)
    derived = dulac_corner_expansion(normal_form, Q(1), order=2)
    assert derived.expansion.remainder_bound == Q(0)


def test_dulac_corner_expansion_rejects_bad_arguments() -> None:
    diag = _bernoulli_diagonal(Q(1, 3))
    normal_form = resonant_normal_form(diag, order=3)
    with pytest.raises(ValueError, match="positive integer"):
        dulac_corner_expansion(normal_form, Q(1), order=0)
    with pytest.raises(ValueError, match="positive Fraction"):
        dulac_corner_expansion(normal_form, Q(-1), order=2)
    with pytest.raises(ValueError, match="positive Fraction"):
        dulac_corner_expansion(normal_form, 1, order=2)  # type: ignore[arg-type]


def test_certify_and_verify_resonant_normal_form_round_trip() -> None:
    target = named_rational_hyperbolic_graphic()
    certificate = certify_resonant_normal_form(target.field, target.saddle, order=3)
    assert verify_resonant_normal_form(certificate)
    honesty = certificate.formal_seal["honesty"]
    assert honesty["resonant_monomials_derived_not_declared"] is True
    assert honesty["physical_return_membership_proved"] is False
    assert honesty["full_hilbert16_solved"] is False


def test_resonant_normal_form_certificate_tamper_rejected() -> None:
    target = named_rational_hyperbolic_graphic()
    certificate = certify_resonant_normal_form(target.field, target.saddle, order=3)
    assert not verify_resonant_normal_form(replace(certificate, source_digest="tampered"))
    altered = deepcopy(certificate.formal_seal)
    altered["payload"]["equations"] = []
    altered = seal_certificate(altered)
    assert not verify_resonant_normal_form(replace(certificate, formal_seal=altered))


def test_derive_return_map_scales_by_the_certified_regular_arc_jacobian() -> None:
    diag = _bernoulli_diagonal(Q(1, 3))
    normal_form = resonant_normal_form(diag, order=3)
    corner = dulac_corner_expansion(normal_form, Q(1), order=2)
    regular_arc = StoppedEventResult(
        request=None,  # type: ignore[arg-type]
        source_fingerprint="synthetic-regular-arc",
        status="certified",
        reason="synthetic fixture for composition test",
        slabs=(),
        return_jacobian=((Interval.point(2.0),),),
    )
    composed = derive_return_map(corner, regular_arc)
    for original, scaled in zip(corner.expansion.terms, composed.expansion.terms, strict=True):
        assert scaled.coefficient.lo == pytest.approx(2.0 * float(original.coefficient.lo))
        assert scaled.coefficient.hi == pytest.approx(2.0 * float(original.coefficient.hi))
        assert scaled.exponent == original.exponent
        assert scaled.log_power == original.log_power


def test_derive_return_map_rejects_an_uncertified_regular_arc() -> None:
    diag = _bernoulli_diagonal(Q(1, 3))
    normal_form = resonant_normal_form(diag, order=3)
    corner = dulac_corner_expansion(normal_form, Q(1), order=2)
    uncertified = StoppedEventResult(
        request=None,  # type: ignore[arg-type]
        source_fingerprint="synthetic-uncertified",
        status="excluded",
        reason="synthetic fixture",
        slabs=(),
    )
    with pytest.raises(ValueError, match="certified regular arc"):
        derive_return_map(corner, uncertified)
