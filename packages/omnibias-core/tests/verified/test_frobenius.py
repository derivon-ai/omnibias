# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Rigor tests for the certified Frobenius / Puiseux local series (`frobenius.py`).

The soundness contract is *containment*: every packaged series must enclose the
true analytic solution at every sample point inside its validated radius. Two
independent references anchor this:

* Part A -- Bessel's equation, whose indicial roots and coefficient formulas
  are classical closed forms (checked exactly with :class:`fractions.Fraction`
  for the ordinary equation, and cross-checked against ``mpmath.besseli`` /
  the repo's own :func:`besseli_point` for the modified equation).
* Part B -- ``y**2 = x**3`` (an exact algebraic identity, ``y = +-x**1.5``)
  and ``y - x*y - x = 0`` (``y = x/(1-x)``, a geometric series with a
  genuinely nonzero, slowly-decaying tail -- unlike the first example, whose
  higher Puiseux coefficients are all exactly zero).

Negative tests prove the "raise rather than silently approximate" doctrine:
the non-negative-integer root-gap resonance, complex indicial roots, Newton
polygon degeneracies (no edge, ambiguous edge, unsupported edge shape,
non-simple root), and -- the test that must be able to *fail* -- a
consecutive-ratio hypothesis that is analytically wrong, which is shown to
produce an enclosure a directly-sampled true value actually escapes.
"""

from __future__ import annotations

import math
import random
from fractions import Fraction

import pytest
from omnibias.core.verified.frobenius import (
    FrobeniusSolution,
    PuiseuxBranch,
    consecutive_ratio_tail_bound,
    frobenius_coefficients,
    indicial_roots,
    newton_polygon_leading_term,
    puiseux_coefficients,
    roots_gap_blocks_second_solution,
    solve_frobenius,
    solve_puiseux,
)
from omnibias.core.verified.interval import Interval
from omnibias.core.verified.transcend import besseli_point

mpmath = pytest.importorskip("mpmath")


def _true_besseli(nu: float, x: float) -> float:
    with mpmath.workdps(50):
        return float(mpmath.besseli(mpmath.mpf(nu), mpmath.mpf(x)))


# --------------------------------------------------------------------------- #
# Part A -- indicial roots
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("nu", [0.25, 1.0, 1.5, 2.0, 3.0])
def test_indicial_roots_are_plus_minus_nu_for_bessel(nu: float) -> None:
    """Bessel's equation has ``p0 = 1``, ``q0 = -nu**2``; roots are ``+-nu``."""
    r_plus, r_minus = indicial_roots(1.0, -nu * nu)
    assert r_plus.contains(nu)
    assert r_minus.contains(-nu)
    assert r_plus.width < 1e-9
    assert r_minus.width < 1e-9


def test_indicial_roots_handles_an_exactly_repeated_root() -> None:
    """Euler equation ``x**2 y'' - x y' + y = 0``: ``p0=-1``, ``q0=1`` -> ``r=1,1``.

    The discriminant computes to *exactly* zero mathematically; outward
    rounding of that exact-zero subtraction unavoidably produces a
    sign-straddling interval, which must not be mistaken for "complex roots".
    """
    r_plus, r_minus = indicial_roots(-1.0, 1.0)
    assert r_plus.contains(1.0)
    assert r_minus.contains(1.0)


def test_indicial_roots_rejects_a_certifiably_complex_pair() -> None:
    """``r**2 + 1 = 0`` (``p0=1``, ``q0=1``) has discriminant exactly ``-4``."""
    with pytest.raises(ValueError, match="complex-conjugate"):
        indicial_roots(1.0, 1.0)


def test_roots_gap_blocks_second_solution_detects_integer_gaps() -> None:
    rp, rm = indicial_roots(1.0, -1.0)  # Bessel nu=1: roots +-1, gap 2
    assert roots_gap_blocks_second_solution(rp, rm) is True
    rp2, rm2 = indicial_roots(1.0, -(1.0 / 9.0))  # nu=1/3: gap 2/3, not an integer
    assert roots_gap_blocks_second_solution(rp2, rm2) is False


# --------------------------------------------------------------------------- #
# Part A -- Frobenius recurrence coefficients: exact rational cross-check
# --------------------------------------------------------------------------- #
def _ordinary_bessel_coeffs_exact(nu: Fraction, count: int) -> list[Fraction]:
    r"""Exact ``a_n`` for ``J_nu(x)/(x/2)**nu = sum a_n x**n`` (odd ``n`` are 0).

    ``a_{2k} = (-1)**k / (4**k k! (nu+1)(nu+2)...(nu+k))``, ``a_0 = 1``.
    """
    out = [Fraction(0)] * count
    if count > 0:
        out[0] = Fraction(1)
    k = 1
    while 2 * k < count:
        denom = Fraction(4) ** k * math.factorial(k)
        prod = Fraction(1)
        for j in range(1, k + 1):
            prod *= nu + j
        out[2 * k] = (Fraction(-1) ** k) / (denom * prod)
        k += 1
    return out


@pytest.mark.parametrize("nu", [Fraction(1, 3), Fraction(1), Fraction(3, 2), Fraction(2)])
def test_frobenius_coefficients_match_ordinary_bessel_exactly(nu: Fraction) -> None:
    """``x**2 y'' + x y' + (x**2 - nu**2) y = 0``: ``p=[1]``, ``q=[-nu**2, 0, 1]``."""
    p = [1.0]
    q = [-float(nu) * float(nu), 0.0, 1.0]
    num_terms = 11
    coeffs = frobenius_coefficients(p, q, float(nu), num_terms=num_terms)
    exact = _ordinary_bessel_coeffs_exact(nu, num_terms)
    for n, (c, e) in enumerate(zip(coeffs, exact, strict=True)):
        assert c.contains(float(e)), f"a_{n}={c} does not contain exact value {e}"


# --------------------------------------------------------------------------- #
# Part A -- containment against the modified Bessel function I_nu
# --------------------------------------------------------------------------- #
#: y = x**nu * sum a_n x**n solves the *modified* Bessel equation for
#: I_nu(x) = y(x) / (2**nu * Gamma(nu+1)); num_terms=13 keeps through the
#: even (nonzero) index 12, avoiding the always-exactly-zero odd coefficients
#: as the *last* retained term (see module docstring on tail underflow).
_NUM_TERMS = 13
_RATIO = 0.1
_NU_WEIGHT = 1.0


def _modified_bessel_solution(nu: float) -> FrobeniusSolution:
    p = [1.0]
    q = [-nu * nu, 0.0, -1.0]
    return solve_frobenius(p, q, nu, num_terms=_NUM_TERMS, ratio=_RATIO, nu=_NU_WEIGHT)


@pytest.mark.parametrize("nu", [1.0, 2.0, 0.5, 1.5, 1.0 / 3.0])
@pytest.mark.parametrize("x", [0.1, 0.3, 0.7])
def test_solve_frobenius_encloses_modified_bessel(nu: float, x: float) -> None:
    sol = _modified_bessel_solution(nu)
    y = sol.evaluate(x)
    scale = (2.0**nu) * math.gamma(nu + 1.0)
    predicted = Interval(y.lo / scale, y.hi / scale)
    true_val = _true_besseli(nu, x)
    assert predicted.lo <= true_val <= predicted.hi
    assert predicted.width < 1e-9 * max(abs(true_val), 1.0)


@pytest.mark.parametrize("nu", [1, 2, 3])
@pytest.mark.parametrize("x", [0.2, 0.6, 0.95])
def test_solve_frobenius_agrees_with_besseli_point_for_integer_order(nu: int, x: float) -> None:
    """Independent cross-check against the repo's own verified ``besseli_point``."""
    sol = _modified_bessel_solution(float(nu))
    y = sol.evaluate(x)
    scale = (2.0**nu) * math.factorial(nu)
    predicted = Interval(y.lo / scale, y.hi / scale)
    reference = besseli_point(nu, x)
    assert predicted.lo <= reference.hi and reference.lo <= predicted.hi


def test_solve_frobenius_random_containment_against_besseli() -> None:
    """Random ``(nu, x)`` pairs inside the certified radius, dense not cherry-picked."""
    rng = random.Random(0x0F0B1A5)
    for _ in range(60):
        nu = rng.uniform(0.05, 3.0)
        x = rng.uniform(0.01, 0.95)
        sol = _modified_bessel_solution(nu)
        y = sol.evaluate(x)
        scale = (2.0**nu) * math.gamma(nu + 1.0)
        predicted = Interval(y.lo / scale, y.hi / scale)
        true_val = _true_besseli(nu, x)
        assert predicted.lo <= true_val <= predicted.hi, (
            f"nu={nu} x={x}: predicted {predicted} excludes true {true_val}"
        )


# --------------------------------------------------------------------------- #
# Part A -- negative tests: root-gap resonance and root/coefficient mismatches
# --------------------------------------------------------------------------- #
def test_frobenius_coefficients_rejects_smaller_root_past_the_resonance() -> None:
    """Euler equation with roots ``0, 2`` (``p0=-1``, ``q0=0``): gap ``m=2``."""
    p, q = [-1.0], [0.0]
    with pytest.raises(ValueError, match="non-negative integer 2"):
        frobenius_coefficients(p, q, 0.0, num_terms=5)


def test_frobenius_coefficients_allows_the_smaller_root_before_the_resonance() -> None:
    p, q = [-1.0], [0.0]
    coeffs = frobenius_coefficients(p, q, 0.0, num_terms=2)
    assert coeffs[0].contains(1.0)


def test_frobenius_coefficients_larger_root_of_a_gapped_pair_never_resonates() -> None:
    """The classical carve-out: only the *smaller* root of a gapped pair resonates."""
    p, q = [-1.0], [0.0]
    coeffs = frobenius_coefficients(p, q, 2.0, num_terms=10)
    assert len(coeffs) == 10
    assert coeffs[0].contains(1.0)


def test_frobenius_coefficients_repeated_root_never_resonates() -> None:
    p, q = [-1.0], [1.0]  # roots 1, 1
    coeffs = frobenius_coefficients(p, q, 1.0, num_terms=10)
    assert len(coeffs) == 10


def test_frobenius_coefficients_rejects_a_root_matching_neither_indicial_root() -> None:
    p, q = [1.0], [-1.0]  # roots +-1
    with pytest.raises(ValueError, match="does not match either certified indicial root"):
        frobenius_coefficients(p, q, 5.0, num_terms=3)


def test_frobenius_coefficients_rejects_num_terms_below_one() -> None:
    with pytest.raises(ValueError, match="num_terms"):
        frobenius_coefficients([1.0], [-1.0], 1.0, num_terms=0)


# --------------------------------------------------------------------------- #
# Part A -- negative test: a wrong ratio hypothesis must be catchable
# --------------------------------------------------------------------------- #
def test_solve_frobenius_wrong_ratio_hypothesis_produces_a_violated_bound() -> None:
    """A ratio hypothesis far tighter than the true asymptotics is unsound.

    This does not exercise a bug in the implementation -- ``ratio`` is
    explicitly a *trusted* input (see the module docstring). It instead
    proves the test methodology itself is meaningful: if containment checks
    elsewhere in this file could never fail, they would not be evidence of
    correctness. Few retained terms + an absurdly tiny ``ratio`` must escape.
    """
    nu = 1.0
    p, q = [1.0], [-nu * nu, 0.0, -1.0]
    sol = solve_frobenius(p, q, nu, num_terms=5, ratio=1e-8, nu=1.0)
    x = 0.9
    y = sol.evaluate(x)
    scale = (2.0**nu) * math.gamma(nu + 1.0)
    predicted = Interval(y.lo / scale, y.hi / scale)
    true_val = _true_besseli(nu, x)
    assert not (predicted.lo <= true_val <= predicted.hi), (
        "expected the deliberately-wrong ratio hypothesis to produce an "
        f"unsound enclosure, but {predicted} still contained {true_val}"
    )


# --------------------------------------------------------------------------- #
# Part A -- consecutive_ratio_tail_bound directly
# --------------------------------------------------------------------------- #
def test_consecutive_ratio_tail_bound_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError, match="ratio"):
        consecutive_ratio_tail_bound(1.0, 0.0, 1.0, 3)
    with pytest.raises(ValueError, match="n_trunc"):
        consecutive_ratio_tail_bound(1.0, 0.5, 1.0, -1)
    with pytest.raises(ValueError, match="weighted ratio"):
        consecutive_ratio_tail_bound(1.0, 2.0, 1.0, 3)  # nu*ratio = 2 >= 1


def test_consecutive_ratio_tail_bound_is_always_non_negative() -> None:
    """Regression: an exactly-zero last term must not underflow to a negative `lo`."""
    tail = consecutive_ratio_tail_bound(Interval.point(0.0), 0.05, 1.0, 13)
    assert tail.lo >= 0.0


# --------------------------------------------------------------------------- #
# Part B -- Newton-polygon leading term on y**2 = x**3
# --------------------------------------------------------------------------- #
_Y2_EQ_X3 = {(0, 2): 1.0, (3, 0): -1.0}


def test_newton_polygon_leading_term_recovers_the_known_branch_exponent() -> None:
    k0, roots = newton_polygon_leading_term(_Y2_EQ_X3, e=2)
    assert k0 == 3
    assert len(roots) == 2
    mags = sorted(r.mag for r in roots)
    assert mags[0] == pytest.approx(1.0, abs=1e-9)
    assert mags[1] == pytest.approx(1.0, abs=1e-9)
    assert any(r.lo > 0 for r in roots)
    assert any(r.hi < 0 for r in roots)


def test_puiseux_coefficients_y2_eq_x3_has_all_higher_terms_exactly_zero() -> None:
    k0, roots = newton_polygon_leading_term(_Y2_EQ_X3, e=2)
    for c0 in roots:
        d = puiseux_coefficients(_Y2_EQ_X3, e=2, k0=k0, c0=c0, num_terms=6)
        assert d[0].lo == c0.lo and d[0].hi == c0.hi  # d_0 == c0 by construction
        for j, dj in enumerate(d[1:], start=1):
            assert dj.contains(0.0), f"d_{j}={dj} should enclose exactly 0"


@pytest.mark.parametrize("x", [0.1, 0.5, 0.9, 2.0])
def test_solve_puiseux_encloses_x_to_the_three_halves(x: float) -> None:
    k0, roots = newton_polygon_leading_term(_Y2_EQ_X3, e=2)
    for c0, sign in zip(sorted(roots, key=lambda r: -r.mag if r.lo > 0 else r.mag), (1.0, -1.0), strict=True):
        branch = solve_puiseux(_Y2_EQ_X3, e=2, k0=k0, c0=c0, num_terms=5, ratio=0.1, nu=2.0)
        y = branch.evaluate_x(x)
        true_y = sign * x**1.5
        assert y.lo <= true_y <= y.hi, f"branch {sign}: {y} excludes {true_y} at x={x}"


def test_puiseux_random_containment_on_y2_eq_x3() -> None:
    k0, roots = newton_polygon_leading_term(_Y2_EQ_X3, e=2)
    rng = random.Random(0xC0FFEE)
    branches = [
        (c0, sign)
        for c0, sign in zip(
            sorted(roots, key=lambda r: -r.mag if r.lo > 0 else r.mag), (1.0, -1.0), strict=True
        )
    ]
    for _ in range(60):
        x = rng.uniform(0.001, 4.0)
        for c0, sign in branches:
            branch = solve_puiseux(_Y2_EQ_X3, e=2, k0=k0, c0=c0, num_terms=5, ratio=0.1, nu=2.0)
            y = branch.evaluate_x(x)
            true_y = sign * x**1.5
            assert y.lo <= true_y <= y.hi


# --------------------------------------------------------------------------- #
# Part B -- a curve with a genuinely nonzero tail: y - x*y - x = 0, y = x/(1-x)
# --------------------------------------------------------------------------- #
_GEOMETRIC_CURVE = {(0, 1): 1.0, (1, 1): -1.0, (1, 0): -1.0}


def test_newton_polygon_leading_term_recovers_the_geometric_branch() -> None:
    k0, roots = newton_polygon_leading_term(_GEOMETRIC_CURVE, e=1)
    assert k0 == 1
    assert len(roots) == 1
    assert roots[0].contains(1.0)


def test_puiseux_coefficients_geometric_branch_are_all_exactly_one() -> None:
    k0, roots = newton_polygon_leading_term(_GEOMETRIC_CURVE, e=1)
    d = puiseux_coefficients(_GEOMETRIC_CURVE, e=1, k0=k0, c0=roots[0], num_terms=8)
    for j, dj in enumerate(d):
        assert dj.contains(1.0), f"d_{j}={dj} should enclose the exact value 1"


def test_solve_puiseux_geometric_branch_random_containment() -> None:
    k0, roots = newton_polygon_leading_term(_GEOMETRIC_CURVE, e=1)
    branch = solve_puiseux(
        _GEOMETRIC_CURVE, e=1, k0=k0, c0=roots[0], num_terms=6, ratio=1.0, nu=0.6
    )
    rng = random.Random(0xFACE)
    for _ in range(60):
        x = rng.uniform(0.001, 0.59)
        y = branch.evaluate_x(x)
        true_y = x / (1.0 - x)
        assert y.lo <= true_y <= y.hi, f"x={x}: {y} excludes {true_y}"


def test_solve_puiseux_wrong_ratio_hypothesis_produces_a_violated_bound() -> None:
    """Mirrors the Part A negative test: an unjustifiably tight ``ratio`` for a
    branch whose true consecutive coefficient ratio is exactly ``1`` must be
    catchable, proving the containment tests above are not vacuous."""
    k0, roots = newton_polygon_leading_term(_GEOMETRIC_CURVE, e=1)
    branch = solve_puiseux(
        _GEOMETRIC_CURVE, e=1, k0=k0, c0=roots[0], num_terms=3, ratio=0.05, nu=1.0
    )
    x = 0.5
    y = branch.evaluate_x(x)
    true_y = x / (1.0 - x)
    assert not (y.lo <= true_y <= y.hi), (
        f"expected the wrong ratio hypothesis to escape containment, but {y} "
        f"still contained {true_y}"
    )


# --------------------------------------------------------------------------- #
# Part B -- negative tests: Newton-polygon degeneracies
# --------------------------------------------------------------------------- #
def test_newton_polygon_leading_term_rejects_non_positive_e() -> None:
    with pytest.raises(ValueError, match="ramification index"):
        newton_polygon_leading_term(_Y2_EQ_X3, e=0)


def test_newton_polygon_leading_term_rejects_a_single_monomial() -> None:
    with pytest.raises(ValueError, match=">= 2 nonzero monomials"):
        newton_polygon_leading_term({(3, 0): 1.0}, e=2)


def test_newton_polygon_leading_term_rejects_when_no_candidate_balances() -> None:
    """``x + y**2`` at ``e=1``: the only balancing ``k0`` is ``1/2``, not an integer."""
    with pytest.raises(ValueError, match="no candidate leading exponent"):
        newton_polygon_leading_term({(1, 0): 1.0, (0, 2): 1.0}, e=1)


def test_newton_polygon_leading_term_rejects_an_ambiguous_polygon() -> None:
    """``x**4 + x*y + y**2`` at ``e=1`` has two genuine Newton-polygon edges
    (``k0=1`` from the ``x*y``/``y**2`` pair, ``k0=3`` from the ``x**4``/``x*y``
    pair): automatic full-polygon disambiguation is out of scope."""
    with pytest.raises(ValueError, match="ambiguous Newton-polygon leading term"):
        newton_polygon_leading_term({(4, 0): 1.0, (1, 1): 1.0, (0, 2): 1.0}, e=1)


def test_newton_polygon_leading_term_rejects_an_unsupported_multi_power_edge() -> None:
    """``x**2 + x*y + y**2`` at ``e=1``: all three monomials tie at ``k0=1``,
    a three-``y``-power edge outside the linear/quadratic scope."""
    with pytest.raises(NotImplementedError, match="distinct y-powers"):
        newton_polygon_leading_term({(2, 0): 1.0, (1, 1): 1.0, (0, 2): 1.0}, e=1)


def test_newton_polygon_leading_term_rejects_an_unsupported_cubic_edge() -> None:
    """``y**3 - x = 0`` at ``e=3``: a genuine degree-3 edge, out of scope."""
    with pytest.raises(NotImplementedError, match="degree-3 solve"):
        newton_polygon_leading_term({(0, 3): 1.0, (1, 0): -1.0}, e=3)


def test_puiseux_coefficients_rejects_a_non_root_c0() -> None:
    with pytest.raises(ValueError, match="not certifiably a root"):
        puiseux_coefficients(_Y2_EQ_X3, e=2, k0=3, c0=5.0, num_terms=3)


def test_puiseux_coefficients_rejects_non_positive_e_or_negative_k0() -> None:
    with pytest.raises(ValueError, match="ramification index"):
        puiseux_coefficients(_Y2_EQ_X3, e=0, k0=3, c0=1.0, num_terms=3)
    with pytest.raises(ValueError, match="k0 must be >= 0"):
        puiseux_coefficients(_Y2_EQ_X3, e=2, k0=-1, c0=1.0, num_terms=3)


def test_puiseux_coefficients_rejects_num_terms_below_one() -> None:
    with pytest.raises(ValueError, match="num_terms"):
        puiseux_coefficients(_Y2_EQ_X3, e=2, k0=3, c0=1.0, num_terms=0)


# --------------------------------------------------------------------------- #
# evaluate()/evaluate_x() domain and radius checks
# --------------------------------------------------------------------------- #
def test_frobenius_solution_evaluate_requires_positive_x() -> None:
    sol = _modified_bessel_solution(1.0)
    with pytest.raises(ValueError, match="requires x > 0"):
        sol.evaluate(0.0)
    with pytest.raises(ValueError, match="requires x > 0"):
        sol.evaluate(-1.0)


def test_frobenius_solution_evaluate_rejects_x_past_the_validated_radius() -> None:
    sol = _modified_bessel_solution(1.0)  # nu weight = 1.0
    with pytest.raises(ValueError, match="exceeds the validated radius"):
        sol.evaluate(1.5)


def test_puiseux_branch_evaluate_x_requires_non_negative_x() -> None:
    k0, roots = newton_polygon_leading_term(_Y2_EQ_X3, e=2)
    branch = solve_puiseux(_Y2_EQ_X3, e=2, k0=k0, c0=roots[0], num_terms=4, ratio=0.1, nu=2.0)
    with pytest.raises(ValueError, match="requires x >= 0"):
        branch.evaluate_x(-1.0)


def test_puiseux_branch_evaluate_x_at_zero_is_the_branch_point() -> None:
    k0, roots = newton_polygon_leading_term(_Y2_EQ_X3, e=2)
    branch = solve_puiseux(_Y2_EQ_X3, e=2, k0=k0, c0=roots[0], num_terms=4, ratio=0.1, nu=2.0)
    assert branch.evaluate_x(0.0).contains(0.0)  # k0=3 > 0 -> y(0) = 0


def test_puiseux_branch_evaluate_t_matches_evaluate_x_via_x_eq_t_pow_e() -> None:
    k0, roots = newton_polygon_leading_term(_Y2_EQ_X3, e=2)
    branch = solve_puiseux(_Y2_EQ_X3, e=2, k0=k0, c0=roots[0], num_terms=5, ratio=0.1, nu=2.0)
    t = 0.6
    via_t = branch.evaluate_t(t)
    via_x = branch.evaluate_x(t**2)
    assert via_t.lo <= via_x.hi and via_x.lo <= via_t.hi


# --------------------------------------------------------------------------- #
# dataclass smoke: construction is inert without evaluate()
# --------------------------------------------------------------------------- #
def test_frobenius_solution_and_puiseux_branch_are_plain_dataclasses() -> None:
    sol = _modified_bessel_solution(1.0)
    assert isinstance(sol, FrobeniusSolution)
    assert sol.series.order == _NUM_TERMS - 1

    k0, roots = newton_polygon_leading_term(_Y2_EQ_X3, e=2)
    branch = solve_puiseux(_Y2_EQ_X3, e=2, k0=k0, c0=roots[0], num_terms=4, ratio=0.1, nu=2.0)
    assert isinstance(branch, PuiseuxBranch)
    assert branch.e == 2
    assert branch.k0 == k0
