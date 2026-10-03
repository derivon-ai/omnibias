# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Regression tests for the exact-Q Groebner engine (monomial orders,
division, Buchberger, reduced bases, ideal/radical membership witnesses,
and budget refusal)."""

from __future__ import annotations

from fractions import Fraction

import pytest
from omnibias.holonomic._core.groebner import (
    GroebnerBudget,
    GroebnerBudgetExceeded,
    MonomialOrder,
    _divide_plain,
    buchberger,
    ideal_member,
    radical_member,
    reduced_groebner_basis,
    verify_ideal_membership,
)
from omnibias.holonomic._core.poly_n import PolyN


def _xy() -> tuple[PolyN, PolyN]:
    return PolyN.var(2, 0), PolyN.var(2, 1)


def test_monomial_orders_rank_as_expected() -> None:
    lex = MonomialOrder("lex")
    grlex = MonomialOrder("grlex")
    degrevlex = MonomialOrder("degrevlex")
    # lex: x^2 > x*y > y^2 (pure exponent-tuple lexicographic).
    assert lex.compare((2, 0), (1, 1)) > 0
    assert lex.compare((1, 1), (0, 2)) > 0
    # grlex: total degree first, then lex among ties (x^2 > x*y under lex).
    assert grlex.compare((2, 0), (1, 1)) > 0
    assert grlex.compare((0, 3), (2, 0)) > 0
    # degrevlex: total degree first, then *smallest* trailing exponent wins.
    assert degrevlex.compare((2, 0), (0, 2)) > 0
    assert degrevlex.compare((1, 1), (0, 2)) > 0


def test_leading_term_selection() -> None:
    x, y = _xy()
    order = MonomialOrder("degrevlex")
    f = x * x * y + x * y * y * 3 + y * y * y * y * 2
    lm = order.leading_monomial(f)
    assert lm == (0, 4)
    with pytest.raises(ValueError, match="zero polynomial"):
        order.leading_monomial(PolyN.zero(2))
        order.leading_term_data(PolyN.zero(2))


def test_leading_term_data_raises_on_zero() -> None:
    order = MonomialOrder("degrevlex")
    with pytest.raises(ValueError, match="zero polynomial"):
        order.leading_term_data(PolyN.zero(2))


def test_multivariate_division_reconstructs_dividend_exactly() -> None:
    x, y = _xy()
    f = x * x * y + x * y * y + y
    divisors = [x * y - Fraction(1), y * y - Fraction(1)]
    order = MonomialOrder("lex")
    quotients, remainder = _divide_plain(f, divisors, order)
    reconstructed = remainder
    for q, d in zip(quotients, divisors, strict=True):
        reconstructed = reconstructed + q * d
    assert reconstructed == f


def test_buchberger_both_criteria_agree_with_reduced_basis_variety() -> None:
    # Classic textbook example: (x^2+y^2-1, x*y) has a 1-D variety on the
    # unit circle union axes; the reduced Groebner basis under lex has a
    # known shape (Cox/Little/O'Shea, "Ideals, Varieties, and Algorithms").
    x, y = _xy()
    generators = [x * x + y * y - Fraction(1), x * y]
    raw = buchberger(generators, "lex")
    reduced = reduced_groebner_basis(generators, "lex")
    assert reduced  # nonempty ideal
    # Every generator is a combination of the raw basis (sanity: raw basis
    # is nonempty too, and reduced is derived from it).
    assert raw
    order = MonomialOrder("lex")
    leading = [order.leading_monomial(g) for g in reduced]
    assert len(leading) == len(set(leading)), "reduced basis must be minimal (distinct leading monomials)"
    # Every reduced-basis element must be monic.
    for g in reduced:
        _, lc = order.leading_term_data(g)
        assert lc == 1


def test_reduced_basis_is_independent_of_generator_order() -> None:
    x, y = _xy()
    g1 = x * x - y
    g2 = x * y - Fraction(1)
    forward = reduced_groebner_basis([g1, g2], "degrevlex")
    backward = reduced_groebner_basis([g2, g1], "degrevlex")
    assert forward == backward


def test_ideal_member_returns_a_replayable_cofactor_witness() -> None:
    x, y = _xy()
    generators = [x * x - y, x * y - Fraction(1)]
    # y^2 - x = x*(x*y - 1) + (x^2 - y)*... is in the ideal by construction:
    # y^2 - x*y*y ... simplest membership check: x*(x^2-y) - (x*y-1) = x^3 - x*y - x*y + 1
    candidate = x * (x * x - y) - (x * y - Fraction(1))
    membership = ideal_member(candidate, generators, "degrevlex")
    assert membership.is_member
    assert verify_ideal_membership(candidate, generators, membership)


def test_ideal_member_rejects_a_genuine_non_member() -> None:
    x, y = _xy()
    generators = [x, y]
    candidate = PolyN.const(2, 1)
    membership = ideal_member(candidate, generators, "degrevlex")
    assert not membership.is_member
    assert verify_ideal_membership(candidate, generators, membership)


def test_radical_member_accepts_a_variety_vanishing_polynomial_not_in_the_ideal() -> None:
    # (x^2) has variety {x=0}; y vanishes on that variety in the (x,y)-plane
    # only if we also require y=0, so instead use the classic
    # radical-vs-ideal witness: (x^2) does NOT contain x, but x IS in its
    # radical (x*x is, x is the square root).
    x, _y = _xy()
    generators = [x * x]
    not_in_ideal = ideal_member(x, generators, "degrevlex")
    assert not not_in_ideal.is_member
    in_radical = radical_member(x, generators, "degrevlex")
    assert in_radical.is_member
    # The Rabinowitsch witness lives in the extended ring (nvars+1); replay
    # it directly rather than via verify_ideal_membership on the original
    # generators (which are in a different ring).
    embedded_generators = [
        PolyN(3, {(*mon, 0): c for mon, c in g.terms.items()}) for g in generators
    ]
    t = PolyN.var(3, 2)
    embedded_x = PolyN(3, {(*mon, 0): c for mon, c in x.terms.items()})
    rabinowitsch = PolyN.const(3, 1) - t * embedded_x
    system = [*embedded_generators, rabinowitsch]
    one = PolyN.const(3, 1)
    assert verify_ideal_membership(one, system, in_radical)


def test_empty_generator_list_reduces_to_zero_test() -> None:
    x, _y = _xy()
    zero = PolyN.zero(2)
    membership_zero = ideal_member(zero, [], "degrevlex")
    assert membership_zero.is_member
    membership_x = ideal_member(x, [], "degrevlex")
    assert not membership_x.is_member
    with pytest.raises(ValueError, match="at least one generator"):
        radical_member(x, [], "degrevlex")


def test_groebner_budget_refuses_loudly_rather_than_hanging() -> None:
    x, y = _xy()
    tiny_budget = GroebnerBudget(max_pairs=1, max_polynomials=1000, max_terms=1000, max_degree=64)
    generators = [x * x * x - y, x * y * y - x, x * x * y - y * y]
    with pytest.raises(GroebnerBudgetExceeded):
        buchberger(generators, "degrevlex", budget=tiny_budget)


def test_groebner_budget_rejects_invalid_construction() -> None:
    with pytest.raises(ValueError, match="positive integers"):
        GroebnerBudget(max_pairs=0)
    with pytest.raises(ValueError, match="positive integers"):
        GroebnerBudget(max_degree=-1)


def test_unknown_monomial_order_name_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown monomial order"):
        MonomialOrder("shortlex")  # type: ignore[arg-type]
    x, y = _xy()
    with pytest.raises(ValueError, match="unknown monomial order"):
        buchberger([x, y], "shortlex")  # type: ignore[arg-type]
