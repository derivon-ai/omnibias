# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Creative telescoping: annihilating recurrences of parametrised sums."""

from __future__ import annotations

from fractions import Fraction
from math import comb

import pytest
from omnibias.holonomic._core.hyperterm import ProperTerm, binomial_nk, geometric_k
from omnibias.holonomic._core.zeilberger import (
    creative_telescoping,
    summand_sum,
    wz_certificate,
    wz_pair,
    zeilberger,
)


def test_summand_sum_is_exact() -> None:
    # sum_{k=0}^{n} C(n, k) = 2^n.
    for n in range(8):
        assert summand_sum(lambda nn, k: comb(nn, k), n, lambda nn: (0, nn)) == 2**n


def test_central_binomial_recurrence() -> None:
    # sum_k C(n, k)^2 = C(2n, n): first-order (n + 1) f(n+1) - (4n + 2) f(n) = 0.
    tele = creative_telescoping(lambda n, k: comb(n, k) ** 2, name="sumCnk2", n_max=14)
    assert tele.order == 1
    assert tele.max_residual() == 0
    # verify the values are the central binomials.
    assert [int(v) for v in tele.values[:6]] == [comb(2 * n, n) for n in range(6)]


def test_franel_recurrence_order_two() -> None:
    # Franel numbers sum_k C(n, k)^3 satisfy an order-2 P-recurrence (no elementary form).
    tele = creative_telescoping(lambda n, k: comb(n, k) ** 3, name="franel", n_max=16)
    assert tele.order == 2
    assert tele.max_residual() == 0


def test_recurrence_actually_annihilates_the_sum() -> None:
    tele = creative_telescoping(lambda n, k: k * comb(n, k), name="kCnk", n_max=14)
    for n in range(len(tele.values) - tele.order):
        assert tele.residual(n) == 0


def test_no_recurrence_raises() -> None:
    # The partition-like sequence C(n, floor(n/2)) is P-recursive; use a genuinely
    # non-P-recursive summand instead: sum over a single Bell-number-indexed term.
    bell = [1, 1, 2, 5, 15, 52, 203, 877, 4140, 21147, 115975, 678570, 4213597]
    with pytest.raises(ValueError, match="no P-recursive annihilator"):
        creative_telescoping(
            lambda n, k: Fraction(bell[n]) if k == 0 else Fraction(0),
            name="bell",
            n_max=11,
            max_order=4,
            max_index_degree=3,
        )


# --------------------------------------------------------------------------- #
# True Zeilberger + WZ (exact, all-n certificates).
# --------------------------------------------------------------------------- #
def test_zeilberger_binomial_row_sum() -> None:
    # sum_k C(n, k) = 2^n: order-1 telescoper, exact certificate, all n.
    cert = zeilberger(binomial_nk())
    assert cert.order == 1
    assert cert.verify_symbolic()
    for n in range(10):
        assert cert.annihilation_residual(n, 0, n + 2) == 0


def test_zeilberger_central_binomial() -> None:
    # sum_k C(n, k)^2 = C(2n, n): (n+1) f(n+1) = (4n+2) f(n).
    cert = zeilberger(binomial_nk().power(2))
    assert cert.order == 1
    assert cert.verify_symbolic()
    for n in range(10):
        assert cert.annihilation_residual(n, 0, n + 2) == 0


def test_zeilberger_certificate_is_exact_rational_identity() -> None:
    cert = zeilberger(binomial_nk())
    # The term-level telescoping holds at every pole-free integer point.
    checked = 0
    for n in range(6):
        for k in range(1, 6):
            try:
                assert cert.relation_residual(n, k) == 0
                checked += 1
            except ValueError:
                pass  # certificate pole -- skip
    assert checked > 0


def test_zeilberger_degenerate_alternating_sum() -> None:
    # sum_k (-1)^k C(n, k) = [n == 0]: the guesser rejects it; Zeilberger handles it.
    term = geometric_k(-1).times(binomial_nk())
    cert = zeilberger(term)
    assert cert.verify_symbolic()
    # f(n) = [n == 0]; L annihilates it for all n.
    for n in range(10):
        assert cert.annihilation_residual(n, 0, n + 2) == 0
    assert term.sum_over_k(0, 0, 2) == 1
    assert all(term.sum_over_k(n, 0, n + 2) == 0 for n in range(1, 10))


def test_zeilberger_alternating_geometric_weight() -> None:
    # sum_k (-1)^k C(n, k) 2^k = sum_k C(n, k) (-2)^k = (1 - 2)^n = (-1)^n.
    term = geometric_k(-2).times(binomial_nk())
    cert = zeilberger(term)
    assert cert.verify_symbolic()
    for n in range(10):
        assert cert.annihilation_residual(n, 0, n + 2) == 0
    assert all(term.sum_over_k(n, 0, n) == (-1) ** n for n in range(10))


def test_zeilberger_degenerate_eventually_zero_sum() -> None:
    # sum_k (-1)^k C(n, k) (k + 1) = 1, -1, 0, 0, 0, ... (finite support, guesser rejects).
    weight = ProperTerm(((0, 1, 1, 1), (0, 1, 0, -1)))  # (k+1)!/k! = k + 1
    term = geometric_k(-1).times(binomial_nk()).times(weight)
    expected = [1, -1, 0, 0, 0, 0, 0, 0]
    assert [term.sum_over_k(n, 0, n) for n in range(8)] == expected
    cert = zeilberger(term)
    assert cert.verify_symbolic()
    for n in range(10):
        assert cert.annihilation_residual(n, 0, n + 2) == 0


def test_wz_certificate_normalised_binomial() -> None:
    # sum_k C(n, k) / 2^n = 1 (constant): WZ pair exists.
    term = ProperTerm(binomial_nk().factors, geom_n=Fraction(1, 2))
    g, cert = wz_pair(term)
    # F(n+1, k) - F(n, k) = G(n, k+1) - G(n, k) at pole-free points.
    checked = 0
    for n in range(5):
        for k in range(1, 5):
            try:
                lhs = term.value(n + 1, k) - term.value(n, k)
                rhs = g(n, k + 1) - g(n, k)
                assert lhs == rhs
                checked += 1
            except ValueError:
                pass
    assert checked > 0
    assert wz_certificate(term) == cert


def test_zeilberger_unsummable_raises() -> None:
    # A term with no low-order telescoper within the bounds is a genuine finding.
    with pytest.raises(ValueError, match="no Zeilberger certificate"):
        zeilberger(binomial_nk().power(3), max_order=1, max_cert_degree=1)
