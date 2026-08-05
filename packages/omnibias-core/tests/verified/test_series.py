# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Certified series summation (the verified Sigma operator).

Rigor checks:

* the geometric closed form ``a/(1-r)`` encloses the analytic limit;
* a truncated certified sum (retained terms + geometric tail) encloses the exact
  infinite sum, for both positive and sign-indefinite series;
* the tail enclosure is a true upper bound on the omitted remainder and shrinks
  as more terms are retained;
* convergence / domain guards reject ``q >= 1`` and ``|r| >= 1``.
"""

from __future__ import annotations

import pytest
from omnibias.core.verified import (
    certified_geometric_series_sum,
    geometric_series_closed_form,
    geometric_tail_enclosure,
)
from omnibias.core.verified.interval import Interval


def test_geometric_closed_form_encloses_limit() -> None:
    # sum 0.5^n = 2
    enc = geometric_series_closed_form(1.0, 0.5)
    assert enc.lo <= 2.0 <= enc.hi
    assert enc.width < 1e-12
    # sum (-0.3)^n = 1/1.3
    enc2 = geometric_series_closed_form(1.0, -0.3)
    assert enc2.lo <= 1.0 / 1.3 <= enc2.hi


def test_certified_sum_encloses_positive_series() -> None:
    terms = [Interval.point(0.5**n) for n in range(10)]
    enc = certified_geometric_series_sum(terms, 0.5)
    assert enc.lo <= 2.0 <= enc.hi
    # retained 10 terms + tail; still a genuine (finite-width) bracket
    assert enc.width < 1e-2


def test_certified_sum_encloses_sign_indefinite_series() -> None:
    # a_n = (-0.3)^n ; |a_{n+1}/a_n| = 0.3
    terms = [Interval.point((-0.3) ** n) for n in range(7)]
    enc = certified_geometric_series_sum(terms, 0.3)
    assert enc.lo <= 1.0 / 1.3 <= enc.hi


def test_certified_sum_brackets_closed_form() -> None:
    terms = [Interval.point(0.25 * 0.4**n) for n in range(6)]
    truncated = certified_geometric_series_sum(terms, 0.4)
    exact = geometric_series_closed_form(0.25, 0.4)
    # the truncated certified bracket must contain the analytic value
    assert truncated.lo <= exact.mid <= truncated.hi


def test_tail_enclosure_bounds_true_remainder() -> None:
    # retain a_0..a_9 of 0.5^n; true omitted tail = sum_{n>=10} 0.5^n = 0.5^9
    tail = geometric_tail_enclosure(Interval.point(0.5**9), 0.5)
    true_tail = 0.5**9
    assert tail.lo <= true_tail <= tail.hi
    assert tail.hi >= true_tail  # genuine upper bound on the magnitude


def test_tail_shrinks_with_more_retained_terms() -> None:
    t5 = geometric_tail_enclosure(Interval.point(0.5**5), 0.5).hi
    t10 = geometric_tail_enclosure(Interval.point(0.5**10), 0.5).hi
    t20 = geometric_tail_enclosure(Interval.point(0.5**20), 0.5).hi
    assert t5 > t10 > t20 > 0.0


def test_zero_last_term_gives_zero_tail() -> None:
    tail = geometric_tail_enclosure(Interval.point(0.0), 0.7)
    assert tail.lo <= 0.0 <= tail.hi
    assert tail.hi < 1e-300  # a zero last term leaves an essentially-zero tail


def test_convergence_and_domain_guards() -> None:
    with pytest.raises(ValueError):
        geometric_tail_enclosure(1.0, 1.0)  # q >= 1 diverges
    with pytest.raises(ValueError):
        geometric_tail_enclosure(1.0, -0.1)  # magnitude ratio must be >= 0
    with pytest.raises(ValueError):
        geometric_series_closed_form(1.0, 1.0)  # |r| >= 1
    with pytest.raises(ValueError):
        geometric_series_closed_form(1.0, -1.0)  # |r| >= 1
    with pytest.raises(ValueError):
        certified_geometric_series_sum([], 0.5)  # need >= 1 retained term
