# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Certified information theory: rigorous entropy / divergence enclosures."""

from __future__ import annotations

import math

import pytest
from omnibias.core.verified.information import (
    binned_distribution_enclosure,
    cross_entropy_enclosure,
    entropy_enclosure,
    js_divergence_enclosure,
    kl_divergence_enclosure,
    mutual_information_enclosure,
)
from omnibias.core.verified.interval import Interval
from omnibias.core.verified.transcend import softplus_iv


def _true_entropy(p: list[float]) -> float:
    return -sum(pi * math.log(pi) for pi in p if pi > 0.0)


def _true_kl(p: list[float], q: list[float]) -> float:
    return sum(pi * math.log(pi / qi) for pi, qi in zip(p, q, strict=True) if pi > 0.0)


def _true_mi(joint: list[list[float]]) -> float:
    m, n = len(joint), len(joint[0])
    px = [sum(joint[i]) for i in range(m)]
    py = [sum(joint[i][j] for i in range(m)) for j in range(n)]
    total = 0.0
    for i in range(m):
        for j in range(n):
            p = joint[i][j]
            if p > 0.0:
                total += p * math.log(p / (px[i] * py[j]))
    return total


# ----- softplus_iv ----------------------------------------------------------


@pytest.mark.parametrize("x", [-50.0, -3.0, -0.5, 0.0, 0.5, 3.0, 50.0, 700.0])
def test_softplus_iv_contains_truth_without_overflow(x: float) -> None:
    enc = softplus_iv(Interval.point(x))
    true = math.log1p(math.exp(-abs(x))) + max(x, 0.0)  # stable softplus
    assert enc.lo <= true <= enc.hi
    assert enc.width < 1e-9
    assert enc.lo >= 0.0


def test_softplus_iv_is_monotone_over_an_interval() -> None:
    enc = softplus_iv(Interval(-1.0, 2.0))
    assert enc.lo <= math.log1p(math.exp(-1.0)) <= enc.hi
    assert enc.lo <= math.log1p(math.exp(2.0)) <= enc.hi


# ----- entropy --------------------------------------------------------------


@pytest.mark.parametrize("k", [2, 3, 4, 8])
def test_entropy_uniform_is_log_k(k: int) -> None:
    enc = entropy_enclosure([1.0 / k] * k)
    assert enc.lo <= math.log(k) <= enc.hi
    assert enc.width < 1e-12


def test_entropy_contains_truth_and_is_tight() -> None:
    p = [0.1, 0.2, 0.3, 0.4]
    enc = entropy_enclosure(p)
    assert enc.lo <= _true_entropy(p) <= enc.hi
    assert enc.width < 1e-12


def test_entropy_handles_exact_zero_as_zero_term() -> None:
    # 0 ln 0 := 0, so a zero atom does not change the entropy (and does not error).
    enc = entropy_enclosure([0.5, 0.5, 0.0])
    assert enc.lo <= math.log(2.0) <= enc.hi


def test_entropy_of_point_mass_is_zero() -> None:
    enc = entropy_enclosure([1.0, 0.0, 0.0])
    assert enc.lo <= 0.0 <= enc.hi
    assert enc.hi < 1e-12


def test_entropy_accepts_interval_probabilities() -> None:
    # Band-mass-style interval inputs flow straight through.
    probs = [Interval(0.25, 0.25), Interval(0.25, 0.25), Interval(0.5, 0.5)]
    enc = entropy_enclosure(probs)
    true = _true_entropy([0.25, 0.25, 0.5])
    assert enc.lo <= true <= enc.hi


def test_entropy_rejects_out_of_range() -> None:
    with pytest.raises(ValueError, match="probability must lie in"):
        entropy_enclosure([1.2, -0.2])


# ----- cross entropy / KL ---------------------------------------------------


def test_cross_entropy_equals_entropy_plus_kl() -> None:
    p = [0.1, 0.2, 0.3, 0.4]
    q = [0.25, 0.25, 0.25, 0.25]
    ce = cross_entropy_enclosure(p, q)
    h = entropy_enclosure(p)
    kl = kl_divergence_enclosure(p, q)
    # H(p,q) = H(p) + D(p||q): the CE enclosure must contain H.lo + KL.lo .. H.hi+KL.hi
    assert ce.lo <= h.hi + kl.hi
    assert ce.hi >= h.lo + kl.lo
    assert ce.lo <= (_true_entropy(p) + _true_kl(p, q)) <= ce.hi


def test_kl_is_zero_for_identical_distributions() -> None:
    p = [0.1, 0.2, 0.3, 0.4]
    enc = kl_divergence_enclosure(p, p)
    assert enc.lo == 0.0  # clamped to the proven sign
    assert enc.hi < 1e-12


def test_kl_contains_truth_and_is_nonnegative() -> None:
    p = [0.7, 0.2, 0.1]
    q = [0.2, 0.3, 0.5]
    enc = kl_divergence_enclosure(p, q)
    assert enc.lo >= 0.0
    assert enc.lo <= _true_kl(p, q) <= enc.hi
    assert enc.width < 1e-12


def test_kl_handles_zero_p_atom() -> None:
    p = [0.0, 0.5, 0.5]
    q = [0.2, 0.3, 0.5]
    enc = kl_divergence_enclosure(p, q)
    assert enc.lo <= _true_kl(p, q) <= enc.hi


def test_kl_rejects_q_zero_where_p_positive() -> None:
    with pytest.raises(ValueError, match="q_i > 0"):
        kl_divergence_enclosure([0.5, 0.5], [0.0, 1.0])


def test_kl_rejects_p_straddling_zero() -> None:
    with pytest.raises(ValueError, match="positive lower bound"):
        kl_divergence_enclosure([Interval(0.0, 0.4), Interval(0.6, 0.6)], [0.5, 0.5])


# ----- Jensen-Shannon -------------------------------------------------------


def test_js_is_symmetric() -> None:
    p = [0.7, 0.2, 0.1]
    q = [0.2, 0.3, 0.5]
    a = js_divergence_enclosure(p, q)
    b = js_divergence_enclosure(q, p)
    assert a.lo <= b.hi and b.lo <= a.hi  # overlapping enclosures


def test_js_is_bounded_by_ln2() -> None:
    p = [1.0, 0.0]
    q = [0.0, 1.0]  # maximally separated -> JS = ln 2
    enc = js_divergence_enclosure(p, q)
    assert 0.0 <= enc.lo <= enc.hi <= math.nextafter(math.log(2.0), math.inf)
    assert enc.lo <= math.log(2.0) <= enc.hi


# ----- mutual information ----------------------------------------------------


def test_mi_zero_for_independent_joint() -> None:
    px = [0.3, 0.7]
    py = [0.2, 0.3, 0.5]
    joint = [[pi * pj for pj in py] for pi in px]
    enc = mutual_information_enclosure(joint)
    assert enc.lo == 0.0  # clamped to the proven sign (I >= 0)
    assert enc.hi < 1e-12


def test_mi_of_perfectly_correlated_equals_marginal_entropy() -> None:
    # A diagonal joint makes Y a deterministic function of X: I = H(marginal).
    p = [0.25, 0.25, 0.5]
    joint = [[p[i] if i == j else 0.0 for j in range(3)] for i in range(3)]
    enc = mutual_information_enclosure(joint)
    h = entropy_enclosure(p)
    assert enc.lo <= _true_entropy(p) <= enc.hi
    assert enc.lo <= h.hi and h.lo <= enc.hi  # overlapping enclosures


def test_mi_contains_truth_and_is_nonnegative() -> None:
    joint = [[0.1, 0.2, 0.05], [0.05, 0.3, 0.3]]  # sums to 1, dependent
    enc = mutual_information_enclosure(joint)
    assert enc.lo >= 0.0
    assert enc.lo <= _true_mi(joint) <= enc.hi
    assert enc.width < 1e-12


def test_mi_is_symmetric_under_transpose() -> None:
    joint = [[0.1, 0.2, 0.05], [0.05, 0.3, 0.3]]
    transposed = [[joint[i][j] for i in range(len(joint))] for j in range(len(joint[0]))]
    a = mutual_information_enclosure(joint)
    b = mutual_information_enclosure(transposed)
    assert a.lo <= b.hi and b.lo <= a.hi  # overlapping enclosures


def test_mi_equals_hx_plus_hy_minus_hxy() -> None:
    # I(X;Y) = H(X) + H(Y) - H(X,Y); check the certified enclosures are consistent.
    joint = [[0.1, 0.2, 0.05], [0.05, 0.3, 0.3]]
    m, n = len(joint), len(joint[0])
    px = [sum(joint[i]) for i in range(m)]
    py = [sum(joint[i][j] for i in range(m)) for j in range(n)]
    flat = [joint[i][j] for i in range(m) for j in range(n)]
    mi = mutual_information_enclosure(joint)
    rhs_lo = entropy_enclosure(px).lo + entropy_enclosure(py).lo - entropy_enclosure(flat).hi
    rhs_hi = entropy_enclosure(px).hi + entropy_enclosure(py).hi - entropy_enclosure(flat).lo
    assert mi.lo <= rhs_hi and rhs_lo <= mi.hi  # overlapping enclosures


def test_mi_accepts_interval_cells() -> None:
    joint = [[Interval(0.1, 0.1), Interval(0.4, 0.4)], [Interval(0.3, 0.3), Interval(0.2, 0.2)]]
    enc = mutual_information_enclosure(joint)
    truth = _true_mi([[0.1, 0.4], [0.3, 0.2]])
    assert enc.lo <= truth <= enc.hi


def test_mi_rejects_ragged_table() -> None:
    with pytest.raises(ValueError, match="rectangular"):
        mutual_information_enclosure([[0.5, 0.5], [1.0]])


def test_mi_rejects_empty_table() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        mutual_information_enclosure([])


# ----- binned model distribution bridge -------------------------------------


def test_binned_distribution_masses_contain_truth() -> None:
    edges = [-3.0, -1.0, 0.0, 1.0, 3.0]
    masses = binned_distribution_enclosure("sigmoid", edges, loc=0.0, scale=1.0)
    assert len(masses) == len(edges) - 1
    for i, m in enumerate(masses):
        lo = 1.0 / (1.0 + math.exp(-edges[i]))
        hi = 1.0 / (1.0 + math.exp(-edges[i + 1]))
        assert m.lo <= (hi - lo) <= m.hi


def test_entropy_of_binned_model_is_certified() -> None:
    edges = [-4.0, -1.0, 0.0, 1.0, 4.0]
    masses = binned_distribution_enclosure("tanh", edges, loc=0.2, scale=1.3)

    def cdf(x: float) -> float:
        return 0.5 * math.tanh((x - 0.2) / 1.3) + 0.5

    true = [cdf(edges[i + 1]) - cdf(edges[i]) for i in range(len(edges) - 1)]
    enc = entropy_enclosure(masses)
    assert enc.lo <= _true_entropy(true) <= enc.hi
