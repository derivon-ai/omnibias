# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Certified generalized divergences + Wasserstein-2 enclosures.

Every functional is returned as a guaranteed :class:`Interval`; the tests pin
that the enclosure brackets an independent hand-computed truth, respects the
proven sign / range clamps, and rejects ill-posed inputs.
"""

from __future__ import annotations

import math
from fractions import Fraction

import pytest
from omnibias.core.verified.information import (
    chi_squared_enclosure,
    hellinger_enclosure,
    total_variation_enclosure,
)
from omnibias.core.verified.transport import (
    certified_wasserstein2_gaussian,
    certified_wasserstein2_samples,
)


def _tv(p: list[float], q: list[float]) -> float:
    return 0.5 * sum(abs(pi - qi) for pi, qi in zip(p, q, strict=True))


def _hellinger(p: list[float], q: list[float]) -> float:
    return math.sqrt(0.5 * sum((math.sqrt(pi) - math.sqrt(qi)) ** 2 for pi, qi in zip(p, q, strict=True)))


def _chi2(p: list[float], q: list[float]) -> float:
    return sum((pi - qi) ** 2 / qi for pi, qi in zip(p, q, strict=True))


_P = [0.5, 0.3, 0.2]
_Q = [0.2, 0.5, 0.3]


# ----- total variation ------------------------------------------------------


def test_tv_encloses_truth() -> None:
    enc = total_variation_enclosure(_P, _Q)
    assert enc.lo <= _tv(_P, _Q) <= enc.hi


def test_tv_self_is_zero_and_symmetric() -> None:
    z = total_variation_enclosure(_P, _P)
    assert z.lo <= 0.0 <= z.hi + 1e-15
    a = total_variation_enclosure(_P, _Q)
    b = total_variation_enclosure(_Q, _P)
    assert a.lo == pytest.approx(b.lo) and a.hi == pytest.approx(b.hi)


def test_tv_is_bounded_in_unit_interval() -> None:
    enc = total_variation_enclosure([1.0, 0.0], [0.0, 1.0])
    assert 0.0 <= enc.lo <= enc.hi <= 1.0
    assert enc.lo <= 1.0 <= enc.hi


# ----- Hellinger ------------------------------------------------------------


def test_hellinger_encloses_truth() -> None:
    enc = hellinger_enclosure(_P, _Q)
    assert enc.lo <= _hellinger(_P, _Q) <= enc.hi


def test_hellinger_self_zero_and_max() -> None:
    z = hellinger_enclosure(_P, _P)
    assert z.lo <= 0.0 <= z.hi + 1e-12
    disjoint = hellinger_enclosure([1.0, 0.0], [0.0, 1.0])
    assert disjoint.lo <= 1.0 <= disjoint.hi


def test_hellinger_uses_exact_rational_input() -> None:
    p = [Fraction(1, 2), Fraction(1, 2)]
    q = [Fraction(1, 4), Fraction(3, 4)]
    enc = hellinger_enclosure(p, q)
    truth = math.sqrt(0.5 * ((0.5**0.5 - 0.25**0.5) ** 2 + (0.5**0.5 - 0.75**0.5) ** 2))
    assert enc.lo <= truth <= enc.hi


# ----- chi-squared ----------------------------------------------------------


def test_chi_squared_encloses_truth() -> None:
    enc = chi_squared_enclosure(_P, _Q)
    assert enc.lo <= _chi2(_P, _Q) <= enc.hi


def test_chi_squared_self_is_zero() -> None:
    z = chi_squared_enclosure(_P, _P)
    assert z.lo <= 0.0 <= z.hi + 1e-12


def test_chi_squared_rejects_zero_denominator() -> None:
    with pytest.raises(ValueError, match="q_i > 0"):
        chi_squared_enclosure([0.5, 0.5], [1.0, 0.0])


# ----- length / empties -----------------------------------------------------


@pytest.mark.parametrize(
    "fn", [total_variation_enclosure, hellinger_enclosure, chi_squared_enclosure]
)
def test_divergences_reject_length_mismatch(fn) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(ValueError, match="length mismatch"):
        fn([0.5, 0.5], [1.0])


# ----- Wasserstein-2 (samples) ----------------------------------------------


def test_w2_samples_encloses_rms_of_sorted_diffs() -> None:
    u = [0.0, 1.0, 2.0, 3.0]
    v = [0.5, 1.5, 2.5, 3.5]
    enc = certified_wasserstein2_samples(u, v)
    assert enc.lo <= 0.5 <= enc.hi  # constant shift -> W2 == shift


def test_w2_samples_handles_unsorted_input() -> None:
    enc = certified_wasserstein2_samples([2.0, 0.0, 1.0], [1.0, 3.0, 2.0])
    truth = math.sqrt(sum((a - b) ** 2 for a, b in zip([0, 1, 2], [1, 2, 3], strict=True)) / 3)
    assert enc.lo <= truth <= enc.hi


def test_w2_samples_rejects_mismatch_and_empty() -> None:
    with pytest.raises(ValueError, match="equal-length"):
        certified_wasserstein2_samples([1.0, 2.0], [1.0])
    with pytest.raises(ValueError, match="at least one"):
        certified_wasserstein2_samples([], [])


# ----- Wasserstein-2 (Gaussian) ---------------------------------------------


def test_w2_gaussian_closed_form() -> None:
    enc = certified_wasserstein2_gaussian(0.0, 1.0, 3.0, 2.0)
    assert enc.lo <= math.sqrt(9.0 + 1.0) <= enc.hi


def test_w2_gaussian_same_distribution_is_zero() -> None:
    enc = certified_wasserstein2_gaussian(1.5, 0.7, 1.5, 0.7)
    assert enc.lo <= 0.0 <= enc.hi + 1e-15


def test_w2_gaussian_rejects_negative_sigma() -> None:
    with pytest.raises(ValueError, match=">= 0"):
        certified_wasserstein2_gaussian(0.0, -1.0, 0.0, 1.0)
