# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Certified optimal transport: rigorous 1-D Wasserstein-1 enclosure."""

from __future__ import annotations

import math

import pytest
from omnibias.core.verified.transport import (
    certified_wasserstein1,
    certified_wasserstein1_samples,
)


def _two_sample_w1(u: list[float], v: list[float]) -> float:
    """Closed-form 1-D two-sample W1: mean |order statistics| difference."""
    us, vs = sorted(u), sorted(v)
    return sum(abs(a - b) for a, b in zip(us, vs, strict=True)) / len(us)


def _logistic_quantiles(n: int, loc: float = 0.0, scale: float = 1.0) -> list[float]:
    """Deterministic logistic samples (= its own quantiles): no RNG, no flakiness."""
    return [loc + scale * math.log(u / (1.0 - u)) for u in ((i + 0.5) / n for i in range(n))]


def _w1_numeric(
    name: str, xs: list[float], loc: float, scale: float, *, half_width: float = 80.0
) -> float:
    """Fine-grid numeric oracle for ``int |F - F_n| dx`` (tails are exponential)."""
    n_grid = 400_001
    a = loc - half_width * scale
    b = loc + half_width * scale
    dx = (b - a) / (n_grid - 1)
    xs_sorted = sorted(xs)
    n = len(xs_sorted)

    def model_cdf(x: float) -> float:
        u = (x - loc) / scale
        if name == "sigmoid":
            return 1.0 / (1.0 + math.exp(-u))
        return 0.5 * math.tanh(u) + 0.5

    # right-continuous empirical CDF via a merge over the sorted grid
    import bisect

    total = 0.0
    prev = None
    for i in range(n_grid):
        x = a + i * dx
        fn = bisect.bisect_right(xs_sorted, x) / n
        val = abs(model_cdf(x) - fn)
        if prev is not None:
            total += 0.5 * (prev + val) * dx
        prev = val
    return total


# ----- enclosure validity ---------------------------------------------------


@pytest.mark.parametrize("name", ["sigmoid", "tanh"])
def test_w1_enclosure_contains_numeric_truth(name: str) -> None:
    loc, scale = 0.3, 1.4
    xs = _logistic_quantiles(48, loc=0.0, scale=1.0)
    enc = certified_wasserstein1(name, xs, loc=loc, scale=scale)
    w1 = _w1_numeric(name, xs, loc, scale)
    assert enc.lo - 2e-3 <= w1 <= enc.hi + 2e-3
    assert enc.width < 1e-6
    assert enc.lo >= 0.0


def test_w1_is_tight_for_a_small_sample() -> None:
    xs = [-1.0, 0.0, 0.5, 2.0]
    enc = certified_wasserstein1("sigmoid", xs, loc=0.0, scale=1.0)
    w1 = _w1_numeric("sigmoid", xs, 0.0, 1.0)
    assert enc.lo - 1e-3 <= w1 <= enc.hi + 1e-3


# ----- structural properties ------------------------------------------------


def test_shifting_the_model_increases_w1_by_the_shift() -> None:
    # W1 is translation equivariant: shifting the model loc by delta moves the
    # whole transport cost by ~delta for a well-sampled model.
    xs = _logistic_quantiles(400, loc=0.0, scale=1.0)
    base = certified_wasserstein1("sigmoid", xs, loc=0.0, scale=1.0)
    shifted = certified_wasserstein1("sigmoid", xs, loc=3.0, scale=1.0)
    assert shifted.lo > base.hi
    # the increase brackets the shift magnitude (3) up to the small base term
    assert 3.0 - 0.2 <= (shifted.mid - base.mid) <= 3.0 + 0.2


def test_tanh_base_equals_sigmoid_at_half_scale() -> None:
    # The tanh CDF at scale S is the logistic CDF at scale S/2.
    xs = _logistic_quantiles(60, loc=0.1, scale=0.9)
    tanh_enc = certified_wasserstein1("tanh", xs, loc=0.2, scale=2.0)
    sig_enc = certified_wasserstein1("sigmoid", xs, loc=0.2, scale=1.0)
    assert tanh_enc.lo <= sig_enc.hi and sig_enc.lo <= tanh_enc.hi


def test_more_dispersed_samples_cost_more() -> None:
    tight = certified_wasserstein1("sigmoid", [-0.1, 0.0, 0.1], loc=0.0, scale=1.0)
    wide = certified_wasserstein1("sigmoid", [-5.0, 0.0, 5.0], loc=0.0, scale=1.0)
    assert wide.lo > tight.hi


# ----- guards ---------------------------------------------------------------


def test_arctan_is_rejected_no_finite_mean() -> None:
    with pytest.raises(NotImplementedError, match="finite"):
        certified_wasserstein1("arctan", [0.0, 1.0], loc=0.0, scale=1.0)


def test_rejects_nonpositive_scale() -> None:
    with pytest.raises(ValueError, match="scale must be > 0"):
        certified_wasserstein1("sigmoid", [0.0], loc=0.0, scale=0.0)


def test_rejects_empty_samples() -> None:
    with pytest.raises(ValueError, match="at least one sample"):
        certified_wasserstein1("sigmoid", [], loc=0.0, scale=1.0)


# ----- certified two-sample W1 ----------------------------------------------


def test_two_sample_w1_contains_closed_form() -> None:
    u = [0.4, -1.2, 3.1, 0.0, 2.7, -0.6]
    v = [1.1, -0.3, 2.0, 0.5, 1.8, -1.0]
    enc = certified_wasserstein1_samples(u, v)
    truth = _two_sample_w1(u, v)
    assert enc.lo <= truth <= enc.hi
    assert enc.width < 1e-9
    assert enc.lo >= 0.0


def test_two_sample_w1_identical_is_zero() -> None:
    u = [3.0, -1.0, 2.0, 0.5]
    enc = certified_wasserstein1_samples(u, u)
    assert enc.lo <= 0.0 <= enc.hi
    assert enc.hi < 1e-12


def test_two_sample_w1_of_shift_equals_shift() -> None:
    u = [0.0, 1.0, 2.0, 3.0, 4.0]
    v = [x + 2.5 for x in u]
    enc = certified_wasserstein1_samples(u, v)
    assert enc.lo <= 2.5 <= enc.hi
    assert enc.width < 1e-9


def test_two_sample_w1_is_order_invariant() -> None:
    # Each sample is sorted internally, so input order cannot matter.
    a = certified_wasserstein1_samples([3.0, 1.0, 2.0], [0.0, 2.0, 1.0])
    b = certified_wasserstein1_samples([1.0, 2.0, 3.0], [2.0, 1.0, 0.0])
    assert a.lo == b.lo and a.hi == b.hi


def test_two_sample_w1_rejects_unequal_length() -> None:
    with pytest.raises(ValueError, match="equal-length"):
        certified_wasserstein1_samples([0.0, 1.0], [0.0])


def test_two_sample_w1_rejects_empty() -> None:
    with pytest.raises(ValueError, match="at least one sample"):
        certified_wasserstein1_samples([], [])
