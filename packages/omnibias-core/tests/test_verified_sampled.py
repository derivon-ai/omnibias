# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Tests for :func:`omnibias.core.verified.sampled.hoeffding_enclosure`.

The Tier C poly-Laplacian estimator (:mod:`omnibias.jax.laplacian` /
:mod:`omnibias.torch.laplacian`) reports its sample mean through this
probabilistic enclosure; these tests pin its coverage rate, its honesty
guards, and its numeric formula independent of either backend.
"""

from __future__ import annotations

import random
from math import isclose, log, sqrt

import pytest
from omnibias.core.verified.sampled import ConcentrationReport, hoeffding_enclosure


def test_half_width_matches_closed_form() -> None:
    samples = [0.1, 0.4, 0.6, 0.9, 0.5]
    delta = 1e-6
    report = hoeffding_enclosure(samples, value_range=(0.0, 1.0), delta=delta)
    n = len(samples)
    want_half_width = 1.0 * sqrt(log(2.0 / delta) / (2.0 * n))
    mean = sum(samples) / n
    assert isclose(report.mean, mean, rel_tol=1e-12)
    # Interval endpoints bracket mean +/- half_width (outward-rounded, so
    # they are never *tighter* than the exact half-width).
    assert report.interval.lo <= mean - want_half_width
    assert report.interval.hi >= mean + want_half_width


def test_returns_a_concentration_report() -> None:
    report = hoeffding_enclosure([1.0, 2.0, 3.0], value_range=(0.0, 5.0))
    assert isinstance(report, ConcentrationReport)
    assert report.n == 3
    assert report.value_range == (0.0, 5.0)


def test_interval_width_shrinks_as_1_over_sqrt_n() -> None:
    rng = random.Random(0)
    small = [rng.uniform(0.0, 1.0) for _ in range(10)]
    large = small + [rng.uniform(0.0, 1.0) for _ in range(990)]
    r_small = hoeffding_enclosure(small, value_range=(0.0, 1.0))
    r_large = hoeffding_enclosure(large, value_range=(0.0, 1.0))
    w_small = r_small.interval.hi - r_small.interval.lo
    w_large = r_large.interval.hi - r_large.interval.lo
    # 100x more samples -> ~10x narrower interval (1/sqrt(n) rate).
    ratio = w_small / w_large
    assert 8.0 < ratio < 12.0


def test_coverage_over_many_trials() -> None:
    """Over many independent trials, the enclosure should contain the true
    mean at least ``1 - delta`` of the time (a weak, but real, empirical
    check of the Hoeffding guarantee, not a substitute for the proof)."""
    rng = random.Random(42)
    true_mean = 0.3
    delta = 0.05
    n_per_trial = 200
    trials = 200
    misses = 0
    for _ in range(trials):
        samples = [1.0 if rng.random() < true_mean else 0.0 for _ in range(n_per_trial)]
        report = hoeffding_enclosure(samples, value_range=(0.0, 1.0), delta=delta)
        if not (report.interval.lo <= true_mean <= report.interval.hi):
            misses += 1
    # Hoeffding is a worst-case bound so the empirical miss rate is typically
    # far below delta; assert a loose multiple of it to keep this non-flaky.
    assert misses / trials <= delta * 3


def test_stderr_is_zero_for_a_single_sample() -> None:
    report = hoeffding_enclosure([0.5], value_range=(0.0, 1.0))
    assert report.stderr == 0.0


def test_degenerate_zero_width_range_gives_a_point_interval() -> None:
    """A zero-width declared range collapses the half-width to 0; the interval
    is still outward-rounded by one ulp (never tighter than the true point)."""
    report = hoeffding_enclosure([2.0, 2.0, 2.0], value_range=(2.0, 2.0))
    assert report.interval.lo <= 2.0 <= report.interval.hi
    assert report.interval.hi - report.interval.lo < 1e-12


def test_rejects_empty_samples() -> None:
    with pytest.raises(ValueError, match="at least one sample"):
        hoeffding_enclosure([], value_range=(0.0, 1.0))


def test_rejects_sample_outside_declared_range() -> None:
    with pytest.raises(ValueError, match="outside the declared value_range"):
        hoeffding_enclosure([0.5, 1.5], value_range=(0.0, 1.0))


def test_rejects_non_finite_sample() -> None:
    with pytest.raises(ValueError, match="is not finite"):
        hoeffding_enclosure([0.5, float("nan")], value_range=(0.0, 1.0))
    with pytest.raises(ValueError, match="is not finite"):
        hoeffding_enclosure([0.5, float("inf")], value_range=(0.0, 2.0))


def test_rejects_bad_value_range() -> None:
    with pytest.raises(ValueError, match="must be <="):
        hoeffding_enclosure([0.5], value_range=(1.0, 0.0))
    with pytest.raises(ValueError, match="must be finite"):
        hoeffding_enclosure([0.5], value_range=(float("-inf"), 1.0))


def test_rejects_bad_delta() -> None:
    with pytest.raises(ValueError, match="delta must be"):
        hoeffding_enclosure([0.5], value_range=(0.0, 1.0), delta=0.0)
    with pytest.raises(ValueError, match="delta must be"):
        hoeffding_enclosure([0.5], value_range=(0.0, 1.0), delta=1.0)
