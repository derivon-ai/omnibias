# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Rigor tests for the verified modified-Bessel enclosure ``besseli_iv``.

The soundness contract is *containment*: every enclosure must contain the true
``I_n`` at every point of its argument interval.  Both backends are exercised --
the mpmath bracket and, by forcing ``_mpmath`` to ``None``, the ascending-series
fallback whose geometric tail bound is what makes the no-mpmath path a proof
rather than a guess.

Even orders get particular attention: ``I_n`` is *even* for even ``n``, so an
interval straddling zero attains its minimum in the interior and a naive
endpoint bound would be unsound.
"""

from __future__ import annotations

import random
from collections.abc import Iterator
from contextlib import contextmanager

import pytest
from omnibias.core.verified import transcend
from omnibias.core.verified.interval import Interval
from omnibias.core.verified.transcend import (
    BESSELI_SERIES_MAX_ARG,
    besseli_iv,
    besseli_point,
    libm_fallback_used,
    set_strict_backend,
)

mpmath = pytest.importorskip("mpmath")

_ORDERS = [0, 1, 2, 3, 5, 8]

# Point, positive, negative, straddling zero, and wide -- the straddling cases
# are where an endpoint-only bound breaks for even orders.
_INTERVALS = [
    (0.0, 0.0),
    (1.0, 1.0),
    (0.0, 0.5),
    (0.25, 2.5),
    (2.0, 6.0),
    (-3.0, -1.0),
    (-1.5, 0.0),
    (-2.0, 2.0),  # straddles the even-order minimum
    (-0.5, 4.0),  # asymmetric straddle
    (-8.0, 8.0),  # wide straddle
    (10.0, 12.0),  # large argument
]


@contextmanager
def _without_mpmath(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Force the ascending-series fallback path."""
    monkeypatch.setattr(transcend, "_mpmath", lambda: None)
    yield


def _true(n: int, z: float) -> float:
    with mpmath.workdps(60):
        return float(mpmath.besseli(n, mpmath.mpf(z)))


def _grid(lo: float, hi: float, count: int = 200) -> list[float]:
    if hi <= lo:
        return [lo]
    step = (hi - lo) / (count - 1)
    return [lo + i * step for i in range(count)]


# --------------------------------------------------------------------------- #
# containment: a dense deterministic grid AND a random sample
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("order", _ORDERS)
@pytest.mark.parametrize(("lo", "hi"), _INTERVALS)
def test_enclosure_contains_a_dense_grid(order: int, lo: float, hi: float) -> None:
    out = besseli_iv(order, Interval(lo, hi))
    for z in _grid(lo, hi):
        value = _true(order, z)
        assert out.lo <= value <= out.hi, f"I_{order}({z}) = {value} escaped {out}"


@pytest.mark.parametrize("order", _ORDERS)
@pytest.mark.parametrize(("lo", "hi"), _INTERVALS)
def test_enclosure_contains_a_random_sample(order: int, lo: float, hi: float) -> None:
    rng = random.Random(0xB5E1 + order)
    out = besseli_iv(order, Interval(lo, hi))
    for _ in range(200):
        z = rng.uniform(lo, hi)
        value = _true(order, z)
        assert out.lo <= value <= out.hi, f"I_{order}({z}) = {value} escaped {out}"


@pytest.mark.parametrize("order", _ORDERS)
@pytest.mark.parametrize(("lo", "hi"), _INTERVALS)
def test_the_series_fallback_contains_a_dense_grid(
    monkeypatch: pytest.MonkeyPatch, order: int, lo: float, hi: float
) -> None:
    with _without_mpmath(monkeypatch):
        out = besseli_iv(order, Interval(lo, hi))
    for z in _grid(lo, hi, count=60):
        value = _true(order, z)
        assert out.lo <= value <= out.hi, f"I_{order}({z}) = {value} escaped {out}"


@pytest.mark.parametrize("order", _ORDERS)
@pytest.mark.parametrize(("lo", "hi"), _INTERVALS)
def test_the_series_fallback_contains_a_random_sample(
    monkeypatch: pytest.MonkeyPatch, order: int, lo: float, hi: float
) -> None:
    rng = random.Random(0x5E12 + order)
    with _without_mpmath(monkeypatch):
        out = besseli_iv(order, Interval(lo, hi))
    for _ in range(60):
        z = rng.uniform(lo, hi)
        value = _true(order, z)
        assert out.lo <= value <= out.hi, f"I_{order}({z}) = {value} escaped {out}"


# --------------------------------------------------------------------------- #
# exact values, parity and monotonicity
# --------------------------------------------------------------------------- #
def test_i0_at_zero_is_one_and_higher_orders_vanish() -> None:
    assert besseli_point(0, 0.0).contains(1.0)
    for order in (1, 2, 3, 7):
        out = besseli_point(order, 0.0)
        assert out.contains(0.0)
        assert out.lo >= 0.0


def test_the_series_fallback_agrees_on_the_exact_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _without_mpmath(monkeypatch):
        assert besseli_point(0, 0.0) == Interval.point(1.0)
        assert besseli_point(3, 0.0) == Interval.point(0.0)


@pytest.mark.parametrize("order", _ORDERS)
def test_it_increases_with_the_argument_on_the_positive_axis(order: int) -> None:
    """Strict separation, so a merely-wide enclosure cannot pass by overlapping."""
    previous = besseli_point(order, 0.0)
    for z in (0.25, 0.5, 1.0, 2.0, 4.0, 7.0):
        current = besseli_point(order, z)
        assert current.lo > previous.hi, f"I_{order} did not increase at z={z}"
        previous = current


@pytest.mark.parametrize("order", _ORDERS)
def test_parity_reflects_the_sign_of_the_argument(order: int) -> None:
    for z in (0.4, 1.3, 3.7):
        positive = besseli_point(order, z)
        negative = besseli_point(order, -z)
        if order % 2 == 0:
            assert negative.lo <= positive.lo and negative.hi >= positive.lo
            assert negative.contains(float(_true(order, z)))
        else:
            assert negative.contains(-float(_true(order, z)))


def test_a_negative_order_folds_onto_the_positive_one() -> None:
    for order in (1, 2, 5):
        for z in (0.3, 2.2):
            assert besseli_point(-order, z) == besseli_point(order, z)


@pytest.mark.parametrize("order", [1, 2, 3, 5])
@pytest.mark.parametrize("z", [0.3, 1.0, 2.5, 6.0])
def test_the_recurrence_holds_inside_the_enclosures(order: int, z: float) -> None:
    r"""``I_{n-1}(z) - I_{n+1}(z) = (2n/z) I_n(z)`` must be satisfiable.

    Interval arithmetic cannot prove an identity, but the enclosures are only
    consistent with it if the difference interval and the scaled interval
    overlap. A wrong order convention or a lost factor shows up here immediately.
    """
    lower = besseli_iv(order - 1, Interval.point(z))
    upper = besseli_iv(order + 1, Interval.point(z))
    middle = besseli_iv(order, Interval.point(z))
    difference = lower - upper
    scaled = middle * Interval.point(2.0 * order) / Interval.point(z)
    assert difference.lo <= scaled.hi and scaled.lo <= difference.hi


@pytest.mark.parametrize("order", [0, 1, 2, 4])
@pytest.mark.parametrize("z", [0.2, 1.0, 3.0, 9.0])
def test_the_two_backends_enclose_each_other(
    monkeypatch: pytest.MonkeyPatch, order: int, z: float
) -> None:
    """Independent derivations must overlap, or one of them is unsound."""
    with_mp = besseli_point(order, z)
    with _without_mpmath(monkeypatch):
        with_series = besseli_point(order, z)
    assert with_mp.lo <= with_series.hi and with_series.lo <= with_mp.hi


# --------------------------------------------------------------------------- #
# tightness and honest limits
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("order", [0, 1, 4])
@pytest.mark.parametrize("z", [0.5, 2.0, 8.0])
def test_the_enclosure_is_tight_not_merely_sound(order: int, z: float) -> None:
    out = besseli_point(order, z)
    value = _true(order, z)
    assert out.width <= 1e-12 * max(abs(value), 1.0)


@pytest.mark.parametrize("z", [0.5, 5.0, 30.0])
def test_the_series_fallback_is_tight_too(
    monkeypatch: pytest.MonkeyPatch, z: float
) -> None:
    with _without_mpmath(monkeypatch):
        out = besseli_point(0, z)
    value = _true(0, z)
    assert out.width <= 1e-10 * max(abs(value), 1.0)


def test_the_series_refuses_an_argument_it_cannot_bound_soundly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Beyond the cap the terms overflow, so refusing beats returning ``[inf, inf]``."""
    with _without_mpmath(monkeypatch), pytest.raises(ValueError, match="overflow"):
        besseli_point(0, BESSELI_SERIES_MAX_ARG + 1.0)


def test_the_series_fallback_is_unconditionally_rigorous(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """It reads no libm value, so it neither trips the flag nor strict mode.

    This is the property that separates it from ``_enclose_point``: there is no
    "assume libm is accurate to 4 ulp" premise to record, so a certificate built
    on it stays unconditional and strict mode has nothing to refuse.
    """
    transcend.clear_libm_fallback_used()
    previous = set_strict_backend(True)
    try:
        with _without_mpmath(monkeypatch):
            out = besseli_point(0, 1.5)
    finally:
        set_strict_backend(previous)
    assert out.contains(_true(0, 1.5))
    assert libm_fallback_used() is False


def test_a_wide_interval_still_encloses_its_interior_minimum() -> None:
    """The even-order trap: the minimum sits at ``z = 0``, not at an endpoint."""
    out = besseli_iv(0, Interval(-4.0, 4.0))
    assert out.lo <= 1.0  # I_0(0) = 1 is the interior minimum
    assert out.contains(1.0)
    assert out.hi >= _true(0, 4.0)
    # An endpoint-only bound would have started at I_0(-4) = I_0(4) ~ 11.3.
    assert out.lo < 2.0
