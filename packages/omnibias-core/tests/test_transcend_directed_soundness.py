# SPDX-License-Identifier: Apache-2.0
"""Directed transcendental regressions, including two formerly false enclosures."""
from __future__ import annotations

import math
import random
from fractions import Fraction

import pytest
from omnibias.core.verified import transcend as tr
from omnibias.core.verified.interval import Interval

mp = pytest.importorskip("mpmath")


def _contains(box: Interval, value: object) -> bool:
    return bool(mp.mpf(box.lo) <= value <= mp.mpf(box.hi))


def test_gaussian_tail_cancellation_counterexample_is_fixed() -> None:
    with tr.certificate_mode(), mp.workdps(140):
        value = mp.erfc(mp.mpf(20)/mp.sqrt(2))/2
        box = tr.gauss_cdf_point(-20.0)
        assert _contains(box, value)
        assert box.lo > 0
        assert box.width < 1e-103
        # The previous mp point 1+erf path returned [0, minimum-subnormal].
        assert value > mp.mpf(math.ulp(0.0))


def test_atan_cap_does_not_cut_off_pi_over_two() -> None:
    with tr.certificate_mode(), mp.workdps(140):
        for x in (1e20, 1e100, 1e300):
            assert _contains(tr.atan_point(x), mp.atan(mp.mpf(x)))
            assert _contains(tr.atan_point(-x), mp.atan(-mp.mpf(x)))
        assert tr.atan_point(1e100).hi > math.pi/2


@pytest.mark.parametrize("name", ["exp", "tanh", "sigmoid", "sin", "cos", "atan", "log"])
def test_directed_elementary_points_grid_and_random(name: str) -> None:
    rng = random.Random(322)
    nodes = [-100., -20., -2., -1e-15, 0., 1e-15, 1., 10., 100.]
    nodes += [rng.uniform(-10, 10) for _ in range(30)]
    if name == "log":
        nodes = [abs(x)+1e-20 for x in nodes]
    with tr.certificate_mode(), mp.workdps(150):
        for x in nodes:
            box = Interval(*tr._enclose_point(name, x))
            z = mp.mpf(x)
            truth = 1/(1+mp.exp(-z)) if name == "sigmoid" else getattr(mp, name)(z)
            assert _contains(box, truth), (name, x, box, truth)


def test_point_context_methods_are_not_the_enclosure_oracle(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("point evaluator was used as proof")
    for name in ("tanh", "exp", "sin", "cos", "atan", "log", "erf", "erfc", "besseli"):
        monkeypatch.setattr(mp, name, forbidden)
    with tr.certificate_mode():
        assert tr.tanh_iv(Interval(-1, 1)).contains_zero()
        assert tr.exp_iv(Interval.point(0)).contains(1)
        assert tr.gauss_cdf_point(-20).lo > 0
        assert tr.besseli_point(0, 2).lo > 2


@pytest.mark.parametrize("x", [-40., -20., -12., -11.314, -8., -2., -.1, 0., .1, 2., 8., 11.314, 12., 20., 40.])
def test_erf_and_cdf_series_and_remainder_bounds(x: float) -> None:
    with tr.certificate_mode(), mp.workdps(150):
        z = mp.mpf(x)
        assert _contains(tr.erf_point(x), mp.erf(z))
        assert _contains(tr.gauss_cdf_point(x), mp.erfc(-z/mp.sqrt(2))/2)


def test_erf_cdf_monotone_interval_contains_dense_and_random() -> None:
    rng = random.Random(43)
    with tr.certificate_mode(), mp.workdps(120):
        for lo, hi in ((-20., -12.), (-9., -7.), (-2., 2.), (7., 9.)):
            erf_box, cdf_box = tr.erf_iv(Interval(lo, hi)), tr.gauss_cdf_iv(Interval(lo, hi))
            nodes = [lo+(hi-lo)*i/80 for i in range(81)]
            nodes += [rng.uniform(lo, hi) for _ in range(40)]
            for x in nodes:
                z = mp.mpf(x)
                assert _contains(erf_box, mp.erf(z))
                assert _contains(cdf_box, mp.erfc(-z/mp.sqrt(2))/2)


def test_dyadic_endpoint_conversion_covers_underflow_and_overflow() -> None:
    context = tr._iv_context(mp)
    for value in (context.mpf(2)**-1075, -context.mpf(2)**-1075,
                  context.mpf(2)**1024, -context.mpf(2)**1024):
        lo, hi = tr._bracket_iv(mp, value)
        with mp.workdps(150):
            raw = mp.make_mpf(value._mpi_[0])
            assert mp.mpf(lo) <= raw <= mp.mpf(hi)
    with tr.certificate_mode(), mp.workdps(150):
        for x in (-746., -745., -744., 709., 710.):
            assert _contains(tr.exp_iv(Interval.point(x)), mp.exp(mp.mpf(x)))


def test_private_context_reuse_does_not_share_caller_or_thread_precision() -> None:
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    context = tr._iv_context(mp)
    assert tr._iv_context(mp) is context
    old = mp.iv.dps
    try:
        mp.iv.dps = 3
        assert tr._iv_context(mp).dps == tr.MPMATH_DPS
    finally:
        mp.iv.dps = old
    barrier = Barrier(2)
    def thread_context() -> int:
        local = tr._iv_context(mp)
        barrier.wait(timeout=5)
        return id(local)
    with ThreadPoolExecutor(max_workers=2) as executor:
        first, second = executor.submit(thread_context), executor.submit(thread_context)
        assert first.result() != second.result()


def test_bessel_series_remains_a_proof_without_mpmath(monkeypatch: pytest.MonkeyPatch) -> None:
    with mp.workdps(100):
        truth = mp.besseli(7, mp.mpf(20))
    monkeypatch.setattr(tr, "_mpmath", lambda: None)
    previous = tr.set_strict_backend(True)
    try:
        assert _contains(tr.besseli_point(7, 20), truth)
        with pytest.raises(RuntimeError, match="strict mode"):
            tr.tanh_iv(Interval.point(1))
    finally:
        tr.set_strict_backend(previous)


@pytest.mark.parametrize("order", [True, 1.5, "1"])
def test_bessel_does_not_round_a_different_order_into_a_certificate(order: object) -> None:
    with pytest.raises(ValueError, match="exact integer"):
        tr.besseli_point(order, 1)
    with pytest.raises(ValueError, match="exact integer"):
        tr.besseli_iv(order, Interval(1, 2))


def test_bessel_refuses_the_unproved_overflow_region_even_with_mpmath() -> None:
    with tr.certificate_mode(), pytest.raises(ValueError, match="overflow"):
        tr.besseli_point(0, tr.BESSELI_SERIES_MAX_ARG+1)


def test_bessel_denominators_use_the_exact_integer_constructor(monkeypatch: pytest.MonkeyPatch) -> None:
    # Observe the small-order series route; do not execute an enormous-order
    # loop just to cross binary64's exact integer range.
    original = Interval.from_value
    seen: list[int] = []
    def exact_integer(cls: type[Interval], value: Interval | float | int | Fraction) -> Interval:
        if type(value) is int:
            seen.append(value)
        return original(value)
    monkeypatch.setattr(Interval, "from_value", classmethod(exact_integer))
    assert tr.besseli_point(5, 1).lo > 0
    assert seen[:5] == [1, 2, 3, 4, 5]
    assert 6 in seen and 14 in seen  # Current and next-term denominators.
    large = 2**53+1
    enclosure = original(large)
    assert Fraction(enclosure.lo) <= large <= Fraction(enclosure.hi)
    assert Fraction(float(large)) != large  # A point(float(...)) would fail.
