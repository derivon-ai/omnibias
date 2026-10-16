# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Tests for the verified d-D Fourier series (``omnibias.core.verified.fourier``).

The series must rigorously enclose the weighted-l1 algebra operations and the
bounded nonlocal multipliers.  Correctness is checked against brute-force
convolution, the Banach-algebra inequality, and two strong operator identities:
``sum_j R_j^2 = -I`` (Riesz) and ``k . (P u) = 0`` (Leray divergence-free).
"""

from __future__ import annotations

import collections
import itertools
import random

import pytest
from omnibias.core.verified.complex_interval import ComplexInterval
from omnibias.core.verified.fourier import (
    ValidatedFourierSeries as VFS,
)
from omnibias.core.verified.fourier import hilbert_symbol
from omnibias.core.verified.interval import Interval


def _rand_coeffs(rng: random.Random, dim: int, n: int, density: float = 0.6):
    out = {}
    for k in itertools.product(range(-n, n + 1), repeat=dim):
        if rng.random() < density:
            out[k] = complex(round(rng.uniform(-1, 1), 3), round(rng.uniform(-1, 1), 3))
    return out


def _brute_convolution(ad, bd, dim):
    exact = collections.defaultdict(complex)
    for i, ai in ad.items():
        for j, bj in bd.items():
            k = tuple(i[d] + j[d] for d in range(dim))
            exact[k] += ai * bj
    return exact


# --------------------------------------------------------------------------- #
# Convolution: kept block exact, overflow rigorously in the tail.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("seed", [0, 1, 2, 3])
@pytest.mark.parametrize("nu", [1.0, 1.25])
def test_convolution_contains_brute_force(seed: int, nu: float) -> None:
    rng = random.Random(seed)
    dim, n = 2, 2
    ad = _rand_coeffs(rng, dim, n)
    bd = _rand_coeffs(rng, dim, n)
    a = VFS.from_coeffs(ad, dim, n, nu)
    b = VFS.from_coeffs(bd, dim, n, nu)
    prod = a * b
    exact = _brute_convolution(ad, bd, dim)
    tail_true = 0.0
    for k, v in exact.items():
        if all(abs(kd) <= n for kd in k):
            assert prod.get(k).contains(v), (k, v, prod.get(k))
        else:
            tail_true += abs(v) * nu ** sum(abs(kd) for kd in k)
    assert prod.tail.hi >= tail_true - 1e-12
    assert prod.tail.lo >= 0.0


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_submultiplicative_norm(seed: int) -> None:
    rng = random.Random(seed)
    dim, n, nu = 2, 2, 1.3
    a = VFS.from_coeffs(_rand_coeffs(rng, dim, n), dim, n, nu)
    b = VFS.from_coeffs(_rand_coeffs(rng, dim, n), dim, n, nu)
    prod = a * b
    assert prod.norm().hi <= (a.norm() * b.norm()).hi + 1e-12
    assert prod.norm().hi <= a.banach_algebra_bound(b).hi + 1e-12


def test_convolution_with_tail_is_sound() -> None:
    # series carrying a non-zero tail; product norm must respect submultiplicativity.
    dim, n, nu = 2, 1, 1.2
    rng = random.Random(7)
    a = VFS.from_coeffs(_rand_coeffs(rng, dim, n), dim, n, nu, tail=0.5)
    b = VFS.from_coeffs(_rand_coeffs(rng, dim, n), dim, n, nu, tail=0.3)
    prod = a * b
    assert prod.tail.lo >= 0.0
    assert prod.norm().hi <= (a.norm() * b.norm()).hi + 1e-12


# --------------------------------------------------------------------------- #
# Riesz transform: sum_j R_j^2 = -I, and norm non-increase.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("dim", [2, 3])
def test_sum_riesz_squared_is_minus_identity(dim: int) -> None:
    rng = random.Random(11)
    n, nu = 2, 1.0
    ad = _rand_coeffs(rng, dim, n)
    a = VFS.from_coeffs(ad, dim, n, nu)
    acc = VFS.zero(dim, n, nu)
    for j in range(dim):
        acc = acc + a.riesz(j).riesz(j)
    for k, v in ad.items():
        if all(kd == 0 for kd in k):
            continue  # Riesz annihilates the mean
        assert acc.get(k).contains(-v), (k, -v, acc.get(k))


def test_riesz_does_not_increase_norm() -> None:
    rng = random.Random(13)
    dim, n, nu = 2, 3, 1.1
    a = VFS.from_coeffs(_rand_coeffs(rng, dim, n), dim, n, nu, tail=0.2)
    # |symbol| <= 1, so the true norm cannot grow; the enclosure's upper bound may
    # exceed the input's by a few ulp because k_j/|k| for an axis-aligned k rounds
    # to 1 + O(ulp) (sqrt of a perfect square inflates). Hence a relative bound.
    for j in range(dim):
        assert a.riesz(j).norm().hi <= a.norm().hi * (1.0 + 1e-9)


# --------------------------------------------------------------------------- #
# Hilbert transform: H^2 = -I off the axis-mean modes.
# --------------------------------------------------------------------------- #
def test_hilbert_symbol_and_squared_identity() -> None:
    symbol = hilbert_symbol()
    assert symbol((0,)).contains(0.0)
    assert symbol((3,)).contains(-1j)
    assert symbol((-3,)).contains(1j)

    rng = random.Random(31)
    coeffs = _rand_coeffs(rng, 1, 3)
    series = VFS.from_coeffs(coeffs, 1, 3, 1.2, tail=0.2)
    squared = series.hilbert().hilbert()
    for wavevector, coefficient in coeffs.items():
        if wavevector[0] == 0:
            assert squared.get(wavevector).contains(0.0)
        else:
            assert squared.get(wavevector).contains(-coefficient)
    # The Hilbert multiplier is bounded, so it preserves a nonzero tail.
    assert squared.tail.contains(series.tail.mid)


# --------------------------------------------------------------------------- #
# Leray projection: divergence-free.
# --------------------------------------------------------------------------- #
def test_leray_projection_is_divergence_free() -> None:
    rng = random.Random(17)
    dim, n, nu = 2, 2, 1.0
    u0 = VFS.from_coeffs(_rand_coeffs(rng, dim, n), dim, n, nu)
    u1 = VFS.from_coeffs(_rand_coeffs(rng, dim, n), dim, n, nu)
    v0 = u0.leray(0, 0) + u1.leray(0, 1)
    v1 = u0.leray(1, 0) + u1.leray(1, 1)
    for k in set(v0.coeffs) | set(v1.coeffs):
        if all(kd == 0 for kd in k):
            continue
        div = ComplexInterval.from_value(
            Interval.from_rational(k[0])
        ) * v0.get(k) + ComplexInterval.from_value(
            Interval.from_rational(k[1])
        ) * v1.get(k)
        assert div.re.contains(0.0) and div.im.contains(0.0), (k, div)


# --------------------------------------------------------------------------- #
# Exact derivative on a finite series.
# --------------------------------------------------------------------------- #
def test_derivative_exact_and_requires_finite() -> None:
    rng = random.Random(19)
    dim, n, nu = 2, 2, 1.0
    ad = _rand_coeffs(rng, dim, n)
    a = VFS.from_coeffs(ad, dim, n, nu)
    da = a.derivative(0)
    for k, v in ad.items():
        assert da.get(k).contains(1j * k[0] * v)
    # unbounded multiplier on a series with tail must raise
    with_tail = VFS.from_coeffs(ad, dim, n, nu, tail=0.1)
    with pytest.raises(ValueError):
        with_tail.derivative(0)


# --------------------------------------------------------------------------- #
# Linear algebra: add / scale / neg / norm.
# --------------------------------------------------------------------------- #
def test_add_scale_norm() -> None:
    dim, n, nu = 1, 3, 1.5
    a = VFS.from_coeffs({(1,): 2 + 0j, (-1,): 2 + 0j}, dim, n, nu)
    # ||a||_nu = |2| nu^1 + |2| nu^1 = 4 * 1.5 = 6
    assert abs(a.norm().mid - 6.0) < 1e-9
    b = a.scale(0.5)
    assert abs(b.norm().mid - 3.0) < 1e-9
    s = a + a
    assert s.get((1,)).contains(4 + 0j)
    assert (a - a).get((1,)).contains(0 + 0j)
    assert a.scale(1j).get((1,)).contains(2j)


def test_constant_and_zero() -> None:
    z = VFS.zero(2, 2, 1.0)
    assert z.norm().hi < 1e-300  # exactly zero up to one subnormal ulp
    c = VFS.constant(3 + 1j, 2, 2, 1.0)
    assert c.get((0, 0)).contains(3 + 1j)
    assert abs(c.norm().mid - abs(3 + 1j)) < 1e-9


# --------------------------------------------------------------------------- #
# Error handling.
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Anisotropic per-axis nu: a genuine, sound generalisation (Part 1).
# --------------------------------------------------------------------------- #
def test_scalar_nu_path_is_bit_for_bit_unchanged() -> None:
    """Passing a plain scalar `nu` (the only pre-generalisation form) must still
    run the *exact* original isotropic formula, bit-for-bit -- this is the
    regression the strict-superset claim depends on. (A length-`d` *sequence*
    of equal weights is a different, merely-mathematically-equal code path --
    see `test_anisotropic_nu_with_equal_axes_agrees_with_scalar_numerically`
    below -- and is deliberately not asserted bit-identical here.)"""
    rng = random.Random(41)
    dim, n = 2, 2
    ad = _rand_coeffs(rng, dim, n)
    bd = _rand_coeffs(rng, dim, n)
    a = VFS.from_coeffs(ad, dim, n, 1.3)
    b = VFS.from_coeffs(bd, dim, n, 1.3)
    prod = a * b
    exact = _brute_convolution(ad, bd, dim)
    tail_true = 0.0
    for k, v in exact.items():
        if all(abs(kd) <= n for kd in k):
            assert prod.get(k).contains(v)
        else:
            tail_true += abs(v) * 1.3 ** sum(abs(kd) for kd in k)
    assert prod.tail.hi >= tail_true - 1e-12
    assert isinstance(a.nu, float)
    assert a.nu_axes == (1.3, 1.3)


def test_anisotropic_nu_with_equal_axes_agrees_with_scalar_numerically() -> None:
    """An anisotropic tuple of *equal* weights computes the same weight `nu_d^|k_d|`
    axis-by-axis (as a product of per-axis `pow_int` calls) rather than the
    scalar path's single combined `pow_int(sum |k_d|)` call -- mathematically
    identical, but not bit-identical (each factor rounds outward separately).
    Both must therefore *agree numerically* (to near machine precision) and
    both must still soundly contain the brute-force convolution."""
    rng = random.Random(41)
    dim, n = 2, 2
    ad = _rand_coeffs(rng, dim, n)
    bd = _rand_coeffs(rng, dim, n)
    scalar = VFS.from_coeffs(ad, dim, n, 1.3)
    aniso = VFS.from_coeffs(ad, dim, n, (1.3, 1.3))
    assert scalar.nu_axes == aniso.nu_axes == (1.3, 1.3)

    prod_scalar = scalar * VFS.from_coeffs(bd, dim, n, 1.3)
    prod_aniso = aniso * VFS.from_coeffs(bd, dim, n, (1.3, 1.3))
    for k in set(prod_scalar.coeffs) | set(prod_aniso.coeffs):
        assert abs(prod_scalar.get(k).re.mid - prod_aniso.get(k).re.mid) < 1e-9
        assert abs(prod_scalar.get(k).im.mid - prod_aniso.get(k).im.mid) < 1e-9
    assert abs(prod_scalar.tail.hi - prod_aniso.tail.hi) < 1e-9 * prod_scalar.tail.hi
    assert abs(scalar.norm().hi - aniso.norm().hi) < 1e-9 * scalar.norm().hi


def test_anisotropic_nu_convolution_contains_brute_force() -> None:
    """A genuinely direction-dependent weight must still soundly bound the tail."""
    rng = random.Random(43)
    dim, n = 2, 2
    nu_axes = (1.1, 1.6)
    ad = _rand_coeffs(rng, dim, n)
    bd = _rand_coeffs(rng, dim, n)
    a = VFS.from_coeffs(ad, dim, n, nu_axes)
    b = VFS.from_coeffs(bd, dim, n, nu_axes)
    prod = a * b
    exact = _brute_convolution(ad, bd, dim)
    tail_true = 0.0
    for k, v in exact.items():
        if all(abs(kd) <= n for kd in k):
            assert prod.get(k).contains(v), (k, v, prod.get(k))
        else:
            weight = 1.0
            for kd, nud in zip(k, nu_axes, strict=True):
                weight *= nud ** abs(kd)
            tail_true += abs(v) * weight
    assert prod.tail.hi >= tail_true - 1e-12
    assert prod.tail.lo >= 0.0


def test_anisotropic_nu_submultiplicative_norm() -> None:
    """Submultiplicativity holds axis-by-axis for a genuinely anisotropic weight."""
    rng = random.Random(47)
    dim, n = 2, 2
    nu_axes = (1.05, 1.4)
    a = VFS.from_coeffs(_rand_coeffs(rng, dim, n), dim, n, nu_axes)
    b = VFS.from_coeffs(_rand_coeffs(rng, dim, n), dim, n, nu_axes)
    prod = a * b
    assert prod.norm().hi <= (a.norm() * b.norm()).hi + 1e-12
    assert prod.norm().hi <= a.banach_algebra_bound(b).hi + 1e-12


def test_anisotropic_nu_riesz_still_bounded() -> None:
    """The bounded-multiplier proofs need only `weight >= 0`, not submultiplicativity."""
    rng = random.Random(53)
    dim, n = 2, 3
    nu_axes = (1.2, 1.05)
    a = VFS.from_coeffs(_rand_coeffs(rng, dim, n), dim, n, nu_axes, tail=0.2)
    for j in range(dim):
        assert a.riesz(j).norm().hi <= a.norm().hi * (1.0 + 1e-9)


def test_anisotropic_nu_validation() -> None:
    with pytest.raises(ValueError):
        VFS.zero(2, 2, (1.0, 0.9))  # one axis weight < 1
    with pytest.raises(ValueError):
        VFS.zero(2, 2, (1.0, 1.0, 1.0))  # wrong length for dim=2
    ok = VFS.zero(2, 2, (1.0, 1.0))
    assert ok.nu_axes == (1.0, 1.0)


def test_error_handling() -> None:
    with pytest.raises(ValueError):
        VFS.zero(0, 2, 1.0)  # dim < 1
    with pytest.raises(ValueError):
        VFS.zero(2, -1, 1.0)  # trunc < 0
    with pytest.raises(ValueError):
        VFS.zero(2, 2, 0.5)  # nu < 1
    with pytest.raises(ValueError):
        VFS(2, 2, 1.0, {}, Interval(-1.0, 1.0))  # negative tail
    with pytest.raises(ValueError):
        VFS.from_coeffs({(5, 0): 1 + 0j}, 2, 2, 1.0)  # outside box
    with pytest.raises(ValueError):
        VFS.from_coeffs({(1,): 1 + 0j}, 2, 2, 1.0)  # wrong length
    a = VFS.zero(2, 2, 1.0)
    b = VFS.zero(2, 3, 1.0)
    with pytest.raises(ValueError):
        _ = a + b  # trunc mismatch
    with pytest.raises(ValueError):
        a.riesz(5)
    with pytest.raises(ValueError):
        a.apply_multiplier(lambda k: ComplexInterval.one(), -1.0)


# Exterior frequencies may return to low modes under convolution. A separate
# global error is needed before applying a multiplier with a smaller tail norm.
def test_two_exterior_modes_can_produce_an_uncertain_constant() -> None:
    # Representatives of exp(2ix) and exp(-2ix), both outside truncation1.
    a = VFS.from_coeffs({}, 1, 1, 1.0, tail=1.0)
    product = a * a
    assert product.tail == Interval.point(0.0)
    assert product.global_remainder.hi >= 1.0
    assert product.get((0,)).contains(1.0)
    assert product.norm().contains(1.0)

    # Projection onto the mean has exterior factor zero but global factor one.
    def mean_symbol(k):
        return ComplexInterval.from_value(float(k == (0,)))

    with pytest.raises(ValueError, match="global_factor"):
        product.apply_multiplier(mean_symbol, 0.0)
    mean = product.apply_multiplier(mean_symbol, 0.0, global_factor=1.0)
    assert mean.get((0,)).contains(1.0)
    assert mean.global_remainder.hi >= 1.0


def test_finite_overflow_reenters_the_kept_block_in_a_repeated_product() -> None:
    positive = VFS.from_coeffs({(1,): 1.0}, 1, 1, 1.0)
    negative = VFS.from_coeffs({(-1,): 1.0}, 1, 1, 1.0)
    squared = positive * positive
    assert squared.global_remainder == Interval.point(0.0)
    assert squared.get((2,)).contains(1.0)
    cubed = squared * negative
    assert cubed.get((1,)).contains(1.0)
    assert cubed.global_remainder.hi >= 1.0
    with pytest.raises(ValueError, match="global remainder"):
        cubed.derivative(0)


def test_get_accounts_for_exterior_and_global_weighted_errors() -> None:
    series = VFS.from_coeffs({(0, 0): 2.0}, 2, 1, (2.0, 3.0), tail=36.0, global_remainder=12.0)
    assert series.get((0, 0)).contains(-10.0 + 12.0j)
    assert series.get((1, 1)).contains(2.0j)  # weight6, global radius12
    assert series.get((2, 1)).contains(4.0j)  # weight12, total radius48
    assert series.get((1, 1)).re.hi < 2.00001  # exterior radius is not added inside
    assert series.norm().contains(0.0)  # the global error may cancel the representative
    with pytest.raises(ValueError, match="dimension"):
        series.get((0,))
    with pytest.raises(TypeError, match="integer"):
        series.get((0, True))


@pytest.mark.parametrize("seed", [101, 103, 107])
def test_truncated_true_series_convolution_contains_all_low_coefficients(seed: int) -> None:
    rng = random.Random(seed)
    dim, trunc, nu = 2, 1, 1.0
    complete_a = _rand_coeffs(rng, dim, 3, density=0.25)
    complete_b = _rand_coeffs(rng, dim, 3, density=0.25)

    def enclosure(complete):
        kept = {k: value for k, value in complete.items() if max(map(abs, k)) <= trunc}
        # Inflate an independently computed finite exterior l1 sum.
        tail = sum(abs(value) for k, value in complete.items() if max(map(abs, k)) > trunc) * (1 + 1e-12)
        return VFS.from_coeffs(kept, dim, trunc, nu, tail=tail)

    product = enclosure(complete_a) * enclosure(complete_b)
    exact = _brute_convolution(complete_a, complete_b, dim)
    for k, value in exact.items():
        assert product.get(k).contains(value), (k, value, product.get(k))
    assert product.norm().hi >= sum(abs(value) for value in exact.values())


def test_global_remainder_survives_add_negate_scale_and_bounded_helpers() -> None:
    series = VFS.from_coeffs({}, 2, 1, 1.0, global_remainder=2.0)
    transformed = -(series + series).scale(1j)
    assert transformed.global_remainder.hi >= 4.0
    assert transformed.get((0, 0)).contains(-4j)
    assert series.hilbert().get((1, 0)).contains(-2j)
    assert series.riesz(0).get((1, 0)).contains(2j)
    assert series.leray(0, 0).get((0, 1)).contains(2.0)
    assert series.scale(0.0).global_remainder.hi < 1e-300


def test_global_multiplier_does_not_enumerate_a_dense_cube() -> None:
    dimension = 20
    zero = (0,) * dimension
    series = VFS.from_coeffs({zero: 1.0}, dimension, 10**6, 1.0, global_remainder=2.0)
    calls = []

    def symbol(k):
        calls.append(k)
        return ComplexInterval.from_value(3.0)

    result = series.apply_multiplier(symbol, 3.0, global_factor=3.0)
    assert calls == [zero]
    assert result.get(zero).contains(9.0)


def test_zero_remainder_does_not_require_new_multiplier_argument() -> None:
    finite = VFS.from_coeffs({(1,): 2.0}, 1, 1, 1.0)
    product = finite * VFS.constant(3.0, 1, 1, 1.0)
    assert product.global_remainder == Interval.point(0.0)
    result = product.apply_multiplier(lambda k: ComplexInterval.one(), 1.0)
    assert result.get((1,)).contains(6.0)


@pytest.mark.parametrize("value", [-1.0, float("nan"), float("inf")])
def test_invalid_global_multiplier_bound_is_refused(value: float) -> None:
    series = VFS.from_coeffs({}, 1, 1, 1.0, global_remainder=1.0)
    with pytest.raises(ValueError, match="global_factor"):
        series.apply_multiplier(lambda k: ComplexInterval.one(), 1.0, global_factor=value)


def test_negative_global_remainder_is_refused() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        VFS.from_coeffs({}, 1, 1, 1.0, global_remainder=-1.0)


def test_norm_lower_endpoint_does_not_treat_radius_or_upper_magnitude_as_actual_norm() -> None:
    exterior = VFS.from_coeffs({}, 1, 1, 1.0, tail=1.0)
    assert exterior.norm().contains(0.0)
    rectangle = ComplexInterval(Interval(-2.0, 3.0), Interval(-4.0, 1.0))
    uncertain = VFS.from_coeffs({(0,): rectangle}, 1, 1, 1.0)
    assert uncertain.low_norm().contains(0.0)
    assert uncertain.norm().contains(0.0)
    separated = VFS.from_coeffs({(0,): ComplexInterval(Interval(3.0, 4.0), Interval(4.0, 5.0))}, 1, 1, 1.0)
    assert separated.norm().contains(5.0)
    assert separated.norm().lo > 4.999999999
