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
from omnibias.core.verified.fourier import ValidatedFourierSeries as VFS
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
