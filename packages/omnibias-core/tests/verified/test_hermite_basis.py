# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Verified physicists' Hermite-function basis: values, self-duality, tails.

Correctness is checked against a ``math``-module-independent low-order
reference (``H_0=1``, ``H_1=2x``, ``H_2=4x^2-2``, ``H_3=8x^3-12x``), against
the exact ``(-i)^n`` self-duality claim, and against a higher-order truncation
used as a proxy for the true infinite sum (a containment check on
:meth:`~omnibias.core.verified.hermite_basis.HermiteExpansion.tail_bound`).
"""

from __future__ import annotations

import math
import random

import pytest
from omnibias.core.verified.complex_interval import ComplexInterval
from omnibias.core.verified.hermite_basis import (
    CRAMER_UNIFORM_BOUND,
    HermiteExpansion,
    gaussian_weight,
    hermite_function,
    hermite_function_normalized,
    hermite_poly_coeffs_exact,
    neg_i_power,
)
from omnibias.core.verified.interval import Interval


# --------------------------------------------------------------------------- #
# Low-order values against an independent math-module reference.
# --------------------------------------------------------------------------- #
def _hermite_phys_ref(n: int, x: float) -> float:
    """Independent reference: direct three-term recurrence in plain floats."""
    h0, h1 = 1.0, 2.0 * x
    if n == 0:
        return h0
    if n == 1:
        return h1
    for m in range(2, n + 1):
        h0, h1 = h1, 2.0 * x * h1 - 2.0 * (m - 1) * h0
    return h1


@pytest.mark.parametrize("x", [-2.0, -0.5, 0.0, 0.3, 1.0, 2.5])
def test_hermite_poly_coeffs_low_order_hand_computed(x: float) -> None:
    # H_0 = 1, H_1 = 2x, H_2 = 4x^2 - 2, H_3 = 8x^3 - 12x (hand-derived, not from math).
    assert hermite_poly_coeffs_exact(0) == (1,)
    assert hermite_poly_coeffs_exact(1) == (0, 2)
    assert hermite_poly_coeffs_exact(2) == (-2, 0, 4)
    assert hermite_poly_coeffs_exact(3) == (0, -12, 0, 8)


@pytest.mark.parametrize("n", [0, 1, 2, 3, 4, 5, 10])
@pytest.mark.parametrize("x", [-2.0, -0.5, 0.0, 0.3, 1.0, 2.5])
def test_hermite_function_contains_independent_reference(n: int, x: float) -> None:
    ref = _hermite_phys_ref(n, x) * math.exp(-0.5 * x * x)
    enclosure = hermite_function(n, x)
    assert enclosure.lo <= ref <= enclosure.hi, (n, x, ref, enclosure)


def test_hermite_poly_coeffs_rejects_negative_order() -> None:
    with pytest.raises(ValueError):
        hermite_poly_coeffs_exact(-1)


def test_gaussian_weight_matches_exp() -> None:
    for x in (-1.5, 0.0, 0.7, 3.0):
        w = gaussian_weight(x)
        ref = math.exp(-0.5 * x * x)
        assert w.lo <= ref <= w.hi


# --------------------------------------------------------------------------- #
# L2-normalized basis: Cramer's uniform bound and consistency with raw psi_n.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n", [0, 1, 2, 3, 4, 5, 8, 15])
@pytest.mark.parametrize("x", [-3.0, -1.0, 0.0, 0.5, 2.0, 4.0])
def test_normalized_hermite_function_respects_cramer_bound(n: int, x: float) -> None:
    val = hermite_function_normalized(n, x)
    assert abs(val.mid) <= CRAMER_UNIFORM_BOUND * (1.0 + 1e-9)


def test_cramer_uniform_bound_value() -> None:
    # pi**-0.25, computed independently of the rigorous PI_IV path.
    assert abs(CRAMER_UNIFORM_BOUND - math.pi**-0.25) < 1e-12


def test_normalized_consistent_with_unnormalized() -> None:
    for n in (0, 1, 2, 5):
        for x in (-1.3, 0.6, 2.1):
            raw = hermite_function(n, x)
            norm = hermite_function_normalized(n, x)
            factor = math.sqrt(2**n * math.factorial(n) * math.sqrt(math.pi))
            assert abs(norm.mid * factor - raw.mid) < 1e-9 * max(1.0, abs(raw.mid))


# --------------------------------------------------------------------------- #
# Exact (-i)^n self-duality symbol.
# --------------------------------------------------------------------------- #
def test_neg_i_power_cycle() -> None:
    expected = [1 + 0j, -1j, -1 + 0j, 1j]
    for n in range(9):
        z = neg_i_power(n)
        want = expected[n % 4]
        assert z.re.lo == z.re.hi == want.real
        assert z.im.lo == z.im.hi == want.imag


def test_neg_i_power_rejects_negative() -> None:
    with pytest.raises(ValueError):
        neg_i_power(-1)


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_fourier_transform_exact_is_bitwise_neg_i_times_coeffs(seed: int) -> None:
    rng = random.Random(seed)
    n = 12
    coeffs = [complex(rng.uniform(-3, 3), rng.uniform(-3, 3)) for _ in range(n)]
    expansion = HermiteExpansion.from_coeffs(coeffs)
    transformed = expansion.fourier_transform_exact()

    powers = [1 + 0j, -1j, -1 + 0j, 1j]
    for k, c in enumerate(coeffs):
        expect = c * powers[k % 4]
        got = transformed.get(k)
        # Exact diagonal relabelling: bitwise equal, not merely "contains".
        assert got.re.lo == got.re.hi
        assert got.im.lo == got.im.hi
        assert got.re.lo == expect.real
        assert got.im.lo == expect.imag


def test_fourier_transform_exact_is_involution_up_to_period_four() -> None:
    # Applying it four times returns exactly the original coefficients: (-i)^4 = 1.
    expansion = HermiteExpansion.from_coeffs([1 + 0j, 2 - 1j, -3 + 4j, 0.5j, 7.0])
    four_times = expansion
    for _ in range(4):
        four_times = four_times.fourier_transform_exact()
    for orig, back in zip(expansion.coeffs, four_times.coeffs, strict=True):
        assert back.re.lo == back.re.hi == orig.re.lo
        assert back.im.lo == back.im.hi == orig.im.lo


def test_fourier_transform_exact_preserves_order_and_tail_bound() -> None:
    expansion = HermiteExpansion.from_coeffs([1.0, 0.5, 0.25, 0.125])
    transformed = expansion.fourier_transform_exact()
    assert transformed.order == expansion.order
    before = expansion.tail_bound(coeff_bound=1.0, ratio=0.5, psi_bound=2.0)
    after = transformed.tail_bound(coeff_bound=1.0, ratio=0.5, psi_bound=2.0)
    assert before.lo == after.lo
    assert before.hi == after.hi


# --------------------------------------------------------------------------- #
# Tail bound: containment vs. a higher-order truncation proxy.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_tail_bound_contains_gap_to_higher_truncation(seed: int) -> None:
    rng = random.Random(seed)
    ratio = 0.6
    coeff_bound = 1.0
    n_trunc = 5
    n_extended = 30
    x = rng.uniform(-1.5, 1.5)

    # Coefficients genuinely satisfy |c_n| <= coeff_bound * ratio**n for all n.
    coeffs = [coeff_bound * (ratio**n) * rng.uniform(-1, 1) for n in range(n_extended + 1)]

    # psi_bound must uniformly bound |hat_psi_n(x)| for n > n_trunc: use the
    # normalized basis with the Cramer constant, which is genuinely uniform in n.
    psi_bound = CRAMER_UNIFORM_BOUND

    truncated = HermiteExpansion.from_coeffs(coeffs[: n_trunc + 1])
    extended = HermiteExpansion.from_coeffs(coeffs)

    kept = truncated.evaluate_kept(x, normalized=True)
    proxy_full = extended.evaluate_kept(x, normalized=True)
    gap = abs(proxy_full.re.mid - kept.re.mid)

    bound = truncated.tail_bound(coeff_bound=coeff_bound, ratio=ratio, psi_bound=psi_bound)
    # The proxy (finite but much longer) sum's gap from the kept block must lie
    # within the rigorous tail bound for the true infinite sum.
    assert gap <= bound.hi * (1.0 + 1e-9), (seed, gap, bound)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_evaluate_contains_higher_truncation_proxy(seed: int) -> None:
    rng = random.Random(seed)
    ratio = 0.5
    coeff_bound = 2.0
    n_trunc = 4
    n_extended = 25
    x = rng.uniform(-1.0, 1.0)

    coeffs = [coeff_bound * (ratio**n) * rng.uniform(-1, 1) for n in range(n_extended + 1)]
    psi_bound = CRAMER_UNIFORM_BOUND

    truncated = HermiteExpansion.from_coeffs(coeffs[: n_trunc + 1])
    extended = HermiteExpansion.from_coeffs(coeffs)

    enclosure = truncated.evaluate(
        x, coeff_bound=coeff_bound, ratio=ratio, psi_bound=psi_bound, normalized=True
    )
    proxy_full = extended.evaluate_kept(x, normalized=True)
    assert enclosure.re.lo <= proxy_full.re.mid <= enclosure.re.hi


def test_tail_bound_rejects_negative_psi_bound() -> None:
    expansion = HermiteExpansion.from_coeffs([1.0, 2.0])
    with pytest.raises(ValueError):
        expansion.tail_bound(coeff_bound=1.0, ratio=0.5, psi_bound=-1.0)


def test_tail_bound_propagates_geometric_tail_bound_errors() -> None:
    expansion = HermiteExpansion.from_coeffs([1.0, 2.0])
    with pytest.raises(ValueError):
        # ratio >= 1 with nu=1 -> divergent tail, must raise (not silently under-certify).
        expansion.tail_bound(coeff_bound=1.0, ratio=1.5, psi_bound=1.0)


# --------------------------------------------------------------------------- #
# Construction / accessor error handling.
# --------------------------------------------------------------------------- #
def test_hermite_expansion_requires_at_least_one_coefficient() -> None:
    with pytest.raises(ValueError):
        HermiteExpansion([])


def test_hermite_expansion_order_and_get() -> None:
    expansion = HermiteExpansion.from_coeffs([1.0, 2.0, 3.0])
    assert expansion.order == 2
    assert expansion.get(0).re.mid == 1.0
    assert expansion.get(2).re.mid == 3.0


def test_hermite_expansion_evaluate_kept_matches_manual_sum() -> None:
    coeffs = [1.0, 0.5, -0.25]
    expansion = HermiteExpansion.from_coeffs(coeffs)
    x = 0.8
    got = expansion.evaluate_kept(x)
    manual = sum(c * hermite_function(n, x).mid for n, c in enumerate(coeffs))
    assert got.re.lo <= manual <= got.re.hi
    assert got.im.lo <= 0.0 <= got.im.hi


def test_hermite_expansion_repr() -> None:
    expansion = HermiteExpansion.from_coeffs([1.0, 2.0])
    text = repr(expansion)
    assert "HermiteExpansion" in text
    assert "order=1" in text


def test_complex_coefficients_round_trip_through_from_value() -> None:
    c = ComplexInterval.point(2 - 3j)
    expansion = HermiteExpansion([c])
    assert expansion.get(0).re.mid == 2.0
    assert expansion.get(0).im.mid == -3.0
