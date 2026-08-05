# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Soundness + Lean-obligation tests for certified surrogate-gradient bounds."""

from __future__ import annotations

import math
import random

import pytest
from omnibias.core.proof import lean_check_available
from omnibias.core.proof.certificate import schema_errors_v1
from omnibias.core.verified.interval import Interval
from omnibias.verify import (
    KERNELS,
    agreement_margin_iv,
    certify_agreement_margin,
    certify_no_dead_unit,
    jet_remainder_iv,
    mollification_bias_iv,
    mollified_lipschitz_iv,
    surrogate_kernel_iv,
    tail_gradient_iv,
)


def _kernel_true(kernel: str, u: float) -> float:
    if kernel == "box":
        return 1.0 if abs(u) <= 1.0 else 0.0
    if kernel == "tanh":
        return 1.0 - math.tanh(u) ** 2
    if kernel == "logistic":
        s = 1.0 / (1.0 + math.exp(-u))
        return 4.0 * s * (1.0 - s)
    if kernel == "gaussian":
        return math.exp(-0.5 * u * u)
    if kernel == "cauchy":
        return 1.0 / (1.0 + u * u)
    raise AssertionError(kernel)


@pytest.mark.parametrize("kernel", list(KERNELS))
@pytest.mark.parametrize("beta", [0.5, 1.0, 3.0])
def test_pointwise_kernel_encloses_truth(kernel: str, beta: float) -> None:
    for i in range(-40, 41):
        z = i * 0.137  # avoid landing exactly on the box edge |beta z| = 1
        iv = surrogate_kernel_iv(beta, z, kernel=kernel)
        assert iv.contains(_kernel_true(kernel, beta * z))
        assert 0.0 <= iv.lo <= iv.hi


@pytest.mark.parametrize("kernel", ["tanh", "logistic", "gaussian", "cauchy"])
def test_region_enclosure_contains_every_interior_point(kernel: str) -> None:
    beta, lo, hi = 1.5, -2.0, 2.0
    region = surrogate_kernel_iv(beta, Interval(lo, hi), kernel=kernel)
    for i in range(101):
        z = lo + (hi - lo) * i / 100.0
        assert region.contains(_kernel_true(kernel, beta * z))


@pytest.mark.parametrize("kernel", ["tanh", "logistic", "gaussian", "cauchy"])
def test_region_enclosure_grid_and_random(kernel: str) -> None:
    """Founding delta->0 soundness rule: the region enclosure contains the true
    kernel value phi(beta*z) at a dense grid AND a random sample in the box."""
    beta, lo, hi = 1.5, -2.0, 2.0
    region = surrogate_kernel_iv(beta, Interval(lo, hi), kernel=kernel)
    rng = random.Random(7)
    samples = [lo + (hi - lo) * i / 60.0 for i in range(61)]
    samples += [rng.uniform(lo, hi) for _ in range(60)]
    for z in samples:
        assert region.contains(_kernel_true(kernel, beta * z))


@pytest.mark.parametrize("kernel", list(KERNELS))
def test_peak_at_zero_is_unit_height(kernel: str) -> None:
    assert surrogate_kernel_iv(2.0, 0.0, kernel=kernel).contains(1.0)


def test_box_tail_is_certified_dead_but_heavy_tails_are_alive() -> None:
    # |beta z| = 2 > 1: the STE box gives an exactly-zero gradient (dead zone).
    box_tail = tail_gradient_iv(2.0, 1.0, kernel="box")
    assert box_tail.lo == 0.0 and box_tail.hi == 0.0
    # Heavy / full-support kernels stay strictly positive, cauchy the largest.
    cauchy = tail_gradient_iv(1.0, 5.0, kernel="cauchy")
    gaussian = tail_gradient_iv(1.0, 5.0, kernel="gaussian")
    assert cauchy.lo > 0.0
    assert gaussian.lo > 0.0
    assert cauchy.hi > gaussian.hi


def test_mollification_bias_decreases_with_beta() -> None:
    soft = mollification_bias_iv(1.0, 1.0)
    sharp = mollification_bias_iv(5.0, 1.0)
    assert soft.contains(1.0 - math.tanh(1.0))
    assert sharp.contains(1.0 - math.tanh(5.0))
    assert sharp.hi < soft.hi  # sharper surrogate agrees with the hard sign sooner


def test_lipschitz_is_beta() -> None:
    assert mollified_lipschitz_iv(3.0).contains(3.0)
    assert agreement_margin_iv(1.0, 1.0).contains(math.tanh(1.0))


def _windowed_average(beta: float, z: float, h: float, n: int = 8001) -> float:
    # high-resolution trapezoid of s'(z+u) = beta (1 - tanh^2(beta(z+u))) over [-h, h]
    total = 0.0
    for i in range(n):
        u = -h + 2.0 * h * i / (n - 1)
        val = beta * (1.0 - math.tanh(beta * (z + u)) ** 2)
        total += val if (0 < i < n - 1) else 0.5 * val
    return total * (2.0 * h / (n - 1)) / (2.0 * h)


def _corrected_slope(beta: float, z: float, h: float) -> float:
    th = math.tanh(beta * z)
    s1 = beta * (1.0 - th * th)
    s3 = beta**3 * (-2.0) * (1.0 - th * th) * (1.0 - 3.0 * th * th)
    return s1 + (h * h / 6.0) * s3


@pytest.mark.parametrize("beta", [2.0, 4.0])
@pytest.mark.parametrize("z", [-0.3, 0.0, 0.4])
def test_jet_remainder_encloses_true_residual(beta: float, z: float) -> None:
    h = 1.0 / beta
    bound = jet_remainder_iv(beta, z)
    true_residual = _windowed_average(beta, z, h) - _corrected_slope(beta, z, h)
    assert bound.lo <= true_residual <= bound.hi
    assert bound.lo == -bound.hi  # symmetric


@pytest.mark.parametrize("z", [-0.3, 0.0, 0.4])
def test_jet_remainder_is_fourth_order_in_the_window(z: float) -> None:
    # Theorem 2 is O(h^4) in the averaging window h at a *fixed* surrogate beta;
    # halving h must shrink the bound by ~16x (the curvature term is exact to O(h^4)).
    beta = 1.0
    coarse = jet_remainder_iv(beta, z, window=0.2).hi
    fine = jet_remainder_iv(beta, z, window=0.1).hi
    finest = jet_remainder_iv(beta, z, window=0.05).hi
    assert 0.0 < finest < fine / 8.0 < coarse / 8.0


def test_certify_no_dead_unit_cauchy_is_lean_checkable() -> None:
    cb = certify_no_dead_unit(1.0, -3.0, 3.0, kernel="cauchy")
    assert cb.interval.lo > 0.0
    assert cb.lean_checkable
    assert "enclosed_quantity_pos" in (cb.lean_obligation or "")
    assert schema_errors_v1(cb.certificate) == []


def test_certify_no_dead_unit_box_tail_is_provably_dead() -> None:
    cb = certify_no_dead_unit(1.0, 2.0, 3.0, kernel="box")
    assert cb.interval.lo == 0.0 and cb.interval.hi == 0.0
    assert not cb.lean_checkable  # cannot prove 0 < phi when phi is exactly 0


def test_certify_agreement_margin_sign_flips_with_margin() -> None:
    far = certify_agreement_margin(1.0, 1.0)  # tanh(1) - 1/2 > 0
    assert far.interval.lo > 0.0
    assert "enclosed_quantity_pos" in (far.lean_obligation or "")
    near = certify_agreement_margin(1.0, 0.1)  # tanh(0.1) - 1/2 < 0
    assert near.interval.hi < 0.0
    assert "enclosed_quantity_neg" in (near.lean_obligation or "")


def test_invalid_inputs_are_rejected() -> None:
    with pytest.raises(ValueError):
        surrogate_kernel_iv(1.0, 0.0, kernel="nope")
    with pytest.raises(ValueError):
        mollification_bias_iv(1.0, 0.0)
    with pytest.raises(ValueError):
        mollified_lipschitz_iv(0.0)


@pytest.mark.skipif(not lean_check_available(), reason="Lean toolchain/kernel not present")
def test_lean_kernel_discharges_positive_certificate() -> None:
    cb = certify_agreement_margin(1.0, 1.0)
    assert cb.lean_verify() is True
