# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Tests for omnibias.core.spectral_design (theory 01-07 G1/G4)."""

from __future__ import annotations

import math

import pytest
from omnibias.core.spectral_design import (
    alpha_for_peak,
    design_band_plan,
    locate_peak_numerically,
    peak_frequency,
    relative_bandwidth,
    response_profile,
)


@pytest.mark.parametrize("base", ("gaussian", "sech"))
@pytest.mark.parametrize("order", range(1, 9))
def test_peak_matches_numerical_argmax(base: str, order: int) -> None:
    """G1: closed-form peak vs argmax of R, relative <= 1e-6."""
    alpha = 1.7
    pred = peak_frequency(base, order, alpha)
    found = locate_peak_numerically(base, order, alpha)
    rel = abs(pred - found) / pred
    assert rel <= 1e-6
    # Inverse is consistent.
    a2 = alpha_for_peak(base, order, pred)
    assert a2 == pytest.approx(alpha, rel=1e-12)


def test_worked_example_peaks() -> None:
    plan = design_band_plan("sech", xi_lo=1.0, xi_hi=32.0, channels=4, order=2)
    peaks = tuple(
        peak_frequency("sech", n, a) for n, a in zip(plan.orders, plan.scales, strict=True)
    )
    assert peaks == pytest.approx((2.0, 4.0, 8.0, 16.0), rel=1e-12)
    assert plan.flatness < 3.0
    assert plan.has_spectral_hole() is False


def test_holed_plan_is_flagged() -> None:
    """G4: two far peaks leave a hole; four-channel plan does not."""
    from omnibias.core.spectral_design import band_plan_from_peaks

    hole = band_plan_from_peaks(
        "sech", peaks=(2.0, 16.0), order=2, xi_lo=1.0, xi_hi=32.0
    )
    dense = design_band_plan("sech", xi_lo=1.0, xi_hi=32.0, channels=4, order=2)
    assert hole.has_spectral_hole() is True
    assert hole.flatness > dense.flatness
    assert dense.has_spectral_hole() is False


def test_relative_bandwidth_shrinks_with_order() -> None:
    assert relative_bandwidth("sech", 4) < relative_bandwidth("sech", 1)
    assert relative_bandwidth("gaussian", 4) == pytest.approx(1.0 / math.sqrt(8.0))


def test_rejects_tanh_and_bad_inputs() -> None:
    with pytest.raises(ValueError, match="L1"):
        response_profile("tanh", 1, 1.0, (1.0,))
    with pytest.raises(ValueError, match="order"):
        peak_frequency("sech", 0, 1.0)
