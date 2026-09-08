# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Theory 07-11: fixed-order weighted coefficient class."""

from __future__ import annotations

from fractions import Fraction

import pytest
from omnibias.core.verified.sequence_space import ValidatedSeries, geometric_tail_bound
from omnibias.core.verified.weighted_class import (
    PulseWeight,
    WeightedSample,
    assert_honesty,
    check_pointwise_bound,
    gevrey_majorant,
    honesty_payload,
    locked_bound,
    locked_samples,
    violating_sample,
)


def test_g1_locked_samples_lie_in_enclosure() -> None:
    report = check_pointwise_bound(locked_samples(), locked_bound())
    assert report.holds
    assert report.failing == ()


def test_g2_violating_sample_is_named() -> None:
    report = check_pointwise_bound((violating_sample(),), locked_bound())
    assert report.holds is False
    assert "too_large" in report.failing


def test_g3_gevrey_majorant_up_to_kmax() -> None:
    coeffs = (Fraction(1), Fraction(1, 2), Fraction(1, 4))
    report = gevrey_majorant(coeffs, s=0, rho=Fraction(1, 2), k_max=2)
    assert report.holds
    assert report.k_max == 2
    with pytest.raises(ValueError, match="out-of-fragment"):
        gevrey_majorant(
            (Fraction(1), Fraction(1, 2), Fraction(1, 4), Fraction(1, 8)),
            s=0,
            rho=Fraction(1, 2),
            k_max=2,
        )


def test_g4_geometric_validated_series_unregressed() -> None:
    series = ValidatedSeries.from_coeffs((1.0, 0.5, 0.25), nu=0.5, tail=0.0)
    assert series.norm().lo >= 0.0
    tail = geometric_tail_bound(1.0, ratio=0.5, nu=0.5, n_trunc=2)
    assert tail.lo >= 0.0


def test_g5_honesty_no_gevrey_or_ns_claim() -> None:
    flags = honesty_payload()
    assert flags["gevrey_class_claim"] is False
    assert flags["navier_stokes_proof_claim"] is False
    assert flags["forced_blowup_reproof_claim"] is False
    with pytest.raises(ValueError, match="gevrey_class_claim"):
        assert_honesty({"gevrey_class_claim": True})


def test_pulse_weight_monomial() -> None:
    pulse = PulseWeight(L_s=Fraction(1), coeffs=(Fraction(1),))
    assert pulse.eval(Fraction(1, 2)) == Fraction(1)
    extra = WeightedSample("env", violating_sample().value, envelope=Fraction(3))
    report = check_pointwise_bound((extra,), locked_bound())
    assert report.holds
