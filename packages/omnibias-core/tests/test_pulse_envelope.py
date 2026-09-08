# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Theory 07-12: pulse envelope exact-D plus mollifier tail."""

from __future__ import annotations

from fractions import Fraction

import pytest
from omnibias.core.pulse_envelope import (
    DECAY_RATE,
    GROWTH_RATE,
    assert_honesty,
    honesty_payload,
    locked_decay_envelope,
    locked_growth_envelope,
    locked_mid_envelope,
    mollifier_tail_contains_truth,
    riccati_sigma_prime,
    tower_sigma_prime,
)


def test_g4_pulse_derivative_matches_tower() -> None:
    growth = locked_growth_envelope()
    decay = locked_decay_envelope()
    mid = locked_mid_envelope()
    assert growth.derivative() == growth.tower_derivative()
    assert decay.derivative() == decay.tower_derivative()
    assert mid.derivative() == Fraction(0)
    assert growth.derivative() == GROWTH_RATE * growth.value()
    assert decay.derivative() == DECAY_RATE * decay.value()
    assert tower_sigma_prime(Fraction(1, 2)) == riccati_sigma_prime(Fraction(1, 2))
    assert tower_sigma_prime(Fraction(1, 2)) == Fraction(1, 4)


def test_g5_mollifier_tail_contains_truth() -> None:
    assert mollifier_tail_contains_truth(half_width=3.0) is True


def test_g6_honesty() -> None:
    flags = honesty_payload()
    assert flags["navier_stokes_proof_claim"] is False
    assert flags["forced_blowup_reproof_claim"] is False
    with pytest.raises(ValueError, match="navier_stokes_proof_claim"):
        assert_honesty({"navier_stokes_proof_claim": True})
