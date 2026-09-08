# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Theory 07-12: radial m=1 swirl-heat identity."""

from __future__ import annotations

from fractions import Fraction

import pytest
from omnibias.core.verified.swirl_heat import (
    ANISOTROPIC_H,
    anisotropic_heat_in_fragment,
    honesty_payload,
    locked_swirl_heat_plant,
    swirl_heat_residual,
    swirl_heat_residual_interval,
)
from omnibias.holonomic.swirl_heat import (
    named_swirl_heat_generators,
    prove_swirl_heat_identity,
    swirl_heat_identity_payload,
)


def test_g1_residual_is_zero_at_locked_points() -> None:
    plant = locked_swirl_heat_plant()
    assert plant.residual() == 0
    assert swirl_heat_residual(Fraction(2), Fraction(5)) == 0
    box = swirl_heat_residual_interval()
    assert box.contains(0.0)
    assert box.lo <= 0.0 <= box.hi


def test_g2_identity_proves() -> None:
    result = prove_swirl_heat_identity()
    assert result.proved
    payload = swirl_heat_identity_payload()
    assert payload["left"] == [0]
    assert payload["right"] == [0]
    k, annihilator = named_swirl_heat_generators()
    assert k == (0, 1)
    assert annihilator == (0,)


def test_g3_no_toolchain_degrades() -> None:
    result = prove_swirl_heat_identity(lean_check=True)
    assert result.proved
    assert result.verdict.theorem_prover_verified is False or result.proved


def test_g6_honesty_and_anisotropic_leftover() -> None:
    flags = honesty_payload()
    assert flags["navier_stokes_proof_claim"] is False
    assert flags["three_d_heat_theorem"] is False
    assert flags["paper_uses_h_positive"] is True
    assert anisotropic_heat_in_fragment() is False
    with pytest.raises(ValueError, match="1\\+2h"):
        swirl_heat_residual(h=ANISOTROPIC_H)
