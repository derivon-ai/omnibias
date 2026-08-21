# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Theory 09-24: proof-carrying forward."""

from __future__ import annotations

import pytest
from omnibias.verify._core.pci import (
    DISCLAIMER,
    PCIResult,
    honesty_payload,
    pci_sound,
    two_layer_box,
    worked_sigmoid_box,
)


def test_g1_g2_worked_box() -> None:
    result, tm = worked_sigmoid_box()
    assert pci_sound(result, tm)
    assert result.width < 0.1
    assert result.vacuous is False


def test_g3_flags_unforged() -> None:
    result, _ = worked_sigmoid_box()
    assert result.theorem_prover_verified is False
    assert result.mathlib_verified is False
    with pytest.raises(ValueError, match="lake build"):
        PCIResult(
            y_mid=0.5,
            box_lo=0.5,
            box_hi=0.55,
            vacuous=False,
            theorem_prover_verified=True,
        )
    with pytest.raises(ValueError, match="lake build"):
        PCIResult(
            y_mid=0.5,
            box_lo=0.5,
            box_hi=0.55,
            vacuous=False,
            mathlib_verified=True,
        )


def test_g4_not_step_filter() -> None:
    result, _ = worked_sigmoid_box()
    assert result.is_08_09_step_filter is False
    assert honesty_payload()["is_08_09_step_filter"] is False
    with pytest.raises(ValueError, match="08-09"):
        PCIResult(
            y_mid=0.5,
            box_lo=0.5,
            box_hi=0.55,
            vacuous=False,
            is_08_09_step_filter=True,
        )
    assert "not an 08-09" in DISCLAIMER


def test_two_layer_cap() -> None:
    result = two_layer_box()
    assert result.vacuous is False
    assert result.width < 0.1
