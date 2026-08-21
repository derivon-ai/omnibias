# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Theory 09-23: sharpness-augmented loss."""

from __future__ import annotations

from omnibias.core.sharp_loss import (
    DISCLAIMER,
    honesty_payload,
    sharpness_skill,
    worked_example,
)


def test_g1_quadratic() -> None:
    ex = worked_example()
    assert ex["abs_hvp_err"] < 1e-12
    assert ex["abs_aug_err"] < 1e-12
    assert ex["aug"] == 1.0
    assert ex["hvp"] == 10.0


def test_g2_skill() -> None:
    report = sharpness_skill()
    assert report["g2_earned"] is True
    assert report["finite"] == 5
    assert report["vs_0806_required"] is False


def test_g3_honesty() -> None:
    payload = honesty_payload()
    assert payload["is_08_06_schedule"] is False
    assert payload["schedule_only"] is False
    assert payload["theorem_prover_verified"] is False
    assert "not 08-06" in DISCLAIMER
