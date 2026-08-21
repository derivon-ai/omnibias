# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Theory 09-18: remainder training."""

from __future__ import annotations

from omnibias.core.remainder_train import (
    DISCLAIMER,
    RemainderTrainConfig,
    exp_jet,
    exp_remainder_skill,
    honesty_payload,
    remainder_loss,
    worked_example,
)


def test_g1_analytic_exp() -> None:
    ex = worked_example()
    assert ex["R2_err"] < 1e-12
    assert ex["loss"] > 0.0


def test_g2_remainder_beats_value() -> None:
    report = exp_remainder_skill()
    assert report["below_1e4"] is True
    assert report["beats_value"] is True


def test_g3_honesty() -> None:
    payload = honesty_payload()
    assert payload["is_03_10"] is False
    assert payload["is_03_13"] is False
    assert payload["stretch_claim"] is False
    assert "not spec 03-10" in DISCLAIMER


def test_birth_hook_is_not_03_13() -> None:
    report = remainder_loss([0.0], exp_jet(2), [0.2])
    assert report["birth_suggested"] is True
    assert report["indicator_peak"] > 1.0
    tiny = remainder_loss(
        [1.22],
        exp_jet(2),
        [0.2],
        config=RemainderTrainConfig(birth_threshold=1.0),
    )
    assert tiny["birth_suggested"] is False
