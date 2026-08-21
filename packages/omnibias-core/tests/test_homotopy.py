# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Theory 09-20: Kantorovich homotopy continuation."""

from __future__ import annotations

from omnibias.core.homotopy import (
    DISCLAIMER,
    homotopy_skill,
    homotopy_train,
    honesty_payload,
    worked_example,
)


def test_g1_tau_tenth() -> None:
    ex = worked_example()
    assert ex["accepted"] is True
    assert float(ex["abs_h"]) < 1e-10
    assert float(ex["residual"]) < 1e-10


def test_g2_skill() -> None:
    report = homotopy_skill()
    assert report["honest"] is True
    assert report["g2_earned"] is True
    assert not (report["quad_claimed_root"] and report["quad_halted"])


def test_g3_honesty() -> None:
    payload = honesty_payload()
    assert payload["stretch_claim"] is False
    assert payload["theorem_prover_verified"] is False
    assert payload["empty_ball_is_reject"] is True
    assert "not CCF stretch" in DISCLAIMER


def test_quadratic_halts_before_empty_root() -> None:
    report = homotopy_train(theta0=1.0, family="quadratic")
    assert report.halted is True
    assert report.claimed_root is False
    assert report.tau_final <= 0.25
