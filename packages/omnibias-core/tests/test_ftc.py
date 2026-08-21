# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Theory 09-03 / 09-17: 1-D FTC cell (not a VPINN)."""

from __future__ import annotations

from omnibias.core.ftc import (
    DISCLAIMER,
    dual_ftc_loss,
    dual_skill_report,
    ftc_block,
    honesty_payload,
    skill_report,
    worked_example,
)


def test_g1_worked_example() -> None:
    ex = worked_example()
    assert ex["collapse_err"] < 1e-12
    assert ex["ftc_err"] < 1e-12
    assert abs(ex["I"] - 0.1) < 1e-12
    integral, deriv, collapse = ftc_block(0.0, 1.0, -0.1, 0.1)
    loss = dual_ftc_loss([integral], [deriv], [deriv], [0.0], I_a=integral)
    assert loss.max_r_D < 1e-12
    assert loss.max_r_I < 1e-12
    loss0 = dual_ftc_loss([integral], [deriv], [0.0], [0.0], I_a=integral)
    assert loss0.max_r_D > 0.04


def test_g2_skill_beats_identity() -> None:
    report = skill_report()
    assert report["below_1e4"] is True
    assert report["beats_identity"] is True
    assert float(report["skill"]) > 0.0
    assert report["claimed_weak_form"] is False


def test_g2_dual_reduces_integral_residual() -> None:
    report = dual_skill_report()
    assert report["below_1e4"] is True
    assert report["ri_below_deriv_only"] is True
    assert float(report["skill"]) > 0.0
    assert report["claimed_vpinn"] is False


def test_g3_honesty() -> None:
    payload = honesty_payload()
    assert payload["claimed_weak_form"] is False
    assert payload["claimed_vpinn"] is False
    assert payload["theorem_prover_verified"] is False
    assert "not a VPINN" in DISCLAIMER
