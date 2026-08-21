# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Theory 09-08: Characteristic-Net."""

from __future__ import annotations

from omnibias.pinn.characteristic import (
    DISCLAIMER,
    honesty_payload,
    learn_v_skill,
    shock_flag_report,
    worked_example,
)


def test_g1_constant_v() -> None:
    ex = worked_example()
    assert ex["err"] < 1e-12
    assert ex["crossed"] == 0.0


def test_g2_learned_v() -> None:
    report = learn_v_skill()
    assert report["below_1e3"] is True
    assert report["v_below_1e3"] is True
    assert float(report["skill_vs_zero"]) > 0.0
    assert report["g2_earned"] is True


def test_g3_shock_flag() -> None:
    report = shock_flag_report()
    assert report["crossed_any"] is True
    assert report["unique_after_shock_claimed"] is False
    payload = honesty_payload()
    assert payload["is_02_13"] is False
    assert payload["ns_claim"] is False
    assert "not 02-13" in DISCLAIMER
