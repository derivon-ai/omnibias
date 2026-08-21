# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Theory 09-16: exact-MAML IFT + Poisson skill."""

from __future__ import annotations

from omnibias.core.exact_maml import (
    DISCLAIMER,
    honesty_payload,
    poisson_skill,
    quadratic_worked_example,
)


def test_g1_quadratic_ift() -> None:
    ex = quadratic_worked_example()
    assert ex["inner_err"] < 1e-12
    assert ex["ift_err"] < 1e-10
    assert abs(ex["meta_grad"]) < 1e-12


def test_g2_poisson_beats_adam() -> None:
    report = poisson_skill()
    assert report["below_1e4"] is True
    assert report["beats_adam"] is True
    assert float(report["skill"]) > 0.0


def test_g3_honesty() -> None:
    payload = honesty_payload()
    assert payload["imagenet_claim"] is False
    assert payload["stretch_claim"] is False
    assert payload["skips_chain_rule"] is False
    assert "not ImageNet" in DISCLAIMER
