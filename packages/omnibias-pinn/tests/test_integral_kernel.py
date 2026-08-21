# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Theory 09-14: pinn.operator integral kernel."""

from __future__ import annotations

from omnibias.pinn.operator import honesty_payload, integral_cell, worked_example
from omnibias.pinn.operator._core.integral_kernel import (
    DISCLAIMER,
    antiderivative_skill,
)


def test_g1_cell() -> None:
    ex = worked_example()
    assert ex["g1_err"] < 1e-12
    assert abs(integral_cell(0.5, -0.5, 0.5) - ex["g1_cell"]) < 1e-12


def test_g2_antiderivative() -> None:
    report = antiderivative_skill()
    assert report["below_1e3"] is True
    assert report["skill_positive"] is True


def test_g3_honesty() -> None:
    payload = honesty_payload()
    assert payload["claimed_bem_net"] is False
    assert payload["claimed_fno_sota"] is False
    assert "not BEM-Net" in DISCLAIMER
