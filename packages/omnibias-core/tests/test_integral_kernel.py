# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Theory 09-14: integral-kernel operator."""

from __future__ import annotations

import pytest
from omnibias.core.ftc import softplus
from omnibias.core.integral_kernel import (
    DISCLAIMER,
    antiderivative_skill,
    honesty_payload,
    integral_cell,
    worked_example,
)


def test_g1_cell() -> None:
    ex = worked_example()
    assert ex["g1_err"] < 1e-12
    assert abs(integral_cell(0.5, -0.5, 0.5) - (softplus(1.0) - softplus(0.0))) < 1e-12
    assert abs(ex["mass"] - 0.5) < 1e-12


def test_g2_antiderivative() -> None:
    report = antiderivative_skill()
    assert report["below_1e3"] is True
    assert report["skill_positive"] is True
    assert float(report["skill"]) > 0.0


def test_g3_honesty() -> None:
    payload = honesty_payload()
    assert payload["claimed_bem_net"] is False
    assert payload["claimed_fno_sota"] is False
    assert payload["theorem_prover_verified"] is False
    assert "not BEM-Net" in DISCLAIMER


def test_degenerate_window() -> None:
    with pytest.raises(ValueError, match="degenerate"):
        integral_cell(0.0, 0.1, 0.1)
