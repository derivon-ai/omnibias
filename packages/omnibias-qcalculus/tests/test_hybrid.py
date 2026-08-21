# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Theory 09-15: q-OMBU hybrid."""

from __future__ import annotations

import pytest
from omnibias.qcalculus._core.hybrid import (
    DISCLAIMER,
    QOMBUConfig,
    honesty_payload,
    q_limit_skill,
    q_ombu_forward,
    q_ombu_limit,
    worked_example,
)


def test_g1_worked_square() -> None:
    ex = worked_example()
    assert ex["y_err"] < 1e-12
    assert ex["limit_err"] < 1e-12
    assert ex["residual_err"] < 1e-12


def test_g2_monotone_q_limit() -> None:
    report = q_limit_skill()
    assert report["monotone"] is True
    assert report["g2_earned"] is True


def test_g3_honesty() -> None:
    payload = honesty_payload()
    assert payload["continuum_claimed_from_q_limit"] is False
    assert payload["theorem_prover_verified"] is False
    assert "not founding bias collapse" in DISCLAIMER


def test_q_one_uses_limit() -> None:
    with pytest.raises(ValueError, match="named limit"):
        q_ombu_forward(2.0, config=QOMBUConfig(q=1.0))
    assert q_ombu_limit(2.0) == 4.0
