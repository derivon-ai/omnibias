# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Theory 09-15: Hilger-OMBU hybrid."""

from __future__ import annotations

import pytest
from omnibias.timescale._core.hybrid import (
    DISCLAIMER,
    HilgerOMBUConfig,
    hilger_ombu_forward,
    hilger_ombu_limit,
    honesty_payload,
    mu_limit_skill,
    worked_example,
)


def test_g1_worked_square() -> None:
    ex = worked_example()
    assert ex["y_err"] < 1e-12
    assert ex["limit_err"] < 1e-12
    assert ex["residual_err"] < 1e-12


def test_g2_monotone_mu_limit() -> None:
    report = mu_limit_skill()
    assert report["monotone"] is True
    assert report["g2_earned"] is True


def test_g3_honesty() -> None:
    payload = honesty_payload()
    assert payload["continuum_claimed_from_mu_limit"] is False
    assert "not a continuum PDE" in DISCLAIMER


def test_mu_zero_uses_limit() -> None:
    with pytest.raises(ValueError, match="named limit"):
        hilger_ombu_forward(2.0, config=HilgerOMBUConfig(mu=0.0))
    assert hilger_ombu_limit(2.0) == 4.0
