# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Theory 09-10: Riccati flow net."""

from __future__ import annotations

import pytest
from omnibias.core.riccati_flow import (
    DISCLAIMER,
    flow_skill,
    honesty_payload,
    logistic_flow,
    worked_example,
)


def test_g1_closed_form() -> None:
    ex = worked_example()
    assert ex["err"] < 1e-12
    assert ex["ds_ds0_err"] < 1e-12


def test_g2_beats_sigmoid_mlp() -> None:
    report = flow_skill()
    assert report["below_1e6"] is True
    assert report["beats_mlp"] is True


def test_g3_honesty() -> None:
    payload = honesty_payload()
    assert payload["claimed_deq"] is False
    assert payload["claimed_cnf"] is False
    assert "not a DEQ" in DISCLAIMER


def test_logistic_domain() -> None:
    with pytest.raises(ValueError, match="s0"):
        logistic_flow(0.0, 1.0)
    with pytest.raises(ValueError, match="s0"):
        logistic_flow(1.0, 1.0)
