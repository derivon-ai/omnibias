# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Theory 09-07: Pack-MoE slab-mass router."""

from __future__ import annotations

import pytest
from omnibias.core.pack_moe import (
    DISCLAIMER,
    ExpertWindow,
    PackMoEConfig,
    honesty_payload,
    pack_moe_forward,
    two_tone_skill,
    worked_example,
)


def test_g1_router() -> None:
    ex = worked_example()
    assert ex["g_a_err"] < 1e-12
    assert ex["g_b_err"] < 1e-12
    assert ex["y_err"] < 1e-12


def test_g2_two_tone() -> None:
    report = two_tone_skill()
    assert report["beats_pack"] is True
    assert report["below_1e2"] is True
    assert float(report["skill"]) > 0.0


def test_g3_honesty_and_softmax_raise() -> None:
    payload = honesty_payload()
    assert payload["router_is_softmax"] is False
    assert payload["temperature_collapse_used"] is False
    with pytest.raises(ValueError, match="softmax"):
        pack_moe_forward(
            0.0,
            (1.0, 3.0),
            (ExpertWindow(-0.2, 0.0), ExpertWindow(0.0, 0.2)),
            config=PackMoEConfig(router="softmax"),
        )
    assert "not a softmax" in DISCLAIMER
