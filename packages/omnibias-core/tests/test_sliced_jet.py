# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Theory 09-28: sliced-jet encoder."""

from __future__ import annotations

import pytest
from omnibias.core.sliced_jet import (
    DISCLAIMER,
    SlicedJetConfig,
    honesty_payload,
    sliced_jet_skill,
    worked_example,
)


def test_g1_reconstruct() -> None:
    ex = worked_example()
    assert ex["encoder_mae"] == 0.0
    assert ex["gap_mae"] == 0.375


def test_g2_skill() -> None:
    report = sliced_jet_skill()
    assert report["g2_earned"] is True


def test_g3_energy_named() -> None:
    with pytest.raises(ValueError, match="unnamed sparse"):
        SlicedJetConfig(top_k=1, energy="none")
    named = SlicedJetConfig(top_k=1, energy="hopfield")
    assert named.energy != "none"


def test_g4_honesty() -> None:
    payload = honesty_payload()
    assert payload["imagenet_claim"] is False
    assert payload["euclidean_RD_claim"] is False
    assert payload["vit_claim"] is False
    assert payload["stretch_claim"] is False
    assert "not a ViT" in DISCLAIMER
