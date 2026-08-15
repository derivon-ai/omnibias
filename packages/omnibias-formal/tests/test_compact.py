# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Tests for planted compact-box residual and finite-matrix gap certificates."""

from __future__ import annotations

from fractions import Fraction

import pytest
from omnibias.formal.compact import (
    LEGAL_COMPACT_FAMILIES,
    TRANSFER_RATIO,
    compact_box_certificate,
    family_facts_hold,
)


def test_legal_families_are_the_two_plants() -> None:
    assert LEGAL_COMPACT_FAMILIES == ("ns_box", "transfer_2x2")


@pytest.mark.parametrize("family", ["ns_box", "transfer_2x2"])
def test_locked_facts_hold(family: str) -> None:
    assert family_facts_hold(family)


def test_transfer_ratio_is_five_eighths() -> None:
    assert TRANSFER_RATIO == Fraction(5, 8)
    assert abs(TRANSFER_RATIO) < 1


def test_unknown_family_raises() -> None:
    with pytest.raises(ValueError, match="unknown compact-box family"):
        compact_box_certificate("dottie")  # type: ignore[arg-type]


def test_ns_box_payload_is_exact_and_sealed() -> None:
    cert = compact_box_certificate("ns_box")
    assert cert["payload"]["type"] == "compact_box"
    assert cert["payload"]["family"] == "ns_box"
    assert cert["payload"]["lo"] == [1, 2]
    assert cert["payload"]["hi"] == [1, 1]
    assert cert["payload"]["residual_lo"] == [1, 2]
    assert "digest" in cert
    assert cert["honesty"]["unproven_claim"] is False


def test_transfer_payload_is_exact_and_sealed() -> None:
    cert = compact_box_certificate("transfer_2x2")
    assert cert["payload"]["type"] == "compact_box"
    assert cert["payload"]["family"] == "transfer_2x2"
    assert cert["payload"]["a00"] == [13, 2]
    assert cert["payload"]["a01"] == [3, 2]
    assert cert["payload"]["ratio"] == [5, 8]
    assert "digest" in cert
    assert cert["honesty"]["unproven_claim"] is False
