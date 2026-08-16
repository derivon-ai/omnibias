# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Tests for planted Racah 6j certificates."""

from __future__ import annotations

from fractions import Fraction

import pytest
from omnibias.formal.sixj import (
    LEGAL_SIXJ_FAMILIES,
    family_facts_hold,
    sixj_certificate,
)


def test_legal_families_are_the_two_plants() -> None:
    assert LEGAL_SIXJ_FAMILIES == ("half_half_zero", "all_half_vanishes")


@pytest.mark.parametrize("family", LEGAL_SIXJ_FAMILIES)
def test_locked_facts_hold(family: str) -> None:
    assert family_facts_hold(family)


def test_unknown_family_raises() -> None:
    with pytest.raises(ValueError, match="unknown sixj family"):
        sixj_certificate("dottie")  # type: ignore[arg-type]


def test_half_payload_is_exact_and_sealed() -> None:
    cert = sixj_certificate("half_half_zero")
    assert cert["payload"]["type"] == "sixj"
    assert cert["payload"]["family"] == "half_half_zero"
    assert cert["payload"]["value"] == [-1, 2]
    assert Fraction(*cert["payload"]["value"]) == Fraction(-1, 2)
    assert "digest" in cert


def test_vanishing_payload_is_exact_and_sealed() -> None:
    cert = sixj_certificate("all_half_vanishes")
    assert cert["payload"]["type"] == "sixj"
    assert cert["payload"]["value"] == [0, 1]
