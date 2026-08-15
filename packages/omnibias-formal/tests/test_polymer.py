# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Tests for planted polymer-coordination certificates."""

from __future__ import annotations

import pytest
from omnibias.formal.polymer import (
    LEGAL_POLYMER_FAMILIES,
    family_facts_hold,
    polymer_backtrack,
    polymer_certificate,
    polymer_crude,
)


def test_legal_families_are_the_two_plants() -> None:
    assert LEGAL_POLYMER_FAMILIES == ("backtrack_4", "crude_4")


@pytest.mark.parametrize("family", ["backtrack_4", "crude_4"])
def test_locked_facts_hold(family: str) -> None:
    assert family_facts_hold(family)


def test_values_match_geometry_coordination() -> None:
    pytest.importorskip("omnibias.geometry.gauge")
    from omnibias.geometry.gauge.transfer.strong_coupling import (
        polymer_coordination,
        polymer_coordination_backtrack,
    )

    assert polymer_backtrack(4) == polymer_coordination_backtrack(4) == 15
    assert polymer_crude(4) == polymer_coordination(4) == 24
    assert polymer_backtrack(4) < polymer_crude(4)


def test_unknown_family_raises() -> None:
    with pytest.raises(ValueError, match="unknown polymer family"):
        polymer_certificate("dottie")  # type: ignore[arg-type]


def test_backtrack_payload_is_exact_and_sealed() -> None:
    cert = polymer_certificate("backtrack_4")
    assert cert["payload"]["type"] == "polymer"
    assert cert["payload"]["family"] == "backtrack_4"
    assert cert["payload"]["value"] == 15
    assert "digest" in cert
    assert cert["honesty"]["unproven_claim"] is False


def test_crude_payload_is_exact_and_sealed() -> None:
    cert = polymer_certificate("crude_4")
    assert cert["payload"]["type"] == "polymer"
    assert cert["payload"]["family"] == "crude_4"
    assert cert["payload"]["value"] == 24
    assert "digest" in cert
