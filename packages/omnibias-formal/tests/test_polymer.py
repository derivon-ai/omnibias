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
    polymer_first_step,
)


def test_legal_families_are_the_three_plants() -> None:
    assert LEGAL_POLYMER_FAMILIES == ("backtrack_4", "crude_4", "first_step_4")


@pytest.mark.parametrize("family", ["backtrack_4", "crude_4", "first_step_4"])
def test_locked_facts_hold(family: str) -> None:
    assert family_facts_hold(family)


def test_values_match_geometry_coordination() -> None:
    pytest.importorskip("omnibias.geometry.gauge")
    from omnibias.geometry.gauge.transfer.strong_coupling import (
        polymer_coordination,
        polymer_coordination_backtrack,
    )
    from omnibias.geometry.gauge.transfer.strong_coupling import (
        polymer_first_step as geometry_first_step,
    )

    assert polymer_backtrack(4) == polymer_coordination_backtrack(4) == 15
    assert polymer_crude(4) == polymer_coordination(4) == 24
    assert polymer_first_step(4) == geometry_first_step(4) == 20
    assert polymer_backtrack(4) < polymer_first_step(4) < polymer_crude(4)


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


def test_first_step_payload_is_exact_and_sealed() -> None:
    cert = polymer_certificate("first_step_4")
    assert cert["payload"]["type"] == "polymer"
    assert cert["payload"]["family"] == "first_step_4"
    assert cert["payload"]["value"] == 20
    assert "digest" in cert
