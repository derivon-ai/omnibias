# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Tests for planted SU(2) / SU(3) Casimir certificates."""

from __future__ import annotations

from fractions import Fraction

import pytest
from omnibias.formal.casimir import (
    LEGAL_CASIMIR_FAMILIES,
    SU2_FUND_GAP,
    SU3_FUND,
    casimir_certificate,
    casimir_of,
    family_facts_hold,
)


def test_legal_families_are_the_two_plants() -> None:
    assert LEGAL_CASIMIR_FAMILIES == ("su2_fund", "su3_fund")


@pytest.mark.parametrize("family", ["su2_fund", "su3_fund"])
def test_locked_facts_hold(family: str) -> None:
    assert family_facts_hold(family)


def test_values_match_geometry_quadratic_casimir() -> None:
    gauge = pytest.importorskip("omnibias.geometry.gauge")
    quadratic_casimir = gauge.quadratic_casimir
    Irrep = gauge.Irrep
    assert casimir_of(2, (0, 0)) == quadratic_casimir(Irrep(n=2, dynkin=(0,)))
    assert casimir_of(2, (1, 0)) == quadratic_casimir(Irrep(n=2, dynkin=(1,)))
    assert casimir_of(2, (2, 0)) == quadratic_casimir(Irrep(n=2, dynkin=(2,)))
    assert casimir_of(3, (1, 0, 0)) == quadratic_casimir(Irrep(n=3, dynkin=(1, 0)))
    assert SU2_FUND_GAP == Fraction(3, 4)
    assert SU3_FUND == Fraction(4, 3)


def test_unknown_family_raises() -> None:
    with pytest.raises(ValueError, match="unknown casimir family"):
        casimir_certificate("dottie")  # type: ignore[arg-type]


def test_su2_payload_is_exact_and_sealed() -> None:
    cert = casimir_certificate("su2_fund")
    assert cert["payload"]["type"] == "casimir"
    assert cert["payload"]["family"] == "su2_fund"
    assert cert["payload"]["value"] == [3, 4]
    assert "digest" in cert
    assert cert["honesty"]["unproven_claim"] is False


def test_su3_payload_is_exact_and_sealed() -> None:
    cert = casimir_certificate("su3_fund")
    assert cert["payload"]["type"] == "casimir"
    assert cert["payload"]["family"] == "su3_fund"
    assert cert["payload"]["value"] == [4, 3]
    assert "digest" in cert
    assert cert["honesty"]["unproven_claim"] is False
