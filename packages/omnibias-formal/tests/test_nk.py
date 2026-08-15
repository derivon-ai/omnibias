# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Tests for planted NK / Krawczyk existence certificates."""

from __future__ import annotations

from fractions import Fraction

import pytest
from omnibias.formal.nk import (
    LEGAL_NK_FAMILIES,
    LEGAL_NK_ROUTES,
    nk_existence_certificate,
    plant_radii_kappa,
    plant_radii_poly,
)


def test_legal_plant_is_the_quadratic() -> None:
    assert LEGAL_NK_FAMILIES == ("quadratic",)
    assert LEGAL_NK_ROUTES == ("radii", "krawczyk")


def test_locked_radii_facts_hold() -> None:
    assert plant_radii_poly() < 0
    assert plant_radii_kappa() < 1
    assert plant_radii_poly() == Fraction(-1, 8)
    assert plant_radii_kappa() == Fraction(1, 3)


def test_unknown_route_raises() -> None:
    with pytest.raises(ValueError, match="unknown NK route"):
        nk_existence_certificate("dottie")  # type: ignore[arg-type]


@pytest.mark.parametrize("route", ["radii", "krawczyk"])
def test_certificate_payload_is_exact_and_sealed(route: str) -> None:
    cert = nk_existence_certificate(route)  # type: ignore[arg-type]
    assert cert["payload"]["type"] == "nk_existence"
    assert cert["payload"]["family"] == "quadratic"
    assert cert["payload"]["route"] == route
    assert cert["payload"]["center"] == [3, 2]
    assert cert["payload"]["radius"] == [1, 4]
    assert cert["payload"]["A"] == [1, 3]
    assert cert["payload"]["Y0"] == [1, 12]
    assert "digest" in cert
    assert cert["honesty"]["unproven_claim"] is False
