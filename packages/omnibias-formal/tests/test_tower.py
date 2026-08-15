# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Tests for exact tower-coefficient certificates."""

from __future__ import annotations

import pytest
from omnibias.core.verified.coeffs import (
    hermite_coeffs_exact,
    sech_poly_coeffs_exact,
    sigmoid_poly_coeffs_exact,
    tanh_poly_coeffs_exact,
)
from omnibias.formal.tower import (
    LEGAL_TOWER_FAMILIES,
    tower_coeffs,
    tower_coeffs_certificate,
)


def test_legal_families_are_the_four_recurrences() -> None:
    assert LEGAL_TOWER_FAMILIES == ("sigmoid", "tanh", "sech", "hermite")


@pytest.mark.parametrize(
    ("family", "n", "expected"),
    [
        ("sigmoid", 1, sigmoid_poly_coeffs_exact(1)),
        ("sigmoid", 2, sigmoid_poly_coeffs_exact(2)),
        ("tanh", 1, tanh_poly_coeffs_exact(1)),
        ("sech", 2, sech_poly_coeffs_exact(2)),
        ("hermite", 2, hermite_coeffs_exact(2)),
    ],
)
def test_tower_coeffs_match_verified(family: str, n: int, expected: tuple[int, ...]) -> None:
    assert tower_coeffs(family, n) == expected


def test_tower_coeffs_unknown_family_raises() -> None:
    with pytest.raises(ValueError, match="unknown tower family"):
        tower_coeffs("mish", 0)


def test_tower_coeffs_negative_order_raises() -> None:
    with pytest.raises(ValueError, match="order n must be"):
        tower_coeffs("sigmoid", -1)


def test_certificate_payload_is_exact_and_sealed() -> None:
    cert = tower_coeffs_certificate("sigmoid", 2)
    assert cert["payload"]["type"] == "tower_coeffs"
    assert cert["payload"]["family"] == "sigmoid"
    assert cert["payload"]["n"] == 2
    assert cert["payload"]["coeffs"] == [0, 1, -3, 2]
    assert "digest" in cert
    assert cert["honesty"]["unproven_claim"] is False
