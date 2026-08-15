# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Tests for planted named unique-zero certificates."""

from __future__ import annotations

import pytest
from omnibias.formal.named import (
    LEGAL_NAMED_FAMILIES,
    family_selfmap_holds,
    named_zero_certificate,
)


def test_legal_families_are_the_three_plants() -> None:
    assert LEGAL_NAMED_FAMILIES == ("circle_line", "hopf_radial", "ccf_chebyshev")


@pytest.mark.parametrize("family", ["circle_line", "hopf_radial", "ccf_chebyshev"])
def test_locked_selfmap_holds(family: str) -> None:
    assert family_selfmap_holds(family)


def test_unknown_family_raises() -> None:
    with pytest.raises(ValueError, match="unknown named-zero family"):
        named_zero_certificate("dottie")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("family", "center", "radius"),
    [
        ("circle_line", [3, 4], [1, 8]),
        ("hopf_radial", [1, 1], [1, 4]),
        ("ccf_chebyshev", [7, 8], [1, 8]),
    ],
)
def test_certificate_payload_is_exact_and_sealed(
    family: str, center: list[int], radius: list[int]
) -> None:
    cert = named_zero_certificate(family)  # type: ignore[arg-type]
    assert cert["payload"]["type"] == "named_zero"
    assert cert["payload"]["family"] == family
    assert cert["payload"]["center"] == center
    assert cert["payload"]["radius"] == radius
    assert "digest" in cert
    assert cert["honesty"]["unproven_claim"] is False
