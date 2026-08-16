# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Tests for planted Weyl-volume prefactor certificates."""

from __future__ import annotations

import pytest
from omnibias.formal.haar import (
    LEGAL_HAAR_FAMILIES,
    family_facts_hold,
    haar_certificate,
)


def test_legal_family_is_the_prefactor_plant() -> None:
    assert LEGAL_HAAR_FAMILIES == ("weyl_prefactor_24",)


def test_locked_facts_hold() -> None:
    assert family_facts_hold("weyl_prefactor_24")


def test_unknown_family_raises() -> None:
    with pytest.raises(ValueError, match="unknown haar family"):
        haar_certificate("dottie")  # type: ignore[arg-type]


def test_payload_is_exact_and_sealed() -> None:
    cert = haar_certificate("weyl_prefactor_24")
    assert cert["payload"]["type"] == "haar_volume"
    assert cert["payload"]["family"] == "weyl_prefactor_24"
    assert cert["payload"]["value"] == 24
    assert "digest" in cert
