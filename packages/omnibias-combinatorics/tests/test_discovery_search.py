# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Blind Ramsey colouring search and extremal template search."""

from __future__ import annotations

from pathlib import Path

from omnibias.combinatorics.extremal import ExtremalSearchFamily
from omnibias.combinatorics.proofmachine import (
    EXTREMAL_TEMPLATE_SEARCH,
    FAMILY_CATALOG,
    RAMSEY_COLOURING_SEARCH,
    build_combinatorics_machine,
)
from omnibias.combinatorics.ramsey_search import RamseyColouringFamily, RamseyK3UniversalFamily
from omnibias.core.proof import Conjecture, run_discovery


def test_ramsey_k5_search_finds_triangle_free() -> None:
    family = RamseyColouringFamily()
    result = run_discovery(family.statement, family, "score_guided", budget=256)
    assert result.status == "PROVED"
    assert result.check is not None
    assert result.check.payload["honesty"]["erdos_183_claim"] is False
    assert result.check.payload["honesty"]["discovered_by_omnibias"] is True


def test_ramsey_search_module_is_blind() -> None:
    src = Path(__file__).resolve().parents[1] / "src/omnibias/combinatorics/ramsey_search.py"
    text = src.read_text(encoding="utf-8")
    assert "pentagon" not in text
    assert "pentagon_two_colouring" not in text


def test_ramsey_k3_universal_is_disproved() -> None:
    family = RamseyK3UniversalFamily()
    result = run_discovery(family.statement, family, "score_guided", budget=8)
    assert result.status == "DISPROVED"
    assert result.search_incomplete is False


def test_extremal_template_search() -> None:
    family = ExtremalSearchFamily()
    result = run_discovery(family.statement, family, "score_guided", budget=4)
    assert result.status == "PROVED"
    assert result.candidate in ("C4", "C6", "j_template", "k_template")
    assert result.check is not None
    assert result.check.payload["honesty"]["erdos_146_claim"] is False


def test_machine_search_kinds() -> None:
    machine = build_combinatorics_machine()
    assert RAMSEY_COLOURING_SEARCH in FAMILY_CATALOG
    assert EXTREMAL_TEMPLATE_SEARCH in set(machine.kinds())
    verdict = machine.evaluate(Conjecture(name="ext", kind=EXTREMAL_TEMPLATE_SEARCH))
    assert verdict.status == "PROVED"
