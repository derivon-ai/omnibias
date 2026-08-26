# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Proof engine: collapse kinds, reason tree, Lean flag unforged."""

from __future__ import annotations

import pytest
from omnibias.core.collapse import reset_collapse_registry
from omnibias.core.proof import (
    ENGINE_KINDS,
    build_engine_machine,
    lean_check_available,
    prove,
)
from omnibias.core.proof.lean_check import generate_obligation


@pytest.fixture(autouse=True)
def _restore_registry() -> None:
    reset_collapse_registry()
    yield
    reset_collapse_registry()


def test_engine_kinds_are_registered() -> None:
    machine = build_engine_machine()
    assert ENGINE_KINDS == frozenset(machine.kinds())


def test_external_is_blocked_not_a_parent_claim() -> None:
    result = prove("external", {"parent": "rh"})
    assert result.blocked
    assert "not in scope" in result.verdict.detail
    assert result.reason[0].honesty["continuum_parent_inferred"] is False
    assert "BLOCKED is not falsity" in result.explain()


def test_residual_three_way_and_float_refused() -> None:
    proved = prove("residual", {"lo": 0.0, "hi": 0.0})
    assert proved.proved
    disproved = prove("residual", {"lo": 1.0, "hi": 1.0})
    assert disproved.disproved
    blocked = prove("residual", {"lo": -0.1, "hi": 0.1})
    assert blocked.blocked
    assert "not false" in blocked.verdict.detail
    refused = prove("residual", {"value": 0.0})
    assert refused.blocked
    assert "float residual" in refused.verdict.detail


def test_enclosure_sign_and_lean_eligibility() -> None:
    positive = prove("enclosure_sign", {"lo": 0.5, "hi": 2.0}, lean_check=True)
    assert positive.proved
    assert positive.verdict.certificate is not None
    assert generate_obligation(positive.verdict.certificate) is not None
    if not lean_check_available():
        assert positive.verdict.theorem_prover_verified is False
    else:  # pragma: no cover - Lean-equipped environment only
        assert positive.verdict.theorem_prover_verified is True
    fat = prove("enclosure_sign", {"lo": -1.0, "hi": 1.0})
    assert fat.blocked
    assert generate_obligation(fat.verdict.certificate or {}) is None


def test_gap_is_enclosure_not_a_named_collapse() -> None:
    from omnibias.core import collapse as package

    assert not hasattr(package, "gap_collapse")
    tight = prove("gap", {"L": 3.0, "U": 3.0})
    assert tight.proved
    assert tight.reason[0].honesty["enclosure_collapse"] is True
    fat = prove("gap", {"L": 0.0, "U": 1.0})
    assert fat.blocked
    wrong = prove("gap", {"L": 0.0, "U": 1.0, "expected": 2.0})
    assert wrong.disproved


def test_identity_proves_and_disproves() -> None:
    proved = prove(
        "identity",
        {"left": [1, 2, 1], "right": [1, 2, 1], "domain": [-2.0, 2.0]},
        lean_check=True,
    )
    assert proved.proved
    assert proved.reason[0].surviving == "germ_identity"
    if not lean_check_available():
        assert proved.verdict.theorem_prover_verified is False
    disproved = prove(
        "identity",
        {"left": [1], "right": [2], "domain": [-1.0, 1.0]},
    )
    assert disproved.disproved
    blocked = prove(
        "identity",
        {"left": [0, 1], "right": [0], "domain": [-1.0, 1.0]},
    )
    assert blocked.blocked


def test_winding_pairing_rank() -> None:
    winding = prove("winding", {"coeffs": [0, 1], "expected": 1}, lean_check=True)
    assert winding.proved
    assert winding.reason[0].surviving == 1
    assert winding.verdict.theorem_prover_verified is False
    wrong = prove("winding", {"coeffs": [0, 1], "expected": 0})
    assert wrong.disproved
    pairing = prove(
        "pairing",
        {"residual": [0, 1], "tests": [[1], [1, 0, 1]]},
    )
    assert pairing.proved
    assert pairing.reason[0].honesty["not_a_strong_solution"] is True
    mass = prove("pairing", {"residual": [1], "tests": [[1]]})
    assert mass.disproved
    rank = prove("rank", {"matrix": [[1, 2], [2, 4]]})
    assert rank.proved
    assert rank.reason[0].surviving == "syzygy"
    full = prove("rank", {"matrix": [[1, 0], [0, 1]]})
    assert full.disproved
    floated = prove("rank", {"matrix": [[1.0, 2.0], [2.0, 4.0]]})
    assert floated.blocked


def test_catalog_family_square_and_exhausted_miss() -> None:
    hit = prove(
        "catalog_family",
        {"family": "integer_square", "lo": -3, "hi": 3, "target_square": 4},
    )
    assert hit.proved
    miss = prove(
        "catalog_family",
        {"family": "integer_square", "lo": -1, "hi": 1, "target_square": 4},
    )
    assert miss.blocked
    assert "not a parent" in miss.explain() or "no witness" in miss.verdict.detail


def test_reason_tree_and_unknown_kind() -> None:
    result = prove("residual", {"lo": 0.0, "hi": 0.0}, name="zero-residual")
    assert result.reason
    assert result.reason[0].kind == "residual"
    text = result.explain()
    assert "A float residual is not a proof." in text
    assert "theorem_prover_verified" in text
    unknown = prove("not_a_kind", {})
    assert unknown.blocked
    assert unknown.verdict.prover == "<none>"


def test_certificate_does_not_assert_lean_flag() -> None:
    result = prove("residual", {"lo": 0.0, "hi": 0.0})
    cert = result.verdict.certificate
    assert cert is not None
    honesty = cert["honesty"]
    assert "theorem_prover_verified" not in honesty
    assert honesty["float_residual_is_proof"] is False
