# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Case A (b11, b21, b31) leftover over Q. Proof-claim stays False."""

from __future__ import annotations

import pytest
from omnibias.core.collapse import reset_collapse_registry
from omnibias.core.proof import Conjecture, catalog_entry, discover, prove
from omnibias.holonomic.jacobian_n2 import (
    JACOBIAN_CONJECTURE_PROOF_CLAIM_ALLOWED,
    JACOBIAN_N2_PARENT,
)
from omnibias.holonomic.jacobian_n2_case_a import (
    CASE_A_B31_KIND,
    case_a_b31_generators,
    case_a_b31_identity_payloads,
    case_a_b31_statement,
    named_case_a_b31_generators,
    replay_case_a_b31,
    seal_case_a_b31,
)
from omnibias.holonomic.proofmachine import (
    JACOBIAN_N2_CASE_A_B31,
    _schema_errors,
    build_holonomic_machine,
)


@pytest.fixture(autouse=True)
def _restore_registry() -> None:
    reset_collapse_registry()
    yield
    reset_collapse_registry()


_PLANTED_LINE = (
    [[[1, 0, 0], "1"], [[0, 0, 0], "-1"]],
    [[[0, 1, 0], "1"]],
    [[[0, 0, 1], "1"]],
)


def test_statement_is_universal_and_parent_stays_open() -> None:
    statement = case_a_b31_statement()
    assert statement.existential is False
    assert statement.parent == JACOBIAN_N2_PARENT
    assert statement.parent_status == "open"
    assert statement.name == CASE_A_B31_KIND
    assert "only at the origin" in statement.obligation
    assert "not jacobian_conjecture_n2" in statement.obligation


def test_extract_replays_named_generators() -> None:
    assert case_a_b31_generators() == named_case_a_b31_generators()


def test_identity_laws_prove_via_engine() -> None:
    for payload in case_a_b31_identity_payloads():
        result = prove("identity", payload)
        assert result.proved
        assert result.reason[0].honesty["continuum_parent_inferred"] is False


def test_identity_disproves_a_false_law() -> None:
    result = prove("identity", {"left": [1], "right": [0], "domain": [-1.0, 1.0]})
    assert result.disproved


def test_residual_blocks_a_float() -> None:
    result = prove("residual", {"value": 0.0})
    assert result.blocked
    assert "float residual" in result.verdict.detail


def test_seal_proves_origin_only() -> None:
    verdict = seal_case_a_b31()
    assert verdict.proved
    honesty = verdict.certificate["honesty"]
    assert honesty["jacobian_conjecture_proof_claim"] is False
    assert honesty["jacobian_n2_claim"] is False
    assert honesty["continuum_parent_inferred"] is False
    assert "theorem_prover_verified" not in honesty
    payload = verdict.certificate["payload"]
    assert payload["parent"] == JACOBIAN_N2_PARENT
    assert payload["parent_status"] == "open"
    assert payload["surviving"] == "origin"
    assert JACOBIAN_CONJECTURE_PROOF_CLAIM_ALLOWED is False


def test_planted_nonzero_q_point_disproves() -> None:
    verdict = seal_case_a_b31(
        {"generators": list(_PLANTED_LINE), "witness": ["1", "0", "0"]}
    )
    assert verdict.disproved
    assert verdict.certificate["honesty"]["jacobian_n2_claim"] is False
    assert verdict.certificate["honesty"]["jacobian_conjecture_proof_claim"] is False
    assert verdict.certificate["payload"]["parent_status"] == "open"


def test_float_witness_is_blocked() -> None:
    verdict = seal_case_a_b31({"witness": [1.0, 0, 0]})
    assert verdict.blocked
    assert "float residual" in verdict.detail
    assert verdict.certificate["payload"]["parent_status"] == "open"


def test_replay_and_forged_proof_claim() -> None:
    sealed = seal_case_a_b31()
    assert replay_case_a_b31(sealed.certificate) is True
    forged = {
        **sealed.certificate,
        "honesty": {
            **sealed.certificate["honesty"],
            "jacobian_conjecture_proof_claim": True,
        },
    }
    with pytest.raises(ValueError):
        replay_case_a_b31(forged)


def test_catalog_and_machine() -> None:
    entry = catalog_entry(JACOBIAN_N2_CASE_A_B31)
    assert entry is not None
    assert entry.parent == JACOBIAN_N2_PARENT
    assert entry.parent_status == "open"
    assert entry.mode == "exact_replay"
    found = discover(JACOBIAN_N2_CASE_A_B31)
    assert found["status"] == "PROVED"
    assert found["honesty"]["jacobian_conjecture_proof_claim"] is False
    machine = build_holonomic_machine()
    verdict = machine.evaluate(
        Conjecture(name="case-a-b31", kind=JACOBIAN_N2_CASE_A_B31, data={}),
        replay=True,
    )
    assert verdict.proved
    assert verdict.certificate is not None
    assert verdict.certificate["honesty"]["jacobian_n2_claim"] is False
    assert verdict.theorem_prover_verified is False
    assert _schema_errors({"honesty": {"jacobian_conjecture_proof_claim": True}})
    blocked = machine.evaluate(
        Conjecture(
            name="float",
            kind=JACOBIAN_N2_CASE_A_B31,
            data={"witness": [1.0, 0, 0]},
        )
    )
    assert blocked.blocked
    disproved = machine.evaluate(
        Conjecture(
            name="planted",
            kind=JACOBIAN_N2_CASE_A_B31,
            data={"generators": list(_PLANTED_LINE), "witness": [1, 0, 0]},
        )
    )
    assert disproved.disproved
