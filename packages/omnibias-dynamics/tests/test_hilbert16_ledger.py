# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
"""Regression tests for the machine-checked Hilbert-16 obligation ledger.

The central guard: on the shipped ledger every derived parent flag
(``full_hilbert16_solved``, ``hilbert16_part_a_solved``,
``hilbert16_part_b_quadratic_solved``) is false, and no tampered payload can
forge one true without genuinely discharging the underlying obligations.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

import pytest
from omnibias.core.proof.certificate import seal_certificate
from omnibias.dynamics.hilbert16_ledger import (
    DRR_CASES,
    DRR_PUBLISHED_CORPUS,
    GATE_NAMES,
    QUADRATIC_GATE_NAMES,
    H16Ledger,
    H16Obligation,
    certify_h16_ledger,
    check_ledger,
    default_h16_ledger,
    derived_parent_flags,
    payload_earns_parent_claim,
    verify_h16_ledger,
)


def test_default_ledger_every_derived_parent_flag_is_false() -> None:
    ledger = default_h16_ledger()
    flags = derived_parent_flags(ledger)
    assert flags["full_hilbert16_solved"] is False
    assert flags["hilbert16_part_a_solved"] is False
    assert flags["hilbert16_part_b_quadratic_solved"] is False
    assert flags["h16_uniform_finiteness_proved"] is False


def test_default_ledger_has_every_gate_and_drr_case_and_is_blocked_or_conditional() -> None:
    ledger = default_h16_ledger()
    gate_names = {e.name for e in ledger.gates()}
    case_names = {e.name for e in ledger.drr_cases()}
    assert gate_names == set(GATE_NAMES)
    assert case_names == set(DRR_CASES)
    assert len(ledger.gates_quadratic()) == len(QUADRATIC_GATE_NAMES)
    assert len(ledger.part_a()) == 5
    assert ledger.by_name(DRR_PUBLISHED_CORPUS).status == "CONDITIONAL"
    result = check_ledger(ledger)
    assert result.status in ("BLOCKED", "CONDITIONAL")
    assert result.status == "BLOCKED"


def test_certify_and_verify_h16_ledger_round_trip() -> None:
    ledger = default_h16_ledger()
    certificate = certify_h16_ledger(ledger)
    assert verify_h16_ledger(certificate)
    assert certificate.parent_flags["full_hilbert16_solved"] is False
    assert certificate.seal["honesty"]["full_hilbert16_solved"] is False
    assert certificate.seal["honesty"]["physical_return_membership_proved"] is False


def test_h16_ledger_certificate_tamper_rejected() -> None:
    ledger = default_h16_ledger()
    certificate = certify_h16_ledger(ledger)
    assert not verify_h16_ledger(replace(certificate, source_digest="tampered"))
    altered = deepcopy(certificate.seal)
    altered["payload"]["parent_flags"]["full_hilbert16_solved"] = True
    altered = seal_certificate(altered)
    assert not verify_h16_ledger(replace(certificate, seal=altered))


def test_payload_earns_parent_claim_cannot_be_forged_by_a_stored_boolean() -> None:
    ledger = default_h16_ledger()
    certificate = certify_h16_ledger(ledger)
    payload = certificate.seal["payload"]
    # The stored payload's parent_flags are already honest (all False); a
    # forged True must still be rejected because payload_earns_parent_claim
    # re-derives the flag from the entries, ignoring any stored boolean.
    forged_payload = deepcopy(payload)
    forged_payload["parent_flags"]["full_hilbert16_solved"] = True
    assert not payload_earns_parent_claim(forged_payload, "full_hilbert16_solved")
    assert not payload_earns_parent_claim(payload, "full_hilbert16_solved")
    assert not payload_earns_parent_claim(payload, "not_a_parent_key")
    assert not payload_earns_parent_claim({"type": "wrong_type"}, "full_hilbert16_solved")


def test_a_fully_discharged_synthetic_ledger_genuinely_earns_the_parent_flag() -> None:
    # A minimal synthetic ledger with every gate, every DRR case, and Part A
    # fully DISCHARGED with no external premises must earn every flag --
    # confirming the derivation is genuinely reachable, not vacuously false.
    from omnibias.dynamics.hilbert16_uniform_ledger import PHASE_NAMES

    entries = tuple(
        H16Obligation(name, f"synthetic discharge of {name}", "DISCHARGED")
        for name in (*GATE_NAMES, *DRR_CASES, DRR_PUBLISHED_CORPUS, *PHASE_NAMES)
    )
    entries += tuple(
        H16Obligation(f"part_a_{tag}", f"synthetic Part-A discharge {tag}", "DISCHARGED")
        for tag in ("wide", "sibling", "four", "archive", "arbitrary")
    )
    ledger = H16Ledger(entries)
    flags = derived_parent_flags(ledger)
    assert flags["hilbert16_part_a_solved"] is True
    assert flags["hilbert16_part_b_quadratic_solved"] is True
    assert flags["h16_uniform_finiteness_proved"] is True
    assert flags["full_hilbert16_solved"] is True
    certificate = certify_h16_ledger(ledger)
    assert verify_h16_ledger(certificate)
    payload = certificate.seal["payload"]
    assert payload_earns_parent_claim(payload, "full_hilbert16_solved")


def test_enrich_ledger_attaches_campaign_evidence_without_changing_flags() -> None:
    from omnibias.dynamics.hilbert16_ledger import enrich_ledger_with_campaign_evidence

    ledger = enrich_ledger_with_campaign_evidence(default_h16_ledger())
    df2a = ledger.by_name("local_df2a_declared_replay")
    inventory = ledger.by_name("local_drr_graphic_inventory")
    assert df2a.evidence_digest
    assert inventory.evidence_digest
    assert not any(derived_parent_flags(ledger).values())


def test_part_a_polygon_sos_audit_is_local_scope_only() -> None:
    ledger = default_h16_ledger()
    entry = ledger.by_name("local_part_a_polygon_sos")
    assert entry.status == "DISCHARGED_LOCAL_SCOPE"
    assert entry.external_premises == ()
    flags = derived_parent_flags(ledger)
    assert flags["hilbert16_part_a_solved"] is False
    assert flags["full_hilbert16_solved"] is False


def test_discharged_local_scope_never_counts_toward_a_parent_flag() -> None:
    entries = tuple(
        H16Obligation(name, f"synthetic discharge of {name}", "DISCHARGED")
        for name in (*GATE_NAMES, *DRR_CASES)
    )
    # Part A is only DISCHARGED_LOCAL_SCOPE: a genuine local win must not
    # leak into the general parent claim.
    entries += (
        H16Obligation(
            "part_a_22_oval_sibling",
            "a local-scope Part-A win",
            "DISCHARGED_LOCAL_SCOPE",
        ),
    )
    ledger = H16Ledger(entries)
    flags = derived_parent_flags(ledger)
    assert flags["hilbert16_part_a_solved"] is False
    assert flags["full_hilbert16_solved"] is False
    result = check_ledger(ledger)
    assert result.status == "CONDITIONAL"


def test_an_external_premise_blocks_full_discharge_even_when_status_is_discharged() -> None:
    entries = tuple(
        H16Obligation(
            name,
            f"discharge of {name} relying on an external result",
            "DISCHARGED",
            external_premises=("relies on an unformalized external theorem",),
        )
        for name in (*GATE_NAMES, *DRR_CASES)
    )
    entries += (H16Obligation("part_a_x", "Part-A discharge", "DISCHARGED"),)
    ledger = H16Ledger(entries)
    flags = derived_parent_flags(ledger)
    assert flags["hilbert16_part_b_quadratic_solved"] is False
    assert flags["full_hilbert16_solved"] is False
    result = check_ledger(ledger)
    assert result.status == "CONDITIONAL"
    assert len(result.blocking) == len(GATE_NAMES) + len(DRR_CASES)


def test_a_single_blocked_entry_dominates_the_ledger_status() -> None:
    entries = tuple(
        H16Obligation(name, f"discharge of {name}", "DISCHARGED")
        for name in (*GATE_NAMES, *DRR_CASES)
    )
    entries += (H16Obligation("part_a_x", "Part-A", "BLOCKED"),)
    ledger = H16Ledger(entries)
    result = check_ledger(ledger)
    assert result.status == "BLOCKED"
    assert len(result.blocking) == 1
    assert result.blocking[0].name == "part_a_x"


def test_h16_ledger_requires_unique_entry_names() -> None:
    with pytest.raises(ValueError, match="unique"):
        H16Ledger(
            (
                H16Obligation("G1", "a", "BLOCKED"),
                H16Obligation("G1", "b", "BLOCKED"),
            )
        )


def test_h16_obligation_requires_a_name_and_statement() -> None:
    with pytest.raises(ValueError, match="name and a statement"):
        H16Obligation("", "a statement", "BLOCKED")
    with pytest.raises(ValueError, match="name and a statement"):
        H16Obligation("G1", "", "BLOCKED")


def test_h16_obligation_rejects_unknown_status() -> None:
    with pytest.raises(ValueError, match="unknown H16 status"):
        H16Obligation("G1", "a statement", "SOLVED")  # type: ignore[arg-type]


def test_fully_discharged_property_requires_empty_premises() -> None:
    clean = H16Obligation("G1", "a statement", "DISCHARGED")
    assert clean.fully_discharged
    conditional_premise = H16Obligation(
        "G2", "a statement", "DISCHARGED", external_premises=("depends on X",)
    )
    assert not conditional_premise.fully_discharged
    local = H16Obligation("G3", "a statement", "DISCHARGED_LOCAL_SCOPE")
    assert not local.fully_discharged
