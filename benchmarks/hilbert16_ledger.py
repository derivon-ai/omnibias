#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""The machine-checked Hilbert-16 obligation ledger (item 4 of the plan).

Four gates replay omnibias.dynamics.hilbert16_ledger: the shipped ledger
is exactly BLOCKED/CONDITIONAL with every derived parent flag false (GH1),
the sealed certificate round-trips and rejects tampering (GH2), a stored
payload cannot forge a parent claim (GH3), and a synthetic, fully-discharged
ledger -- built only for this gate, never the shipped one -- genuinely earns
full_hilbert16_solved while a DISCHARGED_LOCAL_SCOPE entry alone never does
(GH4). Names use the ``gh`` prefix so they do not collide with research
gates ``G1``-``G6``. This benchmark proves the ledger machinery is sound;
it proves nothing about Hilbert's 16th problem itself.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from copy import deepcopy
from dataclasses import replace
from typing import Any

sys.path.insert(0, os.path.dirname(__file__))
from _common import provenance, write_json  # type: ignore[import-not-found]  # noqa: E402
from _gates import gates_block  # type: ignore[import-not-found]  # noqa: E402
from omnibias.core.proof.certificate import seal_certificate  # noqa: E402
from omnibias.dynamics.hilbert16_ledger import (  # noqa: E402
    H16Ledger,
    H16Obligation,
    certify_h16_ledger,
    check_ledger,
    default_h16_ledger,
    derived_parent_flags,
    payload_earns_parent_claim,
    verify_h16_ledger,
)
from omnibias.dynamics.hilbert16_uniform_ledger import PHASE_NAMES  # noqa: E402


def _gh1_shipped_ledger_is_honestly_open() -> dict[str, Any]:
    ledger = default_h16_ledger()
    result = check_ledger(ledger)
    flags = derived_parent_flags(ledger)
    ok = (
        result.status in ("BLOCKED", "CONDITIONAL")
        and len(ledger.gates()) == 7
        and len(ledger.drr_cases()) == 15
        and len(ledger.part_a()) == 5
        and ledger.by_name("drr_published_corpus").status == "CONDITIONAL"
        and not any(flags.values())
    )
    return {
        "name": "gh1_shipped_ledger_open",
        "passed": bool(ok),
        "status": result.status,
        "n_blocking": len(result.blocking),
        "parent_flags": flags,
        "detail": "every gate/DRR/part-A entry present; every derived parent flag false",
    }


def _gh2_certificate_round_trip_and_tamper() -> dict[str, Any]:
    ledger = default_h16_ledger()
    certificate = certify_h16_ledger(ledger)
    round_trip = verify_h16_ledger(certificate)
    tampered_digest = not verify_h16_ledger(replace(certificate, source_digest="tampered"))
    altered = deepcopy(certificate.seal)
    altered["payload"]["result"]["status"] = "PROVED"
    altered = seal_certificate(altered)
    tampered_seal = not verify_h16_ledger(replace(certificate, seal=altered))
    ok = round_trip and tampered_digest and tampered_seal
    return {
        "name": "gh2_certificate_round_trip",
        "passed": bool(ok),
        "round_trip": round_trip,
        "tampered_digest_rejected": tampered_digest,
        "tampered_seal_rejected": tampered_seal,
        "detail": "sealed ledger snapshot replays and rejects every tamper",
    }


def _gh3_payload_cannot_forge_a_parent_claim() -> dict[str, Any]:
    ledger = default_h16_ledger()
    certificate = certify_h16_ledger(ledger)
    payload = certificate.seal["payload"]
    genuine = payload_earns_parent_claim(payload, "full_hilbert16_solved")
    forged = deepcopy(payload)
    forged["parent_flags"] = {**forged["parent_flags"], "full_hilbert16_solved": True}
    forgery_rejected = not payload_earns_parent_claim(forged, "full_hilbert16_solved")
    unknown_key_rejected = not payload_earns_parent_claim(payload, "not_a_real_flag")
    ok = genuine is False and forgery_rejected and unknown_key_rejected
    return {
        "name": "gh3_parent_claim_unforgeable",
        "passed": bool(ok),
        "genuine_flag": genuine,
        "forgery_rejected": forgery_rejected,
        "detail": "the flag is re-derived from entries, never trusted from a stored boolean",
    }


def _discharged(name: str) -> H16Obligation:
    return H16Obligation(name, f"synthetic discharge of {name}", "DISCHARGED", (), source="benchmark")


def _gh4_synthetic_fully_discharged_ledger_earns_the_flag() -> dict[str, Any]:
    gate_names = ("G1", "G2", "G3", "G4", "G5", "G6a", "G6b")
    case_names = (
        "I_2^1", "I_4^1", "I_12^1", "I_13^1", "I_14^1", "I_6b^1",
        "H_13^3", "DI_2b", "H_14^3", "DF_1a", "DF_1b", "DF_2a", "DF_2b", "DH_1", "DH_2",
    )
    discharged_gates = tuple(_discharged(name) for name in gate_names)
    discharged_cases = tuple(_discharged(name) for name in case_names)
    discharged_cases += (_discharged("drr_published_corpus"),)
    discharged_phases = tuple(_discharged(name) for name in PHASE_NAMES)
    part_a = tuple(
        _discharged(name)
        for name in (
            "part_a_22_oval_wide_deep",
            "part_a_22_oval_sibling",
            "part_a_octic_four_tcurve_excluded",
            "part_a_degree_8_archive",
            "part_a_arbitrary_degree",
        )
    )
    full_ledger = H16Ledger(
        (*discharged_gates, *discharged_cases, *discharged_phases, *part_a)
    )
    full_flags = derived_parent_flags(full_ledger)

    local_g1 = H16Obligation(
        "G1", "synthetic local-only discharge of G1", "DISCHARGED_LOCAL_SCOPE", (), source="benchmark"
    )
    local_only = H16Ledger(
        (
            local_g1,
            *discharged_gates[1:],
            *discharged_cases,
            *discharged_phases,
            *part_a,
        )
    )
    local_flags = derived_parent_flags(local_only)

    ok = (
        full_flags["full_hilbert16_solved"] is True
        and full_flags["hilbert16_part_a_solved"] is True
        and full_flags["hilbert16_part_b_quadratic_solved"] is True
        and local_flags["full_hilbert16_solved"] is False
    )
    return {
        "name": "gh4_synthetic_full_discharge_earns_flag",
        "passed": bool(ok),
        "full_flags": full_flags,
        "local_only_flags": local_flags,
        "detail": "a synthetic all-discharged ledger earns the flag; DISCHARGED_LOCAL_SCOPE alone does not",
    }


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args(argv)
    full = bool(args.full)
    artifact = "hilbert16_ledger.json" if full else "hilbert16_ledger_smoke.json"
    t0 = time.perf_counter()
    entries = [
        _gh1_shipped_ledger_is_honestly_open(),
        _gh2_certificate_round_trip_and_tamper(),
        _gh3_payload_cannot_forge_a_parent_claim(),
        _gh4_synthetic_fully_discharged_ledger_earns_the_flag(),
    ]
    for entry in entries:
        print(entry["name"], "ok" if entry["passed"] else "FAIL")
        if not entry["passed"]:
            raise AssertionError(f"{entry['name']} failed: {entry}")
    shipped = default_h16_ledger()
    shipped_flags = derived_parent_flags(shipped)
    payload = {
        **provenance(
            schema="omnibias.benchmarks.hilbert16_ledger.v1",
            config={"family": "hilbert16_ledger", "full": full},
        ),
        "gates": dict(gates_block(entries)),
        "wall_seconds": time.perf_counter() - t0,
        "shipped_ledger": {
            "status": check_ledger(shipped).status,
            "n_entries": len(shipped.entries),
            "parent_flags": shipped_flags,
        },
        "honesty": {
            **shipped_flags,
            "drr_case_closed": False,
        },
        "disclaimer": (
            "The obligation ledger and its derived parent flags are sound "
            "machinery, replayed here on both the shipped (open) ledger and a "
            "synthetic fully-discharged one built only for gate GH4. This is "
            "not a claim that Hilbert's 16th problem, or any of its gates or "
            "DRR cases, is solved."
        ),
    }
    path = write_json(artifact, payload)
    print(f"wrote {path}")
    return payload


if __name__ == "__main__":
    main()
