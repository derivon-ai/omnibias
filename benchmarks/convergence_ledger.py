# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Wave-3: finite rational convergence ledgers (theory 07-08).

Smoke earns the algebra / emission / honesty gates that do not need
``lake``. G1 kernel pass is asserted in the Lean CI job when the
toolchain is present. ``mathlib_verified`` stays false on the kernel
path. Parent flags are derived, never asserted.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from fractions import Fraction
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.dirname(__file__))
from _common import provenance, write_json  # type: ignore[import-not-found]  # noqa: E402
from _gates import gates_block  # type: ignore[import-not-found]  # noqa: E402
from omnibias.core.proof import generate_obligation, lean_check_available
from omnibias.core.proof.certificate import make_certificate, verify_certificate_digest
from omnibias.core.proof.obligations.convergence_ledger import (
    check_ledger,
    curated_convergence_ledgers,
    empty_premise_discharged_ledger,
    failing_margin_ledger,
    honesty_payload,
    navier_stokes_exponent_ledger,
    seal_ledger_certificate,
    strong_coupling_polymer_ledger,
    unbounded_slope_ledger,
)

SCRATCH = Path(os.environ.get("OMNIBIAS_SCRATCH", "artifacts"))


def _run_g1() -> dict[str, Any]:
    sizes: list[dict[str, Any]] = []
    ok = True
    for ledger in curated_convergence_ledgers():
        report = check_ledger(ledger)
        src = generate_obligation(
            seal_ledger_certificate(ledger, run_lean=False).certificate
        )
        if not report.holds or src is None or "allRatLt" not in src:
            ok = False
        sizes.append(
            {
                "name": ledger.name,
                "n_obligations": len(ledger.obligations),
                "strength": report.strength,
                "holds": report.holds,
            }
        )
    ns = check_ledger(navier_stokes_exponent_ledger())
    tipping = ns.binding_threshold
    tipping_ok = (
        tipping is not None
        and tipping[0] == "kappa"
        and tipping[1] == "lt"
        and tipping[2] == Fraction(1, 200)
    )
    polymer = check_ledger(strong_coupling_polymer_ledger())
    ok = ok and tipping_ok and polymer.holds
    return {
        "name": "g1_algebra_emission",
        "passed": ok,
        "n_ledgers": len(sizes),
        "sizes": sizes,
        "kappa_threshold": "1/200" if tipping_ok else None,
        "detail": "curated ledgers discharge; Lean source is emitted; kappa < 1/200",
    }


def _run_g2() -> dict[str, Any]:
    ledger = failing_margin_ledger()
    report = check_ledger(ledger)
    sealed = seal_ledger_certificate(ledger, run_lean=False)
    src = generate_obligation(sealed.certificate)
    passed = (
        report.holds is False
        and report.strength == "BLOCKED"
        and "particular" in report.failing
        and src is None
    )
    return {
        "name": "g2_negative_control",
        "passed": passed,
        "algebra_holds": report.holds,
        "failing": list(report.failing),
        "detail": "a failing margin is BLOCKED and named; no kernel obligation is emitted",
    }


def _run_g3() -> dict[str, Any]:
    report = seal_ledger_certificate(navier_stokes_exponent_ledger(), run_lean=True)
    sealed = verify_certificate_digest(report.certificate)
    no_forge = "theorem_prover_verified" not in report.certificate.get("honesty", {})
    if lean_check_available():
        passed = bool(report.theorem_prover_verified and sealed and no_forge)
    else:
        passed = (
            report.theorem_prover_verified is False
            and report.lean is not None
            and report.lean.available is False
            and sealed
            and no_forge
        )
    return {
        "name": "g3_no_toolchain_or_kernel",
        "passed": passed,
        "lean_available": lean_check_available(),
        "theorem_prover_verified": report.theorem_prover_verified,
        "mathlib_verified": report.mathlib_verified,
        "wall_seconds": report.wall_seconds,
        "detail": "no lake -> flag stays false; lake -> genuine pass",
    }


def _run_g4() -> dict[str, Any]:
    report = seal_ledger_certificate(navier_stokes_exponent_ledger(), run_lean=False)
    tampered = dict(report.certificate)
    payload = dict(tampered["payload"])
    payload["holds"] = False
    tampered["payload"] = payload
    passed = verify_certificate_digest(report.certificate) and (
        verify_certificate_digest(tampered) is False
    )
    return {
        "name": "g4_tamper_evidence",
        "passed": passed,
        "detail": "editing a sealed field invalidates the digest",
    }


def _run_g5() -> dict[str, Any]:
    ns = seal_ledger_certificate(navier_stokes_exponent_ledger(), run_lean=False)
    polymer = seal_ledger_certificate(strong_coupling_polymer_ledger(), run_lean=False)
    slope = check_ledger(unbounded_slope_ledger(holds=True))
    earned = seal_ledger_certificate(empty_premise_discharged_ledger(), run_lean=False)
    honesty = honesty_payload()
    forged = False
    try:
        make_certificate(
            claim="forged",
            payload={"type": "toy"},
            honesty={"navier_stokes_proof_claim": True},
        )
        forged = True
    except ValueError:
        forged = False
    passed = (
        ns.mathlib_verified is False
        and polymer.mathlib_verified is False
        and ns.certificate["honesty"]["navier_stokes_proof_claim"] is False
        and polymer.certificate["honesty"]["yang_mills_mass_gap_claim"] is False
        and earned.certificate["honesty"]["navier_stokes_proof_claim"] is True
        and slope.holds
        and not forged
        and honesty["navier_stokes_proof_claim"] is False
        and honesty["yang_mills_mass_gap_claim"] is False
    )
    return {
        "name": "g5_parent_flag_unforgeable",
        "passed": passed,
        "mathlib_verified": False,
        "detail": (
            "curated ledgers stay CONDITIONAL; empty premises earn the flag; "
            "a hand-stamped True is refused"
        ),
    }


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args(argv)
    full = bool(args.full)
    artifact = "convergence_ledger.json" if full else "convergence_ledger_smoke.json"
    t0 = time.perf_counter()
    print("G1 algebra...")
    g1 = _run_g1()
    print("G2 negative control...")
    g2 = _run_g2()
    print("G3 toolchain...")
    g3 = _run_g3()
    print("G4 tamper...")
    g4 = _run_g4()
    print("G5 honesty...")
    g5 = _run_g5()
    entries = [g1, g2, g3, g4, g5]
    for e in entries:
        if not e["passed"]:
            raise AssertionError(f"{e['name']} failed: {e}")
    payload = {
        **provenance(
            schema="omnibias.benchmarks.convergence_ledger.v1",
            config={
                "family": "convergence_ledger",
                "full": full,
        "collapse": "none",
                "honesty": honesty_payload(),
            },
        ),
        "gates": dict(gates_block(entries)),
        "wall_seconds": time.perf_counter() - t0,
        "n_ledgers": g1["n_ledgers"],
    }
    if full:
        dest = SCRATCH / "training" / "convergence_ledger"
        dest.mkdir(parents=True, exist_ok=True)
        path = dest / artifact
        path.write_text(
            __import__("json").dumps(payload, indent=2) + "\n", encoding="utf-8"
        )
    else:
        path = write_json(artifact, payload)
    print(f"wrote {path}")
    return payload


if __name__ == "__main__":
    main()
