# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Wave-3: rational stencil Lean obligations (theory 01-11).

Smoke earns the algebra / emission / honesty gates that do not need
``lake``. G1 kernel pass and G2 lake-fail are asserted in the Lean CI
job when the toolchain is present. ``mathlib_verified`` stays false.
Lean certifies the algebra; it does not state a collapse.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.dirname(__file__))
from _common import provenance, write_json  # type: ignore[import-not-found]  # noqa: E402
from _gates import gates_block  # type: ignore[import-not-found]  # noqa: E402
from omnibias.core.proof import (
    generate_obligation,
    lean_check_available,
    seal_poisedness_certificate,
    seal_stencil_certificate,
)
from omnibias.core.proof.certificate import verify_certificate_digest
from omnibias.core.proof.obligations.rational_stencil import (
    birkhoff_spec_stencil,
    curated_rational_stencils,
    honesty_payload,
    stencil_consistency_obligation,
)

SCRATCH = Path(os.environ.get("OMNIBIAS_SCRATCH", "artifacts"))


def _run_g1_algebra() -> dict[str, Any]:
    stencils = curated_rational_stencils()
    sizes: list[dict[str, Any]] = []
    ok = len(stencils) >= 20
    for st in stencils:
        obl = stencil_consistency_obligation(st)
        src = generate_obligation(
            seal_stencil_certificate(st, run_lean=False).certificate
        )
        if not obl.holds or src is None:
            ok = False
        sizes.append(
            {
                "name": st.name,
                "n": st.n_conditions,
                "max_abs_int": obl.max_abs_int,
                "holds": obl.holds,
            }
        )
    return {
        "name": "g1_algebra_emission",
        "passed": ok,
        "n_stencils": len(stencils),
        "sizes": sizes,
        "detail": "curated C_j identities hold and Lean source is emitted",
    }


def _run_g2() -> dict[str, Any]:
    bad = birkhoff_spec_stencil().corrupt_first_weight()
    report = seal_stencil_certificate(bad, run_lean=False)
    src = generate_obligation(report.certificate)
    passed = report.obligation.holds is False and src is not None
    lake_failed: bool | None = None
    if lean_check_available():
        live = seal_stencil_certificate(bad, run_lean=True)
        lake_failed = live.theorem_prover_verified is False
        passed = passed and lake_failed
    return {
        "name": "g2_negative_control",
        "passed": passed,
        "algebra_holds": report.obligation.holds,
        "lake_failed": lake_failed,
        "detail": "corrupted weights fail C_j; lake rejects when present",
    }


def _run_g3() -> dict[str, Any]:
    report = seal_stencil_certificate(birkhoff_spec_stencil(), run_lean=True)
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
        "wall_seconds": report.wall_seconds,
        "detail": "no lake -> flag stays false; lake -> genuine pass",
    }


def _run_g4() -> dict[str, Any]:
    report = seal_stencil_certificate(birkhoff_spec_stencil(), run_lean=False)
    tampered = dict(report.certificate)
    payload = dict(tampered["payload"])
    payload["target_order"] = 99
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
    st = seal_stencil_certificate(birkhoff_spec_stencil(), run_lean=True)
    po = seal_poisedness_certificate(birkhoff_spec_stencil(), run_lean=True)
    honesty = honesty_payload()
    passed = (
        st.mathlib_verified is False
        and po.mathlib_verified is False
        and honesty["mathlib_path_used"] is False
        and honesty["collapse_limit_in_lean"] is False
    )
    return {
        "name": "g5_tier_separation",
        "passed": passed,
        "mathlib_verified": False,
        "detail": "mathlib_verified stays false; collapse is not in Lean",
    }


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args(argv)
    full = bool(args.full)
    artifact = "rational_stencil.json" if full else "rational_stencil_smoke.json"
    t0 = time.perf_counter()
    print("G1 algebra...")
    g1 = _run_g1_algebra()
    print("G2 negative control...")
    g2 = _run_g2()
    print("G3 toolchain...")
    g3 = _run_g3()
    print("G4 tamper...")
    g4 = _run_g4()
    print("G5 tier...")
    g5 = _run_g5()
    entries = [g1, g2, g3, g4, g5]
    for e in entries:
        if not e["passed"]:
            raise AssertionError(f"{e['name']} failed: {e}")
    payload = {
        **provenance(
            schema="omnibias.benchmarks.rational_stencil.v1",
            config={
                "family": "rational_stencil",
                "full": full,
                "collapse": "bias_collapse",
                "honesty": honesty_payload(),
            },
        ),
        "gates": dict(gates_block(entries)),
        "wall_seconds": time.perf_counter() - t0,
        "max_abs_int": max(row["max_abs_int"] for row in g1["sizes"]),
    }
    if full:
        dest = SCRATCH / "training" / "rational_stencil"
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
