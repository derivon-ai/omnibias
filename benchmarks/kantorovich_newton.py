# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Wave-3 trainer: Kantorovich-accepted Newton (theory 08-04).

Smoke earns G1 (100% accept on F(x)=x^2-2 at x=1.5; 0% at x=3.0),
G2 (sealed payload continuum_pde_claim is false; claim names a finite
map), and G3 (theorem_prover_verified is absent). Empty ball is a
valid reject. The ball is not a continuum PDE theorem. Bias collapse
may tighten DF via exact sigma'; this gate does not use it.
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
from omnibias.core.proof.certificate import THEOREM_PROVER_VERIFIED_KEY
from omnibias.core.verified.kantorovich import (
    CONTINUUM_PDE_CLAIM_KEY,
    FINITE_RESIDUAL_CLAIM,
    kantorovich_accept_step,
    polynomial_sqrt2_maps,
)

SCRATCH = Path(os.environ.get("OMNIBIAS_SCRATCH", "artifacts"))


def _run_g1() -> dict[str, Any]:
    func, jac, lip = polynomial_sqrt2_maps()
    good = kantorovich_accept_step(
        func, jac, [[1.0 / 3.0]], [1.5], lipschitz_df=lip, r_max=0.2
    )
    far = kantorovich_accept_step(
        func, jac, [[1.0 / 3.0]], [3.0], lipschitz_df=lip, r_max=0.2
    )
    accept_good = 1.0 if good.accepted else 0.0
    accept_far = 1.0 if far.accepted else 0.0
    passed = accept_good == 1.0 and accept_far == 0.0
    return {
        "name": "g1_accept_reject",
        "passed": passed,
        "accept_rate_good": accept_good,
        "accept_rate_far": accept_far,
        "good_reason": good.reason,
        "far_reason": far.reason,
        "good_radius": None if good.certificate is None else good.certificate.radius,
    }


def _run_g2() -> dict[str, Any]:
    func, jac, lip = polynomial_sqrt2_maps()
    decision = kantorovich_accept_step(
        func, jac, [[1.0 / 3.0]], [1.5], lipschitz_df=lip, r_max=0.2
    )
    sealed = None if decision.certificate is None else decision.certificate.certificate
    payload = {} if sealed is None else dict(sealed.get("payload", {}))
    honesty = {} if sealed is None else dict(sealed.get("honesty", {}))
    claim = "" if sealed is None else str(sealed.get("claim", ""))
    passed = (
        decision.accepted
        and payload.get(CONTINUUM_PDE_CLAIM_KEY) is False
        and honesty.get(CONTINUUM_PDE_CLAIM_KEY) is False
        and "finite-dimensional" in claim
        and claim == FINITE_RESIDUAL_CLAIM
    )
    return {
        "name": "g2_never_continuum",
        "passed": bool(passed),
        "continuum_pde_claim": payload.get(CONTINUUM_PDE_CLAIM_KEY),
        "claim": claim,
    }


def _run_g3() -> dict[str, Any]:
    func, jac, lip = polynomial_sqrt2_maps()
    decision = kantorovich_accept_step(
        func, jac, [[1.0 / 3.0]], [1.5], lipschitz_df=lip, r_max=0.2
    )
    honesty = (
        {}
        if decision.certificate is None
        else dict(decision.certificate.certificate.get("honesty", {}))
    )
    present = THEOREM_PROVER_VERIFIED_KEY in honesty
    return {
        "name": "g3_no_forge",
        "passed": not present,
        "theorem_prover_verified_in_honesty": present,
        "note": "radii certificate is the sound-enclosure tier",
    }


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args(argv)
    full = bool(args.full)
    artifact = "kantorovich_newton.json" if full else "kantorovich_newton_smoke.json"
    t0 = time.perf_counter()
    print("G1 accept/reject...")
    g1 = _run_g1()
    print("G2 never continuum...")
    g2 = _run_g2()
    print("G3 no forge...")
    g3 = _run_g3()
    entries = [g1, g2, g3]
    for e in entries:
        if not e["passed"]:
            raise AssertionError(f"{e['name']} failed: {e}")
    gates = dict(gates_block(entries))
    payload = {
        **provenance(
            schema="omnibias.benchmarks.kantorovich_newton.v1",
            config={
                "family": "kantorovich_newton",
                "full": full,
                "honesty": {
                    "navier_stokes_proof_claim": False,
                    "ccf_stretch_cleared": False,
                    "continuum_pde_claim": False,
                    "theorem_prover_verified": False,
                    "temperature_collapse": False,
                },
            },
        ),
        "gates": gates,
        "wall_seconds": time.perf_counter() - t0,
    }
    if full:
        dest = SCRATCH / "training" / "kantorovich_newton"
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
