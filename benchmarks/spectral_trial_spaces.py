# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Frontier 07-05: spectral trial spaces and certified floors.

Smoke earns G1 (alignment on every bound), G2 (pack at 1/4
dimension stays within 5% on ten localized wells), G3
(three adversarial problems reported, none excluded), G4
(1000 known spectra, zero Temple violations), and G6
(honesty flags stay False; kernel flag cannot be forged).
G5 lives in ``benchmarks/sos_adapted_basis.py``.

Fixed operator, one domain, one discretization. Not a
continuum spectral gap and not a Yang-Mills mass gap.
Jets are founding bias collapse, not temperature collapse.
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
from omnibias.core.proof.certificate import make_certificate
from omnibias.core.verified.trial_spaces import (
    DISCLAIMER,
    adversarial_report,
    certified_floor,
    dimension_reduction_report,
    floor_schema_errors,
    honesty_payload,
    seal_floor,
    sine_trial_space,
    soundness_violations,
    synthetic_diagonal,
)

SCRATCH = Path(os.environ.get("OMNIBIAS_SCRATCH", "artifacts"))


def _run_g1() -> dict[str, Any]:
    matrix, evec, rho = synthetic_diagonal(4, seed=1)
    nodes = [0.0, 1.0, 2.0, 3.0]
    cert = certified_floor(
        matrix, sine_trial_space(1, domain=(0.0, 3.0)), nodes, reference_vector=evec, rho=rho
    )
    ok = (
        0.0 <= cert.alignment <= 1.0
        and "sin_theta" in cert.to_payload()
        and not floor_schema_errors(cert)
    )
    return {"name": "g1_alignment", "passed": ok, "alignment": cert.alignment}


def _run_g2() -> dict[str, Any]:
    rows = dimension_reduction_report()
    ok = len(rows) >= 10 and all(bool(r["within_5pct"]) for r in rows)
    return {
        "name": "g2_dimension_reduction",
        "passed": ok,
        "n": len(rows),
        "wins": sum(1 for r in rows if r["within_5pct"]),
        "pack_alignments": [r["pack_alignment"] for r in rows],
        "condition_numbers": [r["pack_cond"] for r in rows],
    }


def _run_g3() -> dict[str, Any]:
    rows = adversarial_report()
    kinds = {r["kind"] for r in rows}
    ok = len(rows) >= 3 and kinds == {"oscillatory", "corner", "discontinuous"}
    return {
        "name": "g3_adversarial",
        "passed": ok,
        "rows": [{"name": r["name"], "kind": r["kind"], "within_5pct": r["within_5pct"]} for r in rows],
    }


def _run_g4(*, full: bool) -> dict[str, Any]:
    n = 1000 if full else 1000
    misses = soundness_violations(n_problems=n)
    return {"name": "g4_soundness", "passed": misses == 0, "n": n, "misses": misses}


def _run_g6() -> dict[str, Any]:
    matrix, evec, rho = synthetic_diagonal(4, seed=2)
    nodes = [0.0, 1.0, 2.0, 3.0]
    cert = certified_floor(
        matrix, sine_trial_space(1, domain=(0.0, 3.0)), nodes, reference_vector=evec, rho=rho
    )
    sealed = seal_floor(cert)
    forged = False
    try:
        make_certificate(claim="forged", payload={}, honesty={"theorem_prover_verified": True})
    except ValueError:
        forged = True
    ok = (
        forged
        and "theorem_prover_verified" not in sealed["honesty"]
        and honesty_payload()["theorem_prover_verified"] is False
        and sealed["honesty"]["yang_mills_mass_gap"] is False
        and DISCLAIMER in str(cert.to_payload()["disclaimer"])
    )
    return {"name": "g6_honesty", "passed": ok}


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args(argv)
    full = bool(args.full)
    artifact = "spectral_trial_spaces.json" if full else "spectral_trial_spaces_smoke.json"
    t0 = time.perf_counter()
    entries = [_run_g1(), _run_g2(), _run_g3(), _run_g4(full=full), _run_g6()]
    for e in entries:
        print(e["name"], "ok" if e["passed"] else "FAIL")
        if not e["passed"]:
            raise AssertionError(f"{e['name']} failed: {e}")
    payload = {
        **provenance(
            schema="omnibias.benchmarks.spectral_trial_spaces.v1",
            config={"family": "spectral_trial_spaces", "full": full, "honesty": honesty_payload()},
        ),
        "gates": dict(gates_block(entries)),
        "wall_seconds": time.perf_counter() - t0,
    }
    if full:
        dest = SCRATCH / "spectral"
        dest.mkdir(parents=True, exist_ok=True)
        path = dest / artifact
        path.write_text(__import__("json").dumps(payload, indent=2) + "\n", encoding="utf-8")
    else:
        path = write_json(artifact, payload)
    print(f"wrote {path}")
    return payload


if __name__ == "__main__":
    main()
