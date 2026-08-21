# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Frontier 07-02: weak-form NS-adjacent enclosures.

Smoke earns G1 (width report first), G2 (weak quad term 2x
narrower on Taylor-Green), G3 (sound coverage; 1000 configs
on ``--full``, 64 on smoke), G4 (pack W_repr independent of
shear thickness; smooth grows as 1/d), G5 (exact Jacobian
horizon at least 1.5x), and G6 (honesty flags stay False).
Not a continuum Navier-Stokes claim. Jets are founding bias
collapse, not temperature collapse.
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
from omnibias.pinn.certified.weak_form import (
    DISCLAIMER,
    certified_weak_residual,
    enclosure_covers,
    honesty_payload,
    lohner_horizon,
    shear_repr_errors,
    weak_form_schema_errors,
    width_decomposition,
)

SCRATCH = Path(os.environ.get("OMNIBIAS_SCRATCH", "artifacts"))


def _run_g1() -> dict[str, Any]:
    cert = certified_weak_residual(form="weak")
    report = width_decomposition(cert)
    ok = report.dominant == "quad" and not weak_form_schema_errors(cert)
    return {"name": "g1_width_report", "passed": ok, "report": report.to_payload()}


def _run_g2() -> dict[str, Any]:
    strong = width_decomposition(certified_weak_residual(form="strong"))
    weak = width_decomposition(certified_weak_residual(form="weak"))
    ratio = strong.w_quad / max(weak.w_quad, 1e-30)
    ok = strong.dominant == "quad" and ratio >= 2.0
    return {
        "name": "g2_weak_narrower",
        "passed": ok,
        "ratio": ratio,
        "strong_dominant": strong.dominant,
        "weak_dominant": weak.dominant,
    }


def _run_g3(*, full: bool) -> dict[str, Any]:
    n = 1000 if full else 64
    misses = enclosure_covers(n=n, form="weak") + enclosure_covers(n=n, form="strong")
    return {"name": "g3_coverage", "passed": misses == 0, "n": n, "misses": misses}


def _run_g4() -> dict[str, Any]:
    ds = (1e-1, 3e-2, 1e-2, 3e-3, 1e-3)
    err = shear_repr_errors(ds)
    pack_flat = all(p == 0.0 for p in err["pack"])
    grew = err["smooth"][-1] / max(err["smooth"][0], 1e-30) >= (ds[0] / ds[-1]) * 0.5
    return {
        "name": "g4_shear",
        "passed": pack_flat and grew,
        "smooth": err["smooth"],
        "pack": err["pack"],
        "conditioning": err["conditioning"],
    }


def _run_g5() -> dict[str, Any]:
    exact = lohner_horizon(exact_jac=True)
    fd = lohner_horizon(exact_jac=False)
    ratio = exact / max(fd, 1)
    return {"name": "g5_horizon", "passed": ratio >= 1.5, "exact": exact, "fd": fd, "ratio": ratio}


def _run_g6() -> dict[str, Any]:
    cert = certified_weak_residual()
    honesty = cert["honesty"]
    ok = (
        honesty["continuum_navier_stokes_claim"] is False
        and honesty["unproven_claim"] is False
        and honesty["three_d_claim"] is False
        and honesty_payload()["theorem_prover_verified"] is False
        and DISCLAIMER in str(cert["payload"]["disclaimer"])
    )
    return {"name": "g6_honesty", "passed": ok}


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args(argv)
    full = bool(args.full)
    artifact = "ns_weak_form_enclosure.json" if full else "ns_weak_form_enclosure_smoke.json"
    t0 = time.perf_counter()
    entries = [_run_g1(), _run_g2(), _run_g3(full=full), _run_g4(), _run_g5(), _run_g6()]
    for e in entries:
        print(e["name"], "ok" if e["passed"] else "FAIL")
        if not e["passed"]:
            raise AssertionError(f"{e['name']} failed: {e}")
    payload = {
        **provenance(
            schema="omnibias.benchmarks.ns_weak_form_enclosure.v1",
            config={"family": "ns_weak_form_enclosure", "full": full, "honesty": honesty_payload()},
        ),
        "gates": dict(gates_block(entries)),
        "wall_seconds": time.perf_counter() - t0,
    }
    if full:
        dest = SCRATCH / "ns_adjacent"
        dest.mkdir(parents=True, exist_ok=True)
        path = dest / artifact
        path.write_text(__import__("json").dumps(payload, indent=2) + "\n", encoding="utf-8")
    else:
        path = write_json(artifact, payload)
    print(f"wrote {path}")
    return payload


if __name__ == "__main__":
    main()
