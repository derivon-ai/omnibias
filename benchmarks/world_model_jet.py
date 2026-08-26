# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Frontier 09-25: world-model-as-jet.

G1 is the order-2 oscillator Taylor. G2 is a sound Lohner box.
G3 records no NS / continuum claim. G4 is Lohner-path purity
(no torch/jax).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any

from omnibias.core.jet_world import (
    DISCLAIMER,
    honesty_payload,
    jet_world_skill,
    source_imports_no_backend,
    worked_example,
)

sys.path.insert(0, os.path.dirname(__file__))
from _common import provenance, write_json  # type: ignore[import-not-found]  # noqa: E402
from _gates import gates_block  # type: ignore[import-not-found]  # noqa: E402

SCRATCH = Path(os.environ.get("OMNIBIAS_SCRATCH", "artifacts"))


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args(argv)
    full = bool(args.full)
    t0 = time.perf_counter()
    ex = worked_example()
    g1 = ex["abs_err"] < 1e-12
    skill = jet_world_skill()
    g2 = bool(skill["g2_earned"])
    hon = honesty_payload()
    g3 = hon["navier_stokes_proof_claim"] is False and hon["continuum_claim"] is False
    g4 = source_imports_no_backend()
    entries = [
        {
            "name": "g1_cell",
            "passed": g1,
            "pred": ex["pred"],
            "named": ex["named"],
            "remainder": ex["remainder"],
        },
        {
            "name": "g2_skill",
            "passed": g2,
            "contains": skill["contains"],
            "width": skill["width"],
            "pred_err": skill["pred_err"],
            "g2_earned": skill["g2_earned"],
        },
        {
            "name": "g3_honesty",
            "passed": g3,
            "navier_stokes_proof_claim": False,
            "continuum_claim": False,
        },
        {
            "name": "g4_purity",
            "passed": g4,
            "imports_backend": False,
        },
    ]
    for entry in entries:
        print(entry["name"], "ok" if entry["passed"] else "FAIL")
        if not entry["passed"]:
            raise AssertionError(f"{entry['name']} failed: {entry}")
    payload = {
        **provenance(
            schema="omnibias.benchmarks.world_model_jet.v1",
            config={"family": "world_model_jet", "full": full, "honesty": hon},
        ),
        "gates": dict(gates_block(entries)),
        "disclaimer": DISCLAIMER,
        "wall_seconds": time.perf_counter() - t0,
    }
    name = "world_model_jet.json" if full else "world_model_jet_smoke.json"
    if full:
        dest = SCRATCH / "inventions" / "world_model_jet"
        dest.mkdir(parents=True, exist_ok=True)
        path = dest / name
        path.write_text(__import__("json").dumps(payload, indent=2) + "\n", encoding="utf-8")
    else:
        path = write_json(name, payload)
    print(f"wrote {path}")
    return payload


if __name__ == "__main__":
    main()
