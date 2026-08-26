# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Frontier 09-26: net-to-annihilator export.

G1 is a D-1 serialize/parse identity. G2 keeps verified flags
false. G3 refuses a continuum Lean sentence. G4 is export-module
purity (no torch/jax).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "packages" / "omnibias-holonomic" / "src"))
from omnibias.holonomic._core.export import (  # noqa: E402
    DISCLAIMER,
    export_skill,
    honesty_payload,
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
    g1 = bool(ex["match"]) and ex["order"] == 1
    skill = export_skill()
    g2 = bool(skill["g2_earned"])
    hon = honesty_payload()
    g3 = hon["theorem_prover_verified"] is False and bool(skill["continuum_raised"])
    g4 = source_imports_no_backend()
    entries = [
        {
            "name": "g1_cell",
            "passed": g1,
            "order": ex["order"],
            "match": ex["match"],
        },
        {
            "name": "g2_skill",
            "passed": g2,
            "flags_false": skill["flags_false"],
            "continuum_raised": skill["continuum_raised"],
            "g2_earned": skill["g2_earned"],
        },
        {
            "name": "g3_honesty",
            "passed": g3,
            "theorem_prover_verified": False,
            "mathlib_verified": False,
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
            schema="omnibias.benchmarks.net_to_annihilator.v1",
            config={"family": "net_to_annihilator", "full": full, "honesty": hon},
        ),
        "gates": dict(gates_block(entries)),
        "disclaimer": DISCLAIMER,
        "wall_seconds": time.perf_counter() - t0,
    }
    name = "net_to_annihilator.json" if full else "net_to_annihilator_smoke.json"
    if full:
        dest = SCRATCH / "inventions" / "annihilator_export"
        dest.mkdir(parents=True, exist_ok=True)
        path = dest / name
        path.write_text(__import__("json").dumps(payload, indent=2) + "\n", encoding="utf-8")
    else:
        path = write_json(name, payload)
    print(f"wrote {path}")
    return payload


if __name__ == "__main__":
    main()
