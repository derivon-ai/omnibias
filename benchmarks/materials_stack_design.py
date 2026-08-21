# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Frontier 07-07c: exact dT/dtheta stack design (tooling, not a new material).

G0 is the quarter-wave closed form (Macleod). G5 is a 10x cheaper
gradient at 20 layers. G6 checks energy and unitarity at every
iterate. Jets are founding bias collapse, not temperature collapse.
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
from omnibias.pinn.stack import (
    DISCLAIMER,
    design_stack,
    g0_quarter_wave_report,
    gradient_cost_report,
    honesty_payload,
)

SCRATCH = Path(os.environ.get("OMNIBIAS_SCRATCH", "artifacts"))


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args(argv)
    full = bool(args.full)
    t0 = time.perf_counter()
    g0 = g0_quarter_wave_report()
    g5 = gradient_cost_report(n_pairs=10)
    design = design_stack(0.99, n_pairs=3, n_steps=4, step_size=1e-4)
    g6 = max(design.unitarity) <= 1e-14 and max(design.energy) <= 1e-12
    entries = [
        {
            "name": "g0_quarter_wave",
            "passed": bool(g0["closed_matches"] and g0["transfer_matches"]),
            "closed_form": g0["closed_form"],
            "transfer": g0["transfer"],
        },
        {
            "name": "g5_gradient_cost",
            "passed": bool(g5["win"]),
            "n_layers": g5["n_layers"],
            "cost_ratio": g5["cost_ratio"],
        },
        {
            "name": "g6_identities",
            "passed": g6,
            "max_unitarity": max(design.unitarity),
            "max_energy": max(design.energy),
        },
    ]
    for entry in entries:
        print(entry["name"], "ok" if entry["passed"] else "FAIL")
        if not entry["passed"]:
            raise AssertionError(f"{entry['name']} failed: {entry}")
    payload = {
        **provenance(
            schema="omnibias.benchmarks.materials_stack_design.v1",
            config={"family": "materials_stack_design", "full": full, "honesty": honesty_payload()},
        ),
        "gates": dict(gates_block(entries)),
        "disclaimer": DISCLAIMER,
        "wall_seconds": time.perf_counter() - t0,
    }
    name = "materials_stack_design.json" if full else "materials_stack_design_smoke.json"
    if full:
        dest = SCRATCH / "domain_programs"
        dest.mkdir(parents=True, exist_ok=True)
        path = dest / name
        path.write_text(__import__("json").dumps(payload, indent=2) + "\n", encoding="utf-8")
    else:
        path = write_json(name, payload)
    print(f"wrote {path}")
    return payload


if __name__ == "__main__":
    main()
