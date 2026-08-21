# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Frontier 07-07b: Harris-sheet resistive layer (tooling, not fusion).

G0 is the Harris 1962 force balance plus imported MHD field ops.
G3: pack dim is independent of sheet thickness; uniform grows as 1/d.
G4: force residual is within 1%. Jets are founding bias collapse,
not temperature collapse.
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
from omnibias.pinn.plasma import (
    DISCLAIMER,
    field_ops_resolve,
    harris_force_errors,
    honesty_payload,
    layer_scaling_report,
)

SCRATCH = Path(os.environ.get("OMNIBIAS_SCRATCH", "artifacts"))


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args(argv)
    full = bool(args.full)
    t0 = time.perf_counter()
    force = harris_force_errors(0.1)
    ops = field_ops_resolve()
    g0 = max(force) <= 1e-12 and ops.get("torch.induction_residual") is True
    rows = layer_scaling_report()
    uniforms = [int(r["uniform_dim"]) for r in rows]
    g3 = (
        len(rows) >= 4
        and {int(r["pack_dim"]) for r in rows} == {4}
        and all(float(r["pack_residual"]) <= 1e-12 for r in rows)
        and uniforms[-1] >= 4 * uniforms[0]
    )
    g4 = max(harris_force_errors(0.05)) <= 0.01
    entries = [
        {"name": "g0_harris_baseline", "passed": g0, "ops": ops},
        {
            "name": "g3_layer_scaling",
            "passed": g3,
            "pack_dims": [int(r["pack_dim"]) for r in rows],
            "uniform_dims": uniforms,
        },
        {"name": "g4_reference_equilibrium", "passed": g4, "max_force": max(force)},
    ]
    for entry in entries:
        print(entry["name"], "ok" if entry["passed"] else "FAIL")
        if not entry["passed"]:
            raise AssertionError(f"{entry['name']} failed: {entry}")
    payload = {
        **provenance(
            schema="omnibias.benchmarks.plasma_resistive_layer.v1",
            config={"family": "plasma_resistive_layer", "full": full, "honesty": honesty_payload()},
        ),
        "gates": dict(gates_block(entries)),
        "disclaimer": DISCLAIMER,
        "wall_seconds": time.perf_counter() - t0,
    }
    name = "plasma_resistive_layer.json" if full else "plasma_resistive_layer_smoke.json"
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
