# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Frontier 09-12: holonomic layer (the block is an Ore annihilator).

G1 matches the ``D-1`` / ``D^2+1`` jets. G2 recovers a rational
multiple of ``D^2+1`` from a sine prefix. G3 refuses a nonlinear PDE
D-finite claim. D-finite class only. Not CCF stretch.
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
from omnibias.holonomic._core.layer import (
    DISCLAIMER,
    honesty_payload,
    sin_skill,
    worked_example,
)

SCRATCH = Path(os.environ.get("OMNIBIAS_SCRATCH", "artifacts"))


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args(argv)
    full = bool(args.full)
    t0 = time.perf_counter()
    ex = worked_example()
    g1 = bool(ex["exp_err"] < 1e-12 and ex["sin_u2"] == 0 and ex["sin_u3"] == -1)
    skill = sin_skill()
    g2 = bool(skill["g2_earned"])
    hon = honesty_payload()
    g3 = hon["nonlinear_pde_claimed_dfinite"] is False
    entries = [
        {
            "name": "g1_exp_sin_jet",
            "passed": g1,
            "exp_err": ex["exp_err"],
            "sin_u2": int(ex["sin_u2"]),
            "sin_u3": int(ex["sin_u3"]),
        },
        {
            "name": "g2_skill",
            "passed": g2,
            "multiple_of_d2_plus_1": skill["multiple_of_d2_plus_1"],
            "max_residual": skill["max_residual"],
        },
        {
            "name": "g3_honesty",
            "passed": g3,
            "nonlinear_pde_claimed_dfinite": False,
        },
    ]
    for entry in entries:
        print(entry["name"], "ok" if entry["passed"] else "FAIL")
        if not entry["passed"]:
            raise AssertionError(f"{entry['name']} failed: {entry}")
    payload = {
        **provenance(
            schema="omnibias.benchmarks.holonomic_layer.v1",
            config={"family": "holonomic_layer", "full": full, "honesty": hon},
        ),
        "gates": dict(gates_block(entries)),
        "disclaimer": DISCLAIMER,
        "wall_seconds": time.perf_counter() - t0,
    }
    name = "holonomic_layer.json" if full else "holonomic_layer_smoke.json"
    if full:
        dest = SCRATCH / "inventions" / "holonomic_layer"
        dest.mkdir(parents=True, exist_ok=True)
        path = dest / name
        path.write_text(__import__("json").dumps(payload, indent=2) + "\n", encoding="utf-8")
    else:
        path = write_json(name, payload)
    print(f"wrote {path}")
    return payload


if __name__ == "__main__":
    main()
