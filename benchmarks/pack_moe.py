# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Frontier 09-07: Pack-MoE (slab-mass router, not softmax).

G1 is the symmetric band-gate identity. G2 is two-tone skill vs a
single clustered pack of the same width. G3 keeps the softmax
router off by default. Jets are founding bias collapse, not
temperature collapse (unless ``beta != 1``). Not ImageNet. Not a
05-02 LightGBM reversal. Not CCF stretch.
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
from omnibias.core.pack_moe import (
    DISCLAIMER,
    honesty_payload,
    two_tone_skill,
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
    g1 = bool(ex["g_a_err"] < 1e-12 and ex["g_b_err"] < 1e-12 and ex["y_err"] < 1e-12)
    skill = two_tone_skill()
    g2 = bool(skill["beats_pack"] and skill["below_1e2"] and float(skill["skill"]) > 0.0)
    hon = honesty_payload()
    g3 = hon["router_is_softmax"] is False and hon["temperature_collapse_used"] is False
    entries = [
        {"name": "g1_router", "passed": g1, "g_a": ex["g_a"], "g_b": ex["g_b"], "y": ex["y"]},
        {
            "name": "g2_skill",
            "passed": g2,
            "moe_median": skill["moe_median"],
            "pack_median": skill["pack_median"],
            "skill": skill["skill"],
        },
        {
            "name": "g3_honesty",
            "passed": g3,
            "router_is_softmax": False,
            "temperature_collapse_used": False,
        },
    ]
    for entry in entries:
        print(entry["name"], "ok" if entry["passed"] else "FAIL")
        if not entry["passed"]:
            raise AssertionError(f"{entry['name']} failed: {entry}")
    payload = {
        **provenance(
            schema="omnibias.benchmarks.pack_moe.v1",
            config={"family": "pack_moe", "full": full, "honesty": hon},
        ),
        "gates": dict(gates_block(entries)),
        "disclaimer": DISCLAIMER,
        "wall_seconds": time.perf_counter() - t0,
    }
    name = "pack_moe.json" if full else "pack_moe_smoke.json"
    if full:
        dest = SCRATCH / "inventions" / "pack_moe"
        dest.mkdir(parents=True, exist_ok=True)
        path = dest / name
        path.write_text(__import__("json").dumps(payload, indent=2) + "\n", encoding="utf-8")
    else:
        path = write_json(name, payload)
    print(f"wrote {path}")
    return payload


if __name__ == "__main__":
    main()
