# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Frontier 09-23: sharpness-augmented loss.

G1 is H=10 and L_sharp(0)=1. G2 trains a 1-D Poisson finite.
G3 records that this is not the 08-06 schedule.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any

from omnibias.core.sharp_loss import (
    DISCLAIMER,
    honesty_payload,
    sharpness_skill,
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
    g1 = ex["abs_hvp_err"] < 1e-12 and ex["abs_aug_err"] < 1e-12
    skill = sharpness_skill()
    g2 = bool(skill["g2_earned"])
    hon = honesty_payload()
    g3 = hon["is_08_06_schedule"] is False
    entries = [
        {
            "name": "g1_cell",
            "passed": g1,
            "hvp": ex["hvp"],
            "aug": ex["aug"],
        },
        {
            "name": "g2_skill",
            "passed": g2,
            "finite": skill["finite"],
            "skill_pos": skill["skill_pos"],
            "vs_0806_required": False,
            "g2_earned": skill["g2_earned"],
        },
        {
            "name": "g3_honesty",
            "passed": g3,
            "is_08_06_schedule": False,
            "schedule_only": False,
        },
    ]
    for entry in entries:
        print(entry["name"], "ok" if entry["passed"] else "FAIL")
        if not entry["passed"]:
            raise AssertionError(f"{entry['name']} failed: {entry}")
    payload = {
        **provenance(
            schema="omnibias.benchmarks.sharpness_regularizer.v1",
            config={"family": "sharpness_regularizer", "full": full, "honesty": hon},
        ),
        "gates": dict(gates_block(entries)),
        "disclaimer": DISCLAIMER,
        "wall_seconds": time.perf_counter() - t0,
    }
    name = "sharpness_regularizer.json" if full else "sharpness_regularizer_smoke.json"
    if full:
        dest = SCRATCH / "inventions" / "sharp_loss"
        dest.mkdir(parents=True, exist_ok=True)
        path = dest / name
        path.write_text(__import__("json").dumps(payload, indent=2) + "\n", encoding="utf-8")
    else:
        path = write_json(name, payload)
    print(f"wrote {path}")
    return payload


if __name__ == "__main__":
    main()
