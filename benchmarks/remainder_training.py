# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Frontier 09-18: remainder training (loss is R_N, not L).

G1 matches analytic exp R_2(0.2). G2 remainder-trains an OMBU
against value-only MSE on the model's own R_2. G3 refuses to call
this spec 03-10 or 03-13. Not CCF stretch. Jets are founding bias
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
from omnibias.core.remainder_train import (
    DISCLAIMER,
    exp_remainder_skill,
    honesty_payload,
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
    g1 = bool(ex["R2_err"] < 1e-12)
    skill = exp_remainder_skill()
    g2 = bool(skill["below_1e4"] and skill["beats_value"])
    hon = honesty_payload()
    g3 = (
        hon["is_03_10"] is False
        and hon["is_03_13"] is False
        and hon["stretch_claim"] is False
    )
    entries = [
        {"name": "g1_analytic", "passed": g1, "R2": ex["R2"], "loss": ex["loss"]},
        {
            "name": "g2_skill",
            "passed": g2,
            "remainder_median": skill["remainder_median"],
            "value_median": skill["value_median"],
        },
        {
            "name": "g3_honesty",
            "passed": g3,
            "is_03_10": False,
            "is_03_13": False,
            "stretch_claim": False,
        },
    ]
    for entry in entries:
        print(entry["name"], "ok" if entry["passed"] else "FAIL")
        if not entry["passed"]:
            raise AssertionError(f"{entry['name']} failed: {entry}")
    payload = {
        **provenance(
            schema="omnibias.benchmarks.remainder_training.v1",
            config={"family": "remainder_training", "full": full, "honesty": hon},
        ),
        "gates": dict(gates_block(entries)),
        "disclaimer": DISCLAIMER,
        "wall_seconds": time.perf_counter() - t0,
    }
    name = "remainder_training.json" if full else "remainder_training_smoke.json"
    if full:
        dest = SCRATCH / "inventions" / "remainder"
        dest.mkdir(parents=True, exist_ok=True)
        path = dest / name
        path.write_text(__import__("json").dumps(payload, indent=2) + "\n", encoding="utf-8")
    else:
        path = write_json(name, payload)
    print(f"wrote {path}")
    return payload


if __name__ == "__main__":
    main()
