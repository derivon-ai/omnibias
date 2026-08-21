# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Frontier 09-19: jet distillation (not ImageNet KD).

G1 recovers the tanh scale. G2: a 4-parameter jet read beats
value-only on the joint metric. G3 keeps ``imagenet_claim`` false.
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
from omnibias.core.jet_token import (
    DISCLAIMER,
    distill_skill,
    honesty_payload,
    recover_tanh_scale,
    ssl_flip_residual,
)

SCRATCH = Path(os.environ.get("OMNIBIAS_SCRATCH", "artifacts"))


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args(argv)
    full = bool(args.full)
    t0 = time.perf_counter()
    rec = recover_tanh_scale(0.5)
    g1 = bool(rec["loss"] < 1e-12 and rec["a"] == 1.0 and ssl_flip_residual(0.5) == 0.0)
    skill = distill_skill()
    g2 = bool(skill["below_1e3"] and skill["beats_value"])
    g3 = honesty_payload()["imagenet_claim"] is False
    entries = [
        {"name": "g1_tanh_scale", "passed": g1, "a": rec["a"], "loss": rec["loss"]},
        {
            "name": "g2_skill",
            "passed": g2,
            "jet_median": skill["jet_median"],
            "value_median": skill["value_median"],
            "tanh_teacher_floor": skill["tanh_teacher_floor"],
        },
        {"name": "g3_honesty", "passed": g3, "imagenet_claim": False},
    ]
    for entry in entries:
        print(entry["name"], "ok" if entry["passed"] else "FAIL")
        if not entry["passed"]:
            raise AssertionError(f"{entry['name']} failed: {entry}")
    payload = {
        **provenance(
            schema="omnibias.benchmarks.jet_distillation.v1",
            config={"family": "jet_distillation", "full": full, "honesty": honesty_payload()},
        ),
        "gates": dict(gates_block(entries)),
        "disclaimer": DISCLAIMER,
        "wall_seconds": time.perf_counter() - t0,
    }
    name = "jet_distillation.json" if full else "jet_distillation_smoke.json"
    if full:
        dest = SCRATCH / "inventions" / "jet_distill"
        dest.mkdir(parents=True, exist_ok=True)
        path = dest / name
        path.write_text(__import__("json").dumps(payload, indent=2) + "\n", encoding="utf-8")
    else:
        path = write_json(name, payload)
    print(f"wrote {path}")
    return payload


if __name__ == "__main__":
    main()
