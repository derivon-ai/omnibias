# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Frontier 09-05: Taylor-model neuron (sound remainder, not a deep net).

G1 is grid + sample containment. G2 requires a non-vacuous remainder
on the worked box. G3 records enclosure explosion at depth 6. G4 is
no backend leak. Jets are founding bias collapse, not temperature
collapse. Not ImageNet. Not CCF stretch.
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
from omnibias.core.verified.tm_neuron import (
    DISCLAIMER,
    TMNeuronSpec,
    contains_grid_and_sample,
    depth_honesty,
    honesty_payload,
    input_tm,
    source_imports_no_backend,
    tm_dense,
    worked_example,
)

SCRATCH = Path(os.environ.get("OMNIBIAS_SCRATCH", "artifacts"))


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args(argv)
    full = bool(args.full)
    t0 = time.perf_counter()
    spec = TMNeuronSpec(order=1, activation="sigmoid")
    tm = tm_dense(input_tm(0.0, 0.2, spec.order), 1.0, 0.0, spec)
    ex = worked_example()
    depth = depth_honesty()
    g1 = contains_grid_and_sample(tm, "sigmoid", 0.0, 0.2)
    g2 = bool(float(ex["remainder_width"]) < 0.1 and not ex["vacuous"])
    g3 = "enclosure_exploded" in depth
    g4 = source_imports_no_backend()
    hon = honesty_payload()
    entries = [
        {"name": "g1_soundness", "passed": g1},
        {
            "name": "g2_non_vacuous",
            "passed": g2,
            "remainder_width": ex["remainder_width"],
            "width": ex["width"],
        },
        {
            "name": "g3_depth_honesty",
            "passed": g3,
            "enclosure_exploded": depth["enclosure_exploded"],
            "remainder_width": depth["remainder_width"],
        },
        {"name": "g4_no_backend", "passed": g4, "imagenet_claim": hon["imagenet_claim"]},
    ]
    for entry in entries:
        print(entry["name"], "ok" if entry["passed"] else "FAIL")
        if not entry["passed"]:
            raise AssertionError(f"{entry['name']} failed: {entry}")
    payload = {
        **provenance(
            schema="omnibias.benchmarks.taylor_model_neuron.v1",
            config={"family": "tm_neuron", "full": full, "honesty": hon},
        ),
        "gates": dict(gates_block(entries)),
        "disclaimer": DISCLAIMER,
        "wall_seconds": time.perf_counter() - t0,
    }
    name = "taylor_model_neuron.json" if full else "taylor_model_neuron_smoke.json"
    if full:
        dest = SCRATCH / "inventions" / "tm_neuron"
        dest.mkdir(parents=True, exist_ok=True)
        path = dest / name
        path.write_text(__import__("json").dumps(payload, indent=2) + "\n", encoding="utf-8")
    else:
        path = write_json(name, payload)
    print(f"wrote {path}")
    return payload


if __name__ == "__main__":
    main()
