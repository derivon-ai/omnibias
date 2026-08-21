# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Frontier 09-24: proof-carrying forward (not an 08-09 step filter).

G1 is grid + sample containment. G2 requires a non-vacuous box of
width < 0.1. G3 keeps Lean flags false. G4 records that this is not
the 08-09 step filter. Jets are founding bias collapse, not
temperature collapse. Not ImageNet. Not CCF stretch.
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
from omnibias.verify._core.pci import (
    DISCLAIMER,
    honesty_payload,
    pci_sound,
    two_layer_box,
    worked_sigmoid_box,
)

SCRATCH = Path(os.environ.get("OMNIBIAS_SCRATCH", "artifacts"))


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args(argv)
    full = bool(args.full)
    t0 = time.perf_counter()
    result, tm = worked_sigmoid_box()
    hon = honesty_payload()
    two = two_layer_box()
    g1 = pci_sound(result, tm)
    g2 = bool(result.width < 0.1 and result.vacuous is False)
    g3 = (
        result.theorem_prover_verified is False
        and result.mathlib_verified is False
        and hon["theorem_prover_verified"] is False
        and hon["mathlib_verified"] is False
    )
    g4 = result.is_08_09_step_filter is False and hon["is_08_09_step_filter"] is False
    entries = [
        {"name": "g1_soundness", "passed": g1},
        {
            "name": "g2_non_vacuous",
            "passed": g2,
            "width": result.width,
            "two_layer_width": two.width,
        },
        {"name": "g3_flags", "passed": g3, "theorem_prover_verified": False},
        {"name": "g4_distinct", "passed": g4, "is_08_09_step_filter": False},
    ]
    for entry in entries:
        print(entry["name"], "ok" if entry["passed"] else "FAIL")
        if not entry["passed"]:
            raise AssertionError(f"{entry['name']} failed: {entry}")
    payload = {
        **provenance(
            schema="omnibias.benchmarks.proof_carrying_forward.v1",
            config={"family": "pci", "full": full, "honesty": hon},
        ),
        "gates": dict(gates_block(entries)),
        "disclaimer": DISCLAIMER,
        "wall_seconds": time.perf_counter() - t0,
    }
    name = "proof_carrying_forward.json" if full else "proof_carrying_forward_smoke.json"
    if full:
        dest = SCRATCH / "inventions" / "pci"
        dest.mkdir(parents=True, exist_ok=True)
        path = dest / name
        path.write_text(__import__("json").dumps(payload, indent=2) + "\n", encoding="utf-8")
    else:
        path = write_json(name, payload)
    print(f"wrote {path}")
    return payload


if __name__ == "__main__":
    main()
