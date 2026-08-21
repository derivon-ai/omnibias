# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Frontier 09-28: sliced-jet encoder.

G1 is the 2×2 reconstruct. G2 beats GAP and same-width CNN+GAP.
G3 refuses an unnamed sparse readout. Not a ViT.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any

from omnibias.core.sliced_jet import (
    DISCLAIMER,
    SlicedJetConfig,
    honesty_payload,
    sliced_jet_skill,
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
    g1 = ex["encoder_mae"] == 0.0 and ex["gap_mae"] == 0.375
    skill = sliced_jet_skill()
    g2 = bool(skill["g2_earned"])
    unnamed_refused = False
    try:
        SlicedJetConfig(top_k=1, energy="none")
    except ValueError:
        unnamed_refused = True
    hon = honesty_payload()
    g3 = unnamed_refused
    g4 = (
        hon["imagenet_claim"] is False
        and hon["vit_claim"] is False
        and hon["euclidean_RD_claim"] is False
    )
    entries = [
        {
            "name": "g1_cell",
            "passed": g1,
            "encoder_mae": ex["encoder_mae"],
            "gap_mae": ex["gap_mae"],
        },
        {
            "name": "g2_skill",
            "passed": g2,
            "encoder_mae": skill["encoder_mae"],
            "gap_mae": skill["gap_mae"],
            "cnn_mae": skill["cnn_mae"],
        },
        {"name": "g3_energy", "passed": g3, "unnamed_sparse_refused": True},
        {
            "name": "g4_honesty",
            "passed": g4,
            "imagenet_claim": False,
            "vit_claim": False,
        },
    ]
    for entry in entries:
        print(entry["name"], "ok" if entry["passed"] else "FAIL")
        if not entry["passed"]:
            raise AssertionError(f"{entry['name']} failed: {entry}")
    payload = {
        **provenance(
            schema="omnibias.benchmarks.sliced_jet_encoder.v1",
            config={"family": "sliced_jet_encoder", "full": full, "honesty": hon},
        ),
        "gates": dict(gates_block(entries)),
        "disclaimer": DISCLAIMER,
        "wall_seconds": time.perf_counter() - t0,
    }
    name = "sliced_jet_encoder.json" if full else "sliced_jet_encoder_smoke.json"
    if full:
        dest = SCRATCH / "inventions" / "sliced_jet_encoder"
        dest.mkdir(parents=True, exist_ok=True)
        path = dest / name
        path.write_text(__import__("json").dumps(payload, indent=2) + "\n", encoding="utf-8")
    else:
        path = write_json(name, payload)
    print(f"wrote {path}")
    return payload


if __name__ == "__main__":
    main()
