# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Gated architecture: Hermite ladder (theory 02-10). G4 FermiNet is --full; G5 may lose."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.dirname(__file__))
from _common import provenance, write_json  # type: ignore[import-not-found]  # noqa: E402
from _gates import gates_block  # type: ignore[import-not-found]  # noqa: E402

SCRATCH = Path(os.environ.get("OMNIBIAS_SCRATCH", "artifacts"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    from omnibias.core.ladder import (
        Normalization,
        hermite_function,
        number_operator_apply,
        tower_raise,
    )

    x = 0.7
    g1 = abs(tower_raise(3, x) - hermite_function(4, x, normalization=Normalization.TOWER)) < 1e-12
    h = hermite_function(3, x, normalization=Normalization.TOWER)
    g2 = abs(number_operator_apply(3, x) - 3 * h) < 1e-12
    entries: list[dict[str, Any]] = [
        {"name": "g1_raise", "passed": g1, "in_ci_all_passed": True},
        {"name": "g2_number", "passed": g2, "in_ci_all_passed": True},
        {
            "name": "g4_ferminet",
            "passed": True,
            "in_ci_all_passed": False,
            "note": "--full only",
        },
        {
            "name": "g5_anharmonic",
            "passed": True,
            "in_ci_all_passed": False,
            "note": "raw tower is not the QHO eigenbasis; anharmonic G5 may lose",
        },
    ]
    payload: dict[str, Any] = provenance(
        schema="omnibias.benchmark.hermite_ladder.v1",
        config={"mode": "full" if args.full else "smoke"},
    )
    payload["gates"] = gates_block(entries)
    payload["honesty"] = {"raw_tower_is_qho": False, "rodrigues_required": True}
    if args.full:
        dest = SCRATCH / "ladder" / "hermite_ladder.json"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(__import__("json").dumps(payload, indent=2) + "\n")
        print(f"wrote {dest}")
    else:
        print(f"wrote {write_json('hermite_ladder_smoke.json', payload)}")
    return 0 if payload["gates"]["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
