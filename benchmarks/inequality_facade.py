# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Frontier 09-30: unified inequality engine catalog.

G1 is dispatch. G2 is the locked twelve. G3 is honesty. G4 is the
propose/rationalize/check pipeline on linear and polynomial plants.
G5 is leftover-recorded (not in CI all_passed).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any

from omnibias.core.proof import Conjecture
from omnibias.core.proof.inequality import (
    INEQUALITY_KIND,
    InequalitySystem,
    build_inequality_machine,
    locked_catalog,
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
    machine = build_inequality_machine()
    kinds = machine.kinds()
    g1 = INEQUALITY_KIND in kinds
    rows = []
    for row in locked_catalog():
        system = InequalitySystem.from_mapping(row)
        verdict = machine.evaluate(
            Conjecture(system.name, INEQUALITY_KIND, system.as_dict()),
            replay=True,
        )
        rows.append(
            {
                "name": system.name,
                "expected": row["expected"],
                "status": verdict.status,
                "replay_ok": verdict.replay_ok,
                "pipeline": (
                    list(verdict.certificate["payload"]["pipeline"])
                    if verdict.certificate is not None
                    else []
                ),
            }
        )
    g2 = all(
        item["status"] == item["expected"] and item["replay_ok"] is True
        for item in rows
    )
    forged = machine.evaluate(
        Conjecture(
            "forged",
            INEQUALITY_KIND,
            next(item for item in locked_catalog() if item["name"] == "linear_box_sat"),
            claims={"p_equals_np_claim": True},
        )
    )
    g3 = forged.status == "BLOCKED" and forged.honesty_ok is False
    g4 = all(
        item["name"] in {"linear_box_sat", "poly_constant_one"}
        and item["pipeline"] == ["propose", "rationalize", "check"]
        for item in rows
        if item["name"] in {"linear_box_sat", "poly_constant_one"}
    )
    entries = [
        {"name": "g1_dispatch", "passed": g1, "kinds": list(kinds)},
        {"name": "g2_catalog", "passed": g2, "n": len(rows)},
        {"name": "g3_honesty", "passed": g3},
        {
            "name": "g4_pipeline",
            "passed": g4,
            "in_ci_all_passed": True,
        },
        {
            "name": "g5_enclosure_width",
            "passed": False,
            "earned": False,
            "leftover_recorded": True,
            "in_ci_all_passed": False,
            "note": "enclosure width vs naive interval is leftover-recorded, not SOTA",
        },
    ]
    for entry in entries:
        if entry["name"] == "g5_enclosure_width":
            continue
        print(entry["name"], "ok" if entry["passed"] else "FAIL")
        if not entry["passed"]:
            raise AssertionError(f"{entry['name']} failed: {entry}")
    gates = dict(gates_block(entries))
    gates["all_passed"] = all(
        bool(item.get("passed"))
        for item in entries
        if item["name"] != "g5_enclosure_width"
    )
    payload = {
        **provenance(
            schema="omnibias.benchmarks.inequality_facade.v1",
            config={"family": "inequality_facade", "full": full},
        ),
        "rows": rows,
        "gates": gates,
        "wall_seconds": time.perf_counter() - t0,
    }
    name = "inequality_facade.json" if full else "inequality_facade_smoke.json"
    if full:
        dest = SCRATCH / "inventions" / "inequality_facade"
        dest.mkdir(parents=True, exist_ok=True)
        path = dest / name
        path.write_text(__import__("json").dumps(payload, indent=2) + "\n", encoding="utf-8")
    else:
        path = write_json(name, payload)
    print(f"wrote {path}")
    return payload


if __name__ == "__main__":
    main()
