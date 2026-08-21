# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Frontier 09-15: q-OMBU / timescale hybrid.

G1 matches D_q z^2 = 4.02. G2 is monotone as q -> 1.
G3 records no continuum claim from the named limit.
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
from omnibias.qcalculus._core.hybrid import (
    DISCLAIMER,
    honesty_payload,
    q_limit_skill,
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
    g1 = bool(ex["y_err"] < 1e-12 and ex["limit_err"] < 1e-12)
    skill = q_limit_skill()
    g2 = bool(skill["monotone"])
    hon = honesty_payload()
    g3 = hon["continuum_claimed_from_q_limit"] is False
    entries = [
        {
            "name": "g1_cell",
            "passed": g1,
            "y": ex["y"],
            "y_err": ex["y_err"],
            "limit": ex["limit"],
        },
        {
            "name": "g2_skill",
            "passed": g2,
            "errs": skill["errs"],
            "g2_earned": skill["g2_earned"],
        },
        {
            "name": "g3_honesty",
            "passed": g3,
            "continuum_claimed_from_q_limit": False,
        },
    ]
    for entry in entries:
        print(entry["name"], "ok" if entry["passed"] else "FAIL")
        if not entry["passed"]:
            raise AssertionError(f"{entry['name']} failed: {entry}")
    payload = {
        **provenance(
            schema="omnibias.benchmarks.q_ombu_timescale.v1",
            config={"family": "q_ombu_timescale", "full": full, "honesty": hon},
        ),
        "gates": dict(gates_block(entries)),
        "disclaimer": DISCLAIMER,
        "wall_seconds": time.perf_counter() - t0,
    }
    name = "q_ombu_timescale.json" if full else "q_ombu_timescale_smoke.json"
    if full:
        dest = SCRATCH / "inventions" / "q_ombu"
        dest.mkdir(parents=True, exist_ok=True)
        path = dest / name
        path.write_text(__import__("json").dumps(payload, indent=2) + "\n", encoding="utf-8")
    else:
        path = write_json(name, payload)
    print(f"wrote {path}")
    return payload


if __name__ == "__main__":
    main()
