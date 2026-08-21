# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Frontier 09-21: exact Hyvärinen score matching.

G1 is ``div(-x) = -1``. G2 trains an affine OMBU score on 1-D
N(0,1). G3 records that CNF exact ``div`` is not claimed as new.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any

from omnibias.core.score_matching import (
    DISCLAIMER,
    honesty_payload,
    score_matching_skill,
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
    g1 = abs(ex["div"] + 1.0) < 1e-12
    skill = score_matching_skill()
    g2 = bool(skill["g2_earned"])
    hon = honesty_payload()
    g3 = hon["cnf_div_claimed_new"] is False
    entries = [
        {
            "name": "g1_cell",
            "passed": g1,
            "div": ex["div"],
            "hyvarinen": ex["hyvarinen"],
        },
        {
            "name": "g2_skill",
            "passed": g2,
            "reached": skill["reached"],
            "exact_div_variance": skill["exact_div_variance"],
            "hutchinson_exact_path_variance": skill["hutchinson_exact_path_variance"],
            "g2_earned": skill["g2_earned"],
        },
        {
            "name": "g3_honesty",
            "passed": g3,
            "cnf_div_claimed_new": False,
        },
    ]
    for entry in entries:
        print(entry["name"], "ok" if entry["passed"] else "FAIL")
        if not entry["passed"]:
            raise AssertionError(f"{entry['name']} failed: {entry}")
    payload = {
        **provenance(
            schema="omnibias.benchmarks.exact_score_matching.v1",
            config={"family": "exact_score_matching", "full": full, "honesty": hon},
        ),
        "gates": dict(gates_block(entries)),
        "disclaimer": DISCLAIMER,
        "wall_seconds": time.perf_counter() - t0,
    }
    name = "exact_score_matching.json" if full else "exact_score_matching_smoke.json"
    if full:
        dest = SCRATCH / "inventions" / "score_matching"
        dest.mkdir(parents=True, exist_ok=True)
        path = dest / name
        path.write_text(__import__("json").dumps(payload, indent=2) + "\n", encoding="utf-8")
    else:
        path = write_json(name, payload)
    print(f"wrote {path}")
    return payload


if __name__ == "__main__":
    main()
