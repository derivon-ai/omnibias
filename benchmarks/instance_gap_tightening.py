# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Named n<=8 MaxSAT Lasserre level-1 vs 2 sandwiches (PNP row).

Every sandwich contains the brute-force OPT. Level 2 is not worse than
level 1. Gaps are reported, never claimed tight, never P = NP.
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.dirname(__file__))
from _common import provenance, write_json  # type: ignore[import-not-found]  # noqa: E402
from _gates import gates_block  # type: ignore[import-not-found]  # noqa: E402

SCRATCH = Path(os.environ.get("OMNIBIAS_SCRATCH", "artifacts"))

# Named n<=8 MaxSAT family with a brute-force OPT oracle.
# Smoke is the n=1 core (level-2 SDP stays CI-cheap). Larger instances are --full.
FAMILY_SMOKE: tuple[tuple[str, list[list[int]], list[float]], ...] = (
    ("unsat_core", [[1], [-1]], [1.0, 1.0]),
)
FAMILY_FULL: tuple[tuple[str, list[list[int]], list[float]], ...] = FAMILY_SMOKE + (
    ("three_clause", [[1, -2], [2, 3], [-1, -3]], [1.5, 2.0, 0.5]),
    ("four_var", [[1, 2], [-1, 3], [2, -4], [-3, -4]], [1.0, 1.0, 1.0, 1.0]),
)


def _honesty() -> dict[str, object]:
    return {
        "p_vs_np_claim": False,
        "tight": False,
        "note": "sound sandwiches on named n<=8 instances; never a parent claim",
    }


def _run_family(
    *,
    bisection_steps: int,
    family: tuple[tuple[str, list[list[int]], list[float]], ...],
) -> dict[str, Any]:
    from omnibias.discrete import brute_force_min, certify_gap, decode, tighten_gap
    from omnibias.discrete.maxsat import max_sat

    rows: list[dict[str, Any]] = []
    sandwiches = True
    level2_ge = True
    ratios: list[float] = []
    certified = True
    for name, clauses, weights in family:
        prob = max_sat(clauses, weights=weights)
        _, opt = brute_force_min(prob)
        assignment, _ = decode(prob, n_starts=16)
        one = certify_gap(prob, assignment, level=1, bisection_steps=bisection_steps)
        both = tighten_gap(
            prob,
            assignment,
            levels=(1, 2),
            bisection_steps=bisection_steps,
            seed_lower=one.lower_bound,
        )
        c2 = both.certificate
        if not (one.is_sound and c2.is_sound):
            sandwiches = False
        if one.lower_bound > opt + 1e-6 or c2.lower_bound > opt + 1e-6:
            sandwiches = False
        if one.energy < opt - 1e-9 or c2.energy < opt - 1e-9:
            sandwiches = False
        if c2.lower_bound + 1e-9 < one.lower_bound:
            level2_ge = False
        span = max(abs(opt), 1e-12)
        ratios.append(both.gap / span)
        if both.tight or both.honesty()["p_vs_np_claim"]:
            certified = False
        if c2.certified is False and c2.method == "sos":
            certified = False
        rows.append(
            {
                "name": name,
                "n": prob.n,
                "opt": opt,
                "level1_lower": one.lower_bound,
                "tightened_lower": c2.lower_bound,
                "level_used": both.level_used,
                "gap": both.gap,
                "certified": c2.certified,
                "tight": both.tight,
            }
        )
    return {
        "n_instances": len(rows),
        "max_n": max(row["n"] for row in rows),
        "sandwiches_contain_opt": sandwiches,
        "level2_not_worse": level2_ge,
        "median_gap_ratio": statistics.median(ratios) if ratios else None,
        "never_tight": all(row["tight"] is False for row in rows),
        "certified_only_when_sound": certified,
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    steps = 20 if args.full else 12
    try:
        import omnibias.sos  # noqa: F401
    except ImportError:
        payload = provenance(
            schema="omnibias.benchmark.instance_gap_tightening.v1",
            config={"mode": "full" if args.full else "smoke", "bisection_steps": steps},
        )
        payload["gates"] = gates_block(
            [{"name": "sos_available", "passed": False}]
        )
        payload["honesty"] = _honesty()
        print(f"wrote {write_json('instance_gap_tightening_smoke.json', payload)}")
        return 1

    report = _run_family(
        bisection_steps=steps,
        family=FAMILY_FULL if args.full else FAMILY_SMOKE,
    )
    entries: list[dict[str, Any]] = [
        {
            "name": "sandwiches_contain_opt",
            "passed": bool(report["sandwiches_contain_opt"] and report["max_n"] <= 8),
        },
        {
            "name": "level2_not_worse",
            "passed": bool(report["level2_not_worse"]),
        },
        {
            "name": "never_tight",
            "passed": bool(report["never_tight"]),
        },
        {
            "name": "honesty",
            "passed": _honesty()["p_vs_np_claim"] is False,
        },
    ]
    payload: dict[str, Any] = provenance(
        schema="omnibias.benchmark.instance_gap_tightening.v1",
        config={"mode": "full" if args.full else "smoke", "bisection_steps": steps},
    )
    payload.update(report)
    payload["gates"] = gates_block(entries)
    payload["honesty"] = _honesty()
    if args.full:
        dest = SCRATCH / "discrete" / "instance_gap_tightening.json"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(__import__("json").dumps(payload, indent=2) + "\n")
        print(f"wrote {dest}")
    else:
        print(f"wrote {write_json('instance_gap_tightening_smoke.json', payload)}")
    return 0 if payload["gates"]["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
