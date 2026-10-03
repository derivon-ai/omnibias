#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
"""Bounded Part-A patchwork and Route-2 campaign benchmark."""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Any

sys.path.insert(0, os.path.dirname(__file__))
from _common import provenance, write_json  # type: ignore[import-not-found]  # noqa: E402
from _gates import gates_block  # type: ignore[import-not-found]  # noqa: E402
from omnibias.geometry.part_a_campaign import run_part_a_campaign  # noqa: E402


def _campaign_gate(*, budget: int, triangulation_limit: int, include_route2: bool) -> dict[str, Any]:
    report = run_part_a_campaign(
        budget=budget,
        triangulation_limit=triangulation_limit,
        complete=False,
        include_route2=include_route2,
    )
    return {
        "name": "part_a_dual_tree_campaign",
        "passed": len(report.targets) == 2 and all(target.oval_count == 22 for target in report.targets),
        "payload": report.to_payload(),
        "detail": "both corrected (19,3) trees searched; hits remain inconclusive until complete exhaustion",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    budget = 256 if args.full else 32
    triangulation_limit = 128 if args.full else 32
    started = time.time()
    entries = [
        _campaign_gate(
            budget=budget,
            triangulation_limit=triangulation_limit,
            include_route2=args.full,
        )
    ]
    report = {
        **provenance(
            schema="omnibias.benchmarks.hilbert16_part_a_campaign.v1",
            config={
                "family": "hilbert16_part_a_campaign",
                "full": args.full,
                "budget": budget,
                "triangulation_limit": triangulation_limit,
            },
        ),
        "benchmark": "hilbert16_part_a_campaign",
        "gates": gates_block(entries),
        "full_hilbert16_solved": False,
        "wall_seconds": time.time() - started,
    }
    artifact = "hilbert16_part_a_campaign.json" if args.full else "hilbert16_part_a_campaign_smoke.json"
    path = write_json(artifact, report)
    print(f"wrote {path}")
    return 0 if report["gates"]["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
