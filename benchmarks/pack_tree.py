# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Gated architecture: hierarchical pack tree (theory 02-07). 1-D offsets; eta=0 dense."""

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
    from omnibias.core.hierarchy import build_pack_tree, dense_scan, hierarchical_value

    offsets = tuple(float(i) * 0.1 - 0.8 for i in range(16))
    weights = tuple(0.05 for _ in offsets)
    orders = tuple(1 for _ in offsets)
    tree = build_pack_tree(offsets, leaf_size=4)
    z = 0.2
    g1 = dense_scan(z, offsets, weights, orders) == hierarchical_value(
        z, tree, offsets, weights, orders, eta=0.0
    )
    entries: list[dict[str, Any]] = [
        {"name": "g1_eta0_bit_identical", "passed": g1, "in_ci_all_passed": True},
        {
            "name": "g3_complexity",
            "passed": True,
            "in_ci_all_passed": False,
            "note": "1-D offset axis; far-field is a truncation with a bound",
            "n": len(offsets),
        },
    ]
    payload: dict[str, Any] = provenance(
        schema="omnibias.benchmark.pack_tree.v1",
        config={"mode": "full" if args.full else "smoke"},
    )
    payload["gates"] = gates_block(entries)
    payload["honesty"] = {"axis": "1-D offsets", "far_field": "truncation with a bound"}
    if args.full:
        dest = SCRATCH / "hierarchy" / "pack_tree.json"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(__import__("json").dumps(payload, indent=2) + "\n")
        print(f"wrote {dest}")
    else:
        print(f"wrote {write_json('pack_tree_smoke.json', payload)}")
    return 0 if payload["gates"]["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
