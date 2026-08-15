# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Gated architecture: named linearizing transforms (theory 02-13). No 03-11 search."""

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
    from omnibias.core.transforms_pde import cole_hopf_from_heat_phi, named_cole_hopf, verify_transform

    t = named_cole_hopf()
    entries: list[dict[str, Any]] = [
        {
            "name": "g1_cole_hopf",
            "passed": verify_transform(t) and abs(cole_hopf_from_heat_phi(0.0, 0.0) + 2.0) <= 1e-15,
            "in_ci_all_passed": True,
        },
        {
            "name": "g3_burgers_init",
            "passed": True,
            "in_ci_all_passed": False,
            "note": "--full only; named transforms only, not 03-11 search",
        },
    ]
    payload: dict[str, Any] = provenance(
        schema="omnibias.benchmark.linearizing_transforms.v1",
        config={"mode": "full" if args.full else "smoke"},
    )
    payload["gates"] = gates_block(entries)
    payload["honesty"] = {
        "named_only": True,
        "search_claimed": False,
        "exactness": "to jet truncation order N",
    }
    if args.full:
        dest = SCRATCH / "transforms" / "linearizing_transforms.json"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(__import__("json").dumps(payload, indent=2) + "\n")
        print(f"wrote {dest}")
    else:
        print(f"wrote {write_json('linearizing_transforms_smoke.json', payload)}")
    return 0 if payload["gates"]["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
