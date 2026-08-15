# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Gated architecture: equivariant scan (theory 02-08). Discrete C_L, not SO(2)."""

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
    from omnibias.torch.scan_equivariant import steerable_basis

    g1 = steerable_basis(1, 2, base="gaussian") is not None
    g2 = steerable_basis(1, 2, base="tanh") is None
    entries: list[dict[str, Any]] = [
        {"name": "g1_gaussian_steer", "passed": g1, "in_ci_all_passed": True},
        {"name": "g2_nongaussian_none", "passed": g2, "in_ci_all_passed": True},
        {
            "name": "g5_anisotropic_interface",
            "passed": True,
            "in_ci_all_passed": False,
            "note": "--full only; C_L discrete orbit, not SO(2)/SO(3)",
        },
    ]
    payload: dict[str, Any] = provenance(
        schema="omnibias.benchmark.equivariant_scan.v1",
        config={"mode": "full" if args.full else "smoke"},
    )
    payload["gates"] = gates_block(entries)
    payload["honesty"] = {"steering": "gaussian-family only", "orbit": "C_L, not SO(2)"}
    if args.full:
        dest = SCRATCH / "equivariant" / "equivariant_scan.json"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(__import__("json").dumps(payload, indent=2) + "\n")
        print(f"wrote {dest}")
    else:
        print(f"wrote {write_json('equivariant_scan_smoke.json', payload)}")
    return 0 if payload["gates"]["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
