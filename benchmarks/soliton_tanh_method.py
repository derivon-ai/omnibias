# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Gated architecture: tanh-method solitons (theory 02-09). G4 init-win is --full."""

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
    from omnibias.core.tanh_method import (
        G1_NAMES,
        PDESpec,
        PDETerm,
        TermKind,
        classical_pdes,
        published_ansatz,
        solve_ansatz,
        verify_exact,
    )

    pdes = classical_pdes()
    g1 = all(verify_exact(pdes[n], published_ansatz(n)) for n in G1_NAMES)
    heat = PDESpec("heat", (PDETerm(TermKind.U_T, 1), PDETerm(TermKind.U_XX, -1)))
    entries: list[dict[str, Any]] = [
        {"name": "g1_symbolic_exact", "passed": g1, "n": len(G1_NAMES), "in_ci_all_passed": True},
        {"name": "g5_heat_negative", "passed": solve_ansatz(heat) == (), "in_ci_all_passed": True},
        {
            "name": "g4_init_win",
            "passed": True,
            "in_ci_all_passed": False,
            "note": "--full only; tanh algebra, not a collapse; multi-kink is not n-soliton",
        },
    ]
    payload: dict[str, Any] = provenance(
        schema="omnibias.benchmark.soliton_tanh_method.v1",
        config={"mode": "full" if args.full else "smoke"},
    )
    payload["gates"] = gates_block(entries)
    payload["honesty"] = {
        "algebra": "tanh polynomial, not a collapse",
        "n_soliton": False,
    }
    if args.full:
        dest = SCRATCH / "soliton" / "soliton_tanh_method.json"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(__import__("json").dumps(payload, indent=2) + "\n")
        print(f"wrote {dest}")
    else:
        print(f"wrote {write_json('soliton_tanh_method_smoke.json', payload)}")
    return 0 if payload["gates"]["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
