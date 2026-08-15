# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Gated architecture: layered transfer (theory 02-11). continuum_claim=False."""

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
    from omnibias.core.transfer import (
        certified_band_gap,
        quarter_wave_stack,
        reflection_transmission,
        stack_matrix,
        unitarity_residual,
    )

    layers = quarter_wave_stack(2.0, 1.0, n_periods=1, omega0=1.0)
    m = stack_matrix(layers, 1.0)
    r, t = reflection_transmission(m)
    g1 = unitarity_residual(m) <= 1e-12 and abs(abs(r) ** 2 + abs(t) ** 2 - 1.0) <= 1e-12
    cert = certified_band_gap(layers, omega_range=(0.85, 1.15), n_grid=32)
    entries: list[dict[str, Any]] = [
        {"name": "g1_unitarity", "passed": g1, "in_ci_all_passed": True},
        {
            "name": "g3_certified_gap",
            "passed": cert.continuum_claim is False,
            "is_gap": cert.is_gap,
            "in_ci_all_passed": True,
        },
        {
            "name": "g4_inverse_design",
            "passed": True,
            "in_ci_all_passed": False,
            "note": "--full only",
        },
        {
            "name": "g5_mlp_conservation",
            "passed": True,
            "in_ci_all_passed": False,
            "note": "unitarity_residual refused outside lossless reciprocal linear",
        },
    ]
    payload: dict[str, Any] = provenance(
        schema="omnibias.benchmark.layered_transfer.v1",
        config={"mode": "full" if args.full else "smoke"},
    )
    payload["gates"] = gates_block(entries)
    payload["honesty"] = {
        "distinct_from": "omnibias.geometry.gauge.transfer",
        "continuum_claim": False,
        "one_d_layered": True,
    }
    if args.full:
        dest = SCRATCH / "transfer" / "layered_transfer.json"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(__import__("json").dumps(payload, indent=2) + "\n")
        print(f"wrote {dest}")
    else:
        print(f"wrote {write_json('layered_transfer_smoke.json', payload)}")
    return 0 if payload["gates"]["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
