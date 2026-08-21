# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Frontier 07-07a: exact oscillator ladder (tooling, not a many-body solution).

G0 is the textbook QHO energy. G1 is ladder coefficients to 1e-14.
G2 reports no improvement against a competent 1-D Gaussian envelope;
a FermiNet many-body run is heavy, not CI. Jets are founding bias
collapse, not temperature collapse.
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
from omnibias.ferminet.hermite import (
    DISCLAIMER,
    gaussian_envelope_energy,
    honesty_payload,
    ladder_coefficient_errors,
    qho_ground_energy,
    vmc_comparison_report,
)

SCRATCH = Path(os.environ.get("OMNIBIAS_SCRATCH", "artifacts"))


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args(argv)
    full = bool(args.full)
    t0 = time.perf_counter()
    g0 = abs(qho_ground_energy() - 0.5) <= 1e-15 and abs(gaussian_envelope_energy(1.0) - 0.5) <= 1e-15
    errs = ladder_coefficient_errors()
    g1 = bool(errs) and max(errs) <= 2e-14
    g2 = vmc_comparison_report(seeds=5)
    entries = [
        {"name": "g0_qho_baseline", "passed": g0, "energy": 0.5},
        {"name": "g1_ladder_exactness", "passed": g1, "max_err": max(errs) if errs else 1.0},
        {
            "name": "g2_vmc_comparison",
            "passed": bool(g2["no_improvement"]) and abs(float(g2["ladder_energy"]) - 0.5) <= 1.6e-3,
            "no_improvement": g2["no_improvement"],
            "reason": g2["reason"],
        },
    ]
    for entry in entries:
        print(entry["name"], "ok" if entry["passed"] else "FAIL")
        if not entry["passed"]:
            raise AssertionError(f"{entry['name']} failed: {entry}")
    payload = {
        **provenance(
            schema="omnibias.benchmarks.quantum_hermite_vmc.v1",
            config={"family": "quantum_hermite_vmc", "full": full, "honesty": honesty_payload()},
        ),
        "gates": dict(gates_block(entries)),
        "disclaimer": DISCLAIMER,
        "wall_seconds": time.perf_counter() - t0,
    }
    name = "quantum_hermite_vmc.json" if full else "quantum_hermite_vmc_smoke.json"
    if full:
        dest = SCRATCH / "domain_programs"
        dest.mkdir(parents=True, exist_ok=True)
        path = dest / name
        path.write_text(__import__("json").dumps(payload, indent=2) + "\n", encoding="utf-8")
    else:
        path = write_json(name, payload)
    print(f"wrote {path}")
    return payload


if __name__ == "__main__":
    main()
