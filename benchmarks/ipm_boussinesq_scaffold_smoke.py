# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""IPM / Boussinesq scaffold smoke with absolute gates block."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))
from _gates import ipm_boussinesq_scaffold_gates  # noqa: E402

from omnibias.pinn.jax.discovery.pipeline import (  # noqa: E402
    BoussinesqAdapter,
    IPMAdapter,
    run_singularity_pipeline,
)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--family", choices=("ipm", "boussinesq"), default="ipm")
    args = p.parse_args(argv)
    adapter = IPMAdapter(n=10, steps=20) if args.family == "ipm" else BoussinesqAdapter(
        n=10, steps=20
    )
    out = run_singularity_pipeline(adapter, None)
    abs_gates = ipm_boussinesq_scaffold_gates(
        family=args.family,
        max_abs_residual=float(out.discovery["max_abs_residual"]),
        navier_stokes_proof_claim=bool(
            out.certificate.get("honesty", {}).get("navier_stokes_proof_claim", False)
        ),
    )
    payload = {
        "benchmark": f"{args.family}_scaffold_smoke",
        "tier": "cpu_smoke",
        "metrics": {
            "lam": out.discovery["lam"],
            "max_abs_residual": out.discovery["max_abs_residual"],
        },
        "absolute_gates": abs_gates,
        "honesty": {
            "navier_stokes_proof_claim": False,
            "scaffold": True,
        },
        "gates": {
            "earned": abs_gates["earned"],
            "passed": abs_gates["earned"],
        },
    }
    docs = ROOT / "docs" / "benchmarks" / f"{args.family}_scaffold_smoke.json"
    docs.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"artifact": str(docs), "earned": abs_gates["earned"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
