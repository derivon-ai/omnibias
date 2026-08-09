# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""CPU smoke: Hardy-basis line CCF discovery + CAP + absolute gate reporting.

Writes ``docs/benchmarks/ccf_line_smoke.json`` with a ``gates`` block.
Absolute Rung-1 gates (published lambda + residual thresholds) are reported
under ``absolute_gates``; smoke CI passes on infrastructure gates even when
absolute gates are not yet earned.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))
from _gates import ccf_absolute_gates  # noqa: E402

from omnibias.pinn.jax.discovery import cap, ccf_line
from omnibias.symbolic import verify_cap_bundle

DEFAULT_OUT = ROOT / "docs" / "benchmarks" / "ccf_line_smoke.json"


def run_smoke(*, steps: int = 40, n_grid: int = 48, seed: int = 0) -> dict:
    cfg = ccf_line.CCFLineDiscoveryConfig(
        n_terms=4,
        n_grid=n_grid,
        y_max=12.0,
        seed=seed,
        optimizer="adam",
        lam_init=0.6057,
        adaptive_every=20,
    )
    t0 = time.perf_counter()
    result = ccf_line.run_ccf_line_discovery(cfg, steps=steps, lr=5e-3)
    wall = time.perf_counter() - t0
    bundle = cap.build_cap_bundle(result, reproduces_published_lambda=None)
    schema_errors = cap.cap_schema_errors(bundle)
    report = verify_cap_bundle(bundle, atol=1e-5)
    max_abs = float(result.diagnostics["max_abs_residual"])
    finite = bool(
        all(
            abs(float(x)) < 1e6
            for x in (max_abs, result.diagnostics["rms_residual"], result.lam)
        )
    )
    absolute = ccf_absolute_gates(
        lam=float(result.lam),
        max_abs_residual=max_abs,
        family="1st_unstable",
    )
    gates = {
        "schema_ok": schema_errors == [],
        "symbolic_replay_match": bool(report.get("residual_samples_match")),
        "residual_finite": finite,
        "max_abs_residual_below": max_abs < 1e3,
        "navier_stokes_proof_claim": False,
        "absolute_rung1_earned": bool(absolute["earned"]),
        "passed": (
            schema_errors == []
            and bool(report.get("residual_samples_match"))
            and finite
            and max_abs < 1e3
        ),
    }
    return {
        "benchmark": "ccf_line_discovery",
        "tier": "cpu_smoke",
        "config": {
            "steps": steps,
            "n_grid": n_grid,
            "n_terms": cfg.n_terms,
            "y_max": cfg.y_max,
            "seed": seed,
            "lam_init": cfg.lam_init,
            "hilbert": "hardy_exact",
        },
        "metrics": {
            "lam": float(result.lam),
            "max_abs_residual": max_abs,
            "rms_residual": float(result.diagnostics["rms_residual"]),
            "wall_seconds": wall,
            "agreement_max_abs_diff": report.get("agreement_max_abs_diff"),
        },
        "honesty": {
            "exact_solution_claim": False,
            "navier_stokes_proof_claim": False,
            "domain": "line_compactified",
            "reproduces_published_lambda": absolute["honesty"]["reproduces_published_lambda"],
            "hilbert": "hardy_exact",
        },
        "absolute_gates": absolute,
        "gates": gates,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=40)
    p.add_argument("--n-grid", type=int, default=48)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = p.parse_args()
    payload = run_smoke(steps=args.steps, n_grid=args.n_grid, seed=args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload["gates"], indent=2))
    raise SystemExit(0 if payload["gates"]["passed"] else 1)


if __name__ == "__main__":
    main()
