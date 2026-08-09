# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""CCF Hardy Rung-1 / Rung-2 acceptance ladder (CPU).

Runs vorticity-form GN discovery → scale-free refine → mpmath polish → dense
vorticity validation gates → whole-line CAP. Writes acceptance JSON under
``docs/benchmarks/`` only when absolute gates are earned.

Rung-1 metric is the Wang vorticity residual on a dense fixed linspace
(``max|Ω + ((1+λ)y−U)Ω_y − Ω U_y|``), not the even-Θ transport residual
(which is structurally nonzero at ``y=0`` whenever ``Θ(0)≠0``).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))
from _gates import (  # noqa: E402
    CCF_LAMBDA_1ST_UNSTABLE,
    CCF_LAMBDA_2ND_UNSTABLE,
    ccf_absolute_gates,
)

from omnibias.pinn.certified.ccf_hardy import (  # noqa: E402
    certified_ccf_hardy_wholeline_blowup_attempt,
    certified_ccf_hardy_wholeline_blowup_attempt_schema_errors,
    refine_ccf_hardy_profile,
)
from omnibias.pinn.jax.discovery import ccf_vorticity, polish_mp  # noqa: E402
from omnibias.pinn.jax.equations.ccf_compactified import alpha_from_lambda  # noqa: E402


def _dense_vorticity(
    coeffs: np.ndarray,
    scales: np.ndarray,
    alphas: np.ndarray,
    lam: float,
    *,
    n_val: int = 4001,
    y_max: float = 40.0,
) -> dict[str, float]:
    return ccf_vorticity.dense_vorticity_residual(
        coeffs, scales, alphas, lam, n_val=n_val, y_max=y_max
    )


def _family_target(family: str) -> float:
    if family == "1st_unstable":
        return CCF_LAMBDA_1ST_UNSTABLE
    if family == "2nd_unstable":
        return CCF_LAMBDA_2ND_UNSTABLE
    raise ValueError(family)


def run_acceptance(
    *,
    family: str = "1st_unstable",
    n_scales: int = 12,
    n_alpha_offsets: int = 3,
    n_grid: int = 256,
    y_max: float = 40.0,
    gn_steps: int = 120,
    polish_dps: int = 40,
    polish_iters: int = 25,
    seed: int = 0,
) -> dict:
    target = _family_target(family)
    cfg = ccf_vorticity.CCFVorticityDiscoveryConfig(
        n_scales=n_scales,
        n_alpha_offsets=n_alpha_offsets,
        n_grid=n_grid,
        y_max=y_max,
        seed=seed,
        lam=target,
        gn_steps=gn_steps,
        enforce_theta0=True,
    )
    t0 = time.perf_counter()
    disc = ccf_vorticity.run_ccf_vorticity_discovery(cfg, steps=gn_steps)
    coeffs = np.asarray(disc.coeffs, dtype=float)
    scales = np.asarray(disc.scales, dtype=float)
    alphas = np.asarray(disc.alphas, dtype=float)
    lam = float(disc.lam)

    # Collapse multi-alpha to single-alpha CAP/refine path by summing
    # same-scale groups is not exact; refine uses primary alpha = 1/(1+lam).
    # For CAP we keep the primary-alpha Hardy even profile (first offset slot).
    alpha0 = float(alpha_from_lambda(lam))
    # Aggregate coefficients onto unique scales at alpha0 by taking offset-0 slots.
    n_off = n_alpha_offsets
    scales_u = scales[::n_off]
    coeffs_u = coeffs[::n_off].copy()
    # Fold higher-alpha energy into primary slots (honest: approximate).
    for k in range(1, n_off):
        coeffs_u = coeffs_u + coeffs[k::n_off]

    n_terms = int(scales_u.size)
    n_free = (n_terms - 1) + n_terms + 1
    nodes = [0.12 * (1.48**k) for k in range(n_free + 8)]
    refined = refine_ccf_hardy_profile(
        coeffs=coeffs_u.tolist(),
        scales=scales_u.tolist(),
        lam=lam,
        nodes=nodes,
        free_scales=True,
        free_lam=False,
        lam_target=target,
        iters=100,
        tol=1e-14,
        velocity_sign=-1.0,
        omega_gauge_point=0.5,
        omega_gauge_value=0.05,
        min_scale=0.2,
        max_scale=20.0,
    )
    coeffs_u = np.asarray(refined["coeffs"], dtype=float)
    scales_u = np.asarray(refined["scales"], dtype=float)
    lam = float(refined["lam"])
    alphas_u = np.full_like(scales_u, alpha_from_lambda(lam))

    polished = polish_mp.polish_hardy_ccf(
        coeffs=coeffs_u,
        scales=scales_u,
        lam=lam,
        nodes=np.asarray(nodes, dtype=float),
        dps=polish_dps,
        max_iter=polish_iters,
        free_lam=False,
        velocity_sign=-1.0,
    )
    coeffs_u = np.asarray(polished["coeffs"], dtype=float)
    scales_u = np.asarray(polished["scales"], dtype=float)
    lam = float(polished["lam"])
    alphas_u = np.full_like(scales_u, float(alpha_from_lambda(lam)))

    refined2 = refine_ccf_hardy_profile(
        coeffs=coeffs_u.tolist(),
        scales=scales_u.tolist(),
        lam=lam,
        nodes=nodes,
        free_scales=True,
        free_lam=False,
        lam_target=target,
        iters=80,
        tol=1e-14,
        velocity_sign=-1.0,
        omega_gauge_point=0.5,
        omega_gauge_value=0.05,
        min_scale=0.2,
        max_scale=20.0,
    )
    coeffs_u = np.asarray(refined2["coeffs"], dtype=float)
    scales_u = np.asarray(refined2["scales"], dtype=float)
    lam = float(refined2["lam"])
    alphas_u = np.full_like(scales_u, float(alpha_from_lambda(lam)))

    dense = _dense_vorticity(coeffs_u, scales_u, alphas_u, lam, y_max=y_max)
    # Reject near-null / gauge-lost ghosts: tiny residual with vanished Omega is not a win.
    gauge_ok = abs(float(dense.get("omega_gauge_sample", 0.0)) - 0.05) <= 0.01
    nontrivial = float(dense.get("omega_max_abs", 0.0)) >= 0.02
    residual_for_gate = float(dense["dense_max_abs_vorticity"])
    if not (gauge_ok and nontrivial):
        residual_for_gate = max(residual_for_gate, 1.0)  # force residual gate fail
    absolute = ccf_absolute_gates(
        lam=lam,
        max_abs_residual=residual_for_gate,
        family=family,
        stretch_mp_residual=float(polished.get("max_abs_residual_mpmath", float("nan"))),
    )

    from omnibias.pinn.certified.navier_stokes import default_ccf_collocation_nodes

    cap_nodes = list(default_ccf_collocation_nodes(n_terms))
    cert = certified_ccf_hardy_wholeline_blowup_attempt(
        coeffs=coeffs_u.tolist(),
        scales=scales_u.tolist(),
        lam=lam,
        nodes=cap_nodes,
        residual_gate=1e-11 if family == "1st_unstable" else 1e-6,
        velocity_sign=-1.0,
    )
    schema_errs = certified_ccf_hardy_wholeline_blowup_attempt_schema_errors(cert)
    wall = time.perf_counter() - t0

    return {
        "benchmark": "ccf_hardy_rung_acceptance",
        "family": family,
        "tier": "cpu_acceptance",
        "config": {
            "n_scales": n_scales,
            "n_alpha_offsets": n_alpha_offsets,
            "n_grid": n_grid,
            "y_max": y_max,
            "gn_steps": gn_steps,
            "target_lam": target,
            "seed": seed,
            "hilbert": "hardy_exact_theta",
            "residual_form": "wang_vorticity",
            "train_lam": False,
        },
        "metrics": {
            "lam": lam,
            "collocation_residual_max_abs": float(refined2["residual_max_abs"]),
            "dense_max_abs_vorticity": dense["dense_max_abs_vorticity"],
            "dense_rms_vorticity": dense["dense_rms_vorticity"],
            "dense_max_abs_vorticity_for_gate": residual_for_gate,
            "theta0": dense.get("theta0", float("nan")),
            "omega_gauge_sample": dense.get("omega_gauge_sample", float("nan")),
            "omega_max_abs": dense.get("omega_max_abs", float("nan")),
            "gauge_ok": bool(gauge_ok),
            "nontrivial_profile": bool(nontrivial),
            "mpmath_max_abs_residual": float(
                polished.get("max_abs_residual_mpmath", float("nan"))
            ),
            "wall_seconds": wall,
            "discovery_train_max_abs_vorticity": float(
                disc.diagnostics["max_abs_vorticity_residual"]
            ),
        },
        "absolute_gates": absolute,
        "rung2": {
            "whole_line_certified": bool(cert["honesty"]["whole_line_certified"]),
            "closure_certified": bool(cert["closure_certified"]),
            "residual_certified_sup": cert["closure_report"]["residual_certified_sup"],
            "schema_ok": schema_errs == [],
            "quantified_gap": cert["closure_report"].get("quantified_gap"),
        },
        "profile": {
            "coeffs": coeffs_u.tolist(),
            "scales": scales_u.tolist(),
            "lam": lam,
            "alpha": float(alpha_from_lambda(lam)),
        },
        "honesty": {
            "navier_stokes_proof_claim": False,
            "reproduces_published_lambda": absolute["honesty"][
                "reproduces_published_lambda"
            ],
            "rung1_earned": bool(absolute["earned"]),
            "rung2_earned": bool(cert["honesty"]["whole_line_certified"]),
            "residual_metric": "wang_vorticity_dense_linspace",
            "measured_gap_note": (
                "Hardy finite dictionaries empirically floor near ~7e-2 absolute "
                "vorticity residual under Omega(0.5)=0.05; gates stay unearned "
                "until dense residual clears published thresholds."
            ),
        },
        "gates": {
            "rung1_earned": bool(absolute["earned"]),
            "rung2_earned": bool(cert["honesty"]["whole_line_certified"]),
            "schema_ok": schema_errs == [],
            "navier_stokes_proof_claim": False,
            "passed": bool(
                absolute["earned"] and cert["honesty"]["whole_line_certified"]
            ),
        },
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--family", choices=("1st_unstable", "2nd_unstable"), default="1st_unstable"
    )
    p.add_argument("--n-scales", type=int, default=12)
    p.add_argument("--n-alpha-offsets", type=int, default=3)
    p.add_argument("--n-grid", type=int, default=256)
    p.add_argument("--y-max", type=float, default=40.0)
    p.add_argument("--gn-steps", type=int, default=120)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()
    out = args.out or (
        ROOT / "docs" / "benchmarks" / f"ccf_line_rung1_{args.family}.json"
    )
    payload = run_acceptance(
        family=args.family,
        n_scales=args.n_scales,
        n_alpha_offsets=args.n_alpha_offsets,
        n_grid=args.n_grid,
        y_max=args.y_max,
        gn_steps=args.gn_steps,
        seed=args.seed,
    )
    if payload["gates"]["rung1_earned"]:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {out}")
    # Always write a scratch status report (not an earned acceptance artifact).
    scratch = Path(
        __import__("os").environ.get("OMNIBIAS_SCRATCH", str(ROOT / "artifacts"))
    )
    scratch.mkdir(parents=True, exist_ok=True)
    status_path = scratch / f"ccf_rung_status_{args.family}.json"
    status_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"status {status_path}")
    print(
        json.dumps(
            {
                "lam": payload["metrics"]["lam"],
                "dense_max_abs_vorticity": payload["metrics"][
                    "dense_max_abs_vorticity"
                ],
                "rung1_earned": payload["gates"]["rung1_earned"],
                "rung2_earned": payload["gates"]["rung2_earned"],
                "residual_certified_sup": payload["rung2"]["residual_certified_sup"],
            },
            indent=2,
        )
    )
    raise SystemExit(0 if payload["gates"]["passed"] else 1)


if __name__ == "__main__":
    main()
