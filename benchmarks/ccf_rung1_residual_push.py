# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Escalate Martens–Grosse / refine / polish until dense Wang residual ≤ 1e-11.

Writes progress under ``$OMNIBIAS_SCRATCH/deepmind_campaign/``. Never weakens
the absolute gate. Adam is not used.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from omnibias.pinn.certified.ccf_hardy import refine_ccf_hardy_profile  # noqa: E402
from omnibias.pinn.jax.discovery import polish_mp  # noqa: E402
from omnibias.pinn.jax.discovery.ccf_vorticity import (  # noqa: E402
    CCFVorticityDiscoveryConfig,
    dense_vorticity_residual,
    run_ccf_vorticity_discovery,
)

GATE = 1e-11
LAM = 0.6057
ROOT = Path(__file__).resolve().parents[1]


def _scratch() -> Path:
    out = Path(os.environ.get("OMNIBIAS_SCRATCH", ROOT / "artifacts")) / "deepmind_campaign"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _anti_ghost(dense: dict[str, float]) -> float:
    r = float(dense["dense_max_abs_vorticity"])
    gauge_ok = abs(float(dense["omega_gauge_sample"]) - 0.05) <= 0.01
    nontrivial = float(dense["omega_max_abs"]) >= 0.02
    if not (gauge_ok and nontrivial):
        return max(r, 1.0)
    return r


def _evaluate(coeffs, scales, gammas, lam: float, y_max: float) -> dict[str, float]:
    return dense_vorticity_residual(coeffs, scales, gammas, lam, n_val=4001, y_max=y_max)


def _refine_safe(coeffs, scales, gammas, lam, *, iters: int, free_scales: bool) -> dict:
    n = len(coeffs)
    nodes = [0.12 * (1.48**k) for k in range(n + 12)]
    return refine_ccf_hardy_profile(
        coeffs=list(map(float, coeffs)),
        scales=list(map(float, scales)),
        lam=float(lam),
        nodes=nodes if free_scales else None,
        form="vorticity",
        gammas=list(map(float, gammas)),
        free_scales=free_scales,
        free_lam=False,
        lam_target=LAM,
        iters=iters,
        tol=1e-14,
        velocity_sign=-1.0,
        omega_gauge_point=0.5,
        omega_gauge_value=0.05,
        min_scale=0.2,
        max_scale=10.0,
    )


def push_once(
    *,
    n_scales: int,
    n_gamma: int,
    n_grid: int,
    gn_steps: int,
    seed: int,
    free_gamma: bool,
    near_power: float,
    hard_gauge: bool,
    y_max: float,
    refine_iters: int,
    polish_iters: int,
    independent_terms: bool = False,
    n_terms: int = 16,
) -> dict:
    if independent_terms:
        cfg = CCFVorticityDiscoveryConfig(
            independent_terms=True,
            n_terms=n_terms,
            n_grid=n_grid,
            gn_steps=gn_steps,
            y_max=y_max,
            seed=seed,
            lam=LAM,
            gauge_weight=120.0,
            hard_gauge_rescale=hard_gauge,
            near_field_power=near_power,
            scale_lo=0.15,
            scale_hi=12.0,
            coeff_l2=1e-8,
            gn_gamma=1e-4,
        )
    else:
        cfg = CCFVorticityDiscoveryConfig(
            n_scales=n_scales,
            n_gamma_multiples=n_gamma,
            n_grid=n_grid,
            gn_steps=gn_steps,
            y_max=y_max,
            seed=seed,
            lam=LAM,
            gauge_weight=120.0,
            free_gamma_offsets=free_gamma,
            hard_gauge_rescale=hard_gauge,
            near_field_power=near_power,
            scale_lo=0.2,
            scale_hi=10.0,
            coeff_l2=1e-8,
            gn_gamma=1e-4,
        )
    t0 = time.perf_counter()
    disc = run_ccf_vorticity_discovery(cfg)
    best = {
        "coeffs": np.asarray(disc.coeffs, dtype=float),
        "scales": np.asarray(disc.scales, dtype=float),
        "gammas": np.asarray(disc.alphas, dtype=float),
        "lam": float(disc.lam),
    }
    dense = _evaluate(best["coeffs"], best["scales"], best["gammas"], best["lam"], y_max)
    residual = _anti_ghost(dense)
    stage = "gn"
    hist = [{"stage": stage, "residual": residual, **dense}]

    # Fixed-scale refine when already in a moderate basin.
    if residual < 0.2:
        ref = _refine_safe(
            best["coeffs"],
            best["scales"],
            best["gammas"],
            best["lam"],
            iters=refine_iters,
            free_scales=False,
        )
        cand = {
            "coeffs": np.asarray(ref["coeffs"], dtype=float),
            "scales": np.asarray(ref["scales"], dtype=float),
            "gammas": np.asarray(ref.get("gammas", best["gammas"]), dtype=float),
            "lam": float(ref["lam"]),
        }
        d2 = _evaluate(cand["coeffs"], cand["scales"], cand["gammas"], cand["lam"], y_max)
        r2 = _anti_ghost(d2)
        hist.append({"stage": "refine_fixed_scales", "residual": r2, **d2})
        if r2 < residual:
            best, residual, dense = cand, r2, d2

    if residual < 0.05:
        ref = _refine_safe(
            best["coeffs"],
            best["scales"],
            best["gammas"],
            best["lam"],
            iters=max(40, refine_iters // 2),
            free_scales=True,
        )
        cand = {
            "coeffs": np.asarray(ref["coeffs"], dtype=float),
            "scales": np.asarray(ref["scales"], dtype=float),
            "gammas": np.asarray(ref.get("gammas", best["gammas"]), dtype=float),
            "lam": float(ref["lam"]),
        }
        d2 = _evaluate(cand["coeffs"], cand["scales"], cand["gammas"], cand["lam"], y_max)
        r2 = _anti_ghost(d2)
        hist.append({"stage": "refine_free_scales", "residual": r2, **d2})
        if r2 < residual and r2 < 1.0:
            best, residual, dense = cand, r2, d2

    if residual < 0.01 and polish_iters > 0:
        nodes = np.asarray([0.12 * (1.48**k) for k in range(len(best["coeffs"]) + 8)])
        pol = polish_mp.polish_hardy_ccf(
            coeffs=best["coeffs"],
            scales=best["scales"],
            lam=best["lam"],
            nodes=nodes,
            form="vorticity",
            gammas=best["gammas"],
            dps=40,
            max_iter=polish_iters,
            free_lam=False,
            velocity_sign=-1.0,
        )
        cand = {
            "coeffs": np.asarray(pol["coeffs"], dtype=float),
            "scales": np.asarray(pol["scales"], dtype=float),
            "gammas": np.asarray(pol.get("gammas", best["gammas"]), dtype=float),
            "lam": float(pol["lam"]),
        }
        d2 = _evaluate(cand["coeffs"], cand["scales"], cand["gammas"], cand["lam"], y_max)
        r2 = _anti_ghost(d2)
        hist.append(
            {
                "stage": "mpmath_polish",
                "residual": r2,
                "mp_resid": pol.get("max_abs_residual_mpmath"),
                **d2,
            }
        )
        if r2 < residual:
            best, residual, dense = cand, r2, d2

    return {
        "residual_for_gate": residual,
        "dense": dense,
        "profile": {
            "coeffs": best["coeffs"].tolist(),
            "scales": best["scales"].tolist(),
            "gammas": best["gammas"].tolist(),
            "lam": best["lam"],
        },
        "history": hist,
        "config": {
            "n_scales": n_scales,
            "n_gamma": n_gamma,
            "n_grid": n_grid,
            "gn_steps": gn_steps,
            "seed": seed,
            "free_gamma": free_gamma,
            "near_power": near_power,
            "hard_gauge": hard_gauge,
            "y_max": y_max,
            "independent_terms": independent_terms,
            "n_terms": n_terms,
        },
        "wall_seconds": time.perf_counter() - t0,
        "rung1_earned": bool(residual <= GATE),
        "navier_stokes_proof_claim": False,
    }


def escalate_loop(*, max_rounds: int = 12) -> dict:
    """Escalating search; keep best residual; stop at gate."""
    # Prefer independent Hardy atoms (flexible γ) — rigid kα grid stalls ~7e-2.
    schedules = [
        # independent, n_terms, n_grid, gn_steps, near, hard_gauge, y_max, refine, polish
        ("indep", 12, 201, 100, 0.5, True, 40.0, 80, 0),
        ("indep", 16, 257, 150, 1.0, True, 40.0, 100, 10),
        ("indep", 20, 301, 200, 1.0, False, 40.0, 120, 15),
        ("indep", 24, 351, 250, 1.5, True, 50.0, 140, 20),
        ("indep", 32, 401, 300, 1.5, True, 50.0, 160, 25),
        ("indep", 40, 451, 350, 2.0, False, 60.0, 180, 30),
        ("grid", (12, 6), 301, 200, 1.0, True, 40.0, 120, 15),
        ("grid", (16, 8), 401, 250, 1.5, True, 40.0, 140, 20),
    ]
    seeds = [0, 1, 2, 7]
    best: dict | None = None
    rounds: list[dict] = []
    round_i = 0
    for sched in schedules:
        for seed in seeds:
            if round_i >= max_rounds:
                break
            kind = sched[0]
            if kind == "indep":
                _, n_terms, n_grid, gn_steps, near_power, hard_gauge, y_max, refine_iters, polish_iters = sched
                n_scales, n_gamma, free_gamma = 0, 0, False
                independent_terms = True
                print(
                    f"ROUND {round_i}: indep terms={n_terms} grid={n_grid} "
                    f"steps={gn_steps} seed={seed}",
                    flush=True,
                )
            else:
                _, (n_scales, n_gamma), n_grid, gn_steps, near_power, hard_gauge, y_max, refine_iters, polish_iters = sched
                n_terms = 0
                free_gamma = True
                independent_terms = False
                print(
                    f"ROUND {round_i}: grid scales={n_scales} gamma={n_gamma} "
                    f"grid={n_grid} steps={gn_steps} seed={seed}",
                    flush=True,
                )
            out = push_once(
                n_scales=n_scales,
                n_gamma=n_gamma,
                n_grid=n_grid,
                gn_steps=gn_steps,
                seed=seed,
                free_gamma=free_gamma,
                near_power=near_power,
                hard_gauge=hard_gauge,
                y_max=y_max,
                refine_iters=refine_iters,
                polish_iters=polish_iters,
                independent_terms=independent_terms,
                n_terms=n_terms,
            )
            print(
                f"  -> residual={out['residual_for_gate']:.6e} "
                f"gauge={out['dense']['omega_gauge_sample']:.5f} "
                f"wall={out['wall_seconds']:.1f}s",
                flush=True,
            )
            rounds.append(out)
            if best is None or out["residual_for_gate"] < best["residual_for_gate"]:
                best = out
                (_scratch() / "rung1_best_profile.json").write_text(
                    json.dumps(best, indent=2) + "\n", encoding="utf-8"
                )
            (_scratch() / "rung1_push_progress.json").write_text(
                json.dumps(
                    {
                        "best_residual": best["residual_for_gate"],
                        "gate": GATE,
                        "rung1_earned": best["rung1_earned"],
                        "rounds_completed": round_i + 1,
                        "navier_stokes_proof_claim": False,
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            if out["rung1_earned"]:
                return {"best": best, "rounds": rounds, "stopped": "gate_earned"}
            round_i += 1
        if round_i >= max_rounds:
            break
    return {"best": best, "rounds": rounds, "stopped": "budget_exhausted"}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--max-rounds", type=int, default=20)
    args = p.parse_args(argv)
    result = escalate_loop(max_rounds=int(args.max_rounds))
    best = result["best"]
    assert best is not None
    summary = {
        "stopped": result["stopped"],
        "best_residual_for_gate": best["residual_for_gate"],
        "gate": GATE,
        "rung1_earned": best["rung1_earned"],
        "orders_to_gate": float(np.log10(best["residual_for_gate"] / GATE))
        if best["residual_for_gate"] > 0
        else None,
        "navier_stokes_proof_claim": False,
        "artifact": str(_scratch() / "rung1_best_profile.json"),
    }
    (_scratch() / "rung1_push_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)
    return 0 if best["rung1_earned"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
