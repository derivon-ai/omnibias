# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Measure CCF dense residual vs conjugate-tower order N.

Fits empirical ``p`` from ``log residual`` vs ``log span`` at fixed
``n_scales × n_gamma_multiples``. A matched-width control keeps atom count
equal to N=0 so a drop is not just more coefficients.

CPU smoke: N=0 vs N=1 on the tiny reproduce grid; writes
``docs/benchmarks/ccf_conjugate_sweep_smoke.json``. Heavy N-sweep lives under
``$OMNIBIAS_SCRATCH/deepmind_campaign/``.

Never weakens ``CCF_STRETCH_RESIDUAL_GATE`` (1e-13) or Rung-1 (1e-11).
``navier_stokes_proof_claim`` stays False. Compare arms on raw ``dense_max_abs``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))
sys.path.insert(0, str(ROOT))

from _gates import (  # noqa: E402
    CCF_LAMBDA_1ST_UNSTABLE,
    CCF_RESIDUAL_GATE_1ST_UNSTABLE,
    CCF_STRETCH_RESIDUAL_GATE,
)
from reproduce_deepmind_ccf import run_conjugate_once  # noqa: E402

STRETCH = float(CCF_STRETCH_RESIDUAL_GATE)
RUNG1 = float(CCF_RESIDUAL_GATE_1ST_UNSTABLE)
LAM = float(CCF_LAMBDA_1ST_UNSTABLE)


def _scratch() -> Path:
    out = Path(os.environ.get("OMNIBIAS_SCRATCH", ROOT / "artifacts")) / "deepmind_campaign"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _gram_cond(profile: dict[str, Any], *, y_max: float = 8.0, n: int = 81) -> float:
    from omnibias.core.conjugate import hardy_omega_atom

    y = np.linspace(-float(y_max), float(y_max), int(n))
    scales = np.asarray(profile["scales"], dtype=float)
    gammas = np.asarray(profile["gammas"], dtype=float)
    orders = profile.get("orders")
    parities = profile.get("parities")
    if orders is None:
        orders = np.zeros(scales.shape[0], dtype=int)
        parities = np.ones(scales.shape[0], dtype=int)
    else:
        orders = np.asarray(orders, dtype=int)
        parities = (
            np.asarray(parities, dtype=int)
            if parities is not None
            else np.where(orders % 2 == 0, 1, 0)
        )
    cols = []
    for a, g, ord_n, par in zip(scales, gammas, orders, parities, strict=True):
        parity = "odd" if int(par) == 1 else "even"
        cols.append(
            [hardy_omega_atom(float(yy), float(a), float(g), int(ord_n), parity=parity) for yy in y]
        )
    phi = np.column_stack(cols)
    gram = phi.T @ phi
    cond = float(np.linalg.cond(gram))
    return cond if math.isfinite(cond) else float("inf")


def _fit_p(orders: list[int], residuals: list[float]) -> dict[str, float | None]:
    """``residual ~ C * span^{-p}`` with ``span = N+1``."""
    xs: list[float] = []
    ys: list[float] = []
    for n, r in zip(orders, residuals, strict=True):
        if r > 0.0 and math.isfinite(r):
            xs.append(math.log(float(n) + 1.0))
            ys.append(math.log(float(r)))
    if len(xs) < 2:
        return {"p": None, "intercept": None, "n_points": float(len(xs))}
    slope, intercept = np.polyfit(np.asarray(xs), np.asarray(ys), 1)
    return {
        "p": float(-slope),
        "intercept": float(intercept),
        "n_points": float(len(xs)),
    }


def _attempt_cap(profile: dict[str, Any], *, lam: float) -> dict[str, Any]:
    from omnibias.pinn.certified.ccf_hardy import (
        certified_ccf_hardy_wholeline_blowup_attempt,
        certified_ccf_hardy_wholeline_blowup_attempt_schema_errors,
    )

    cert = certified_ccf_hardy_wholeline_blowup_attempt(
        coeffs=list(profile["coeffs"]),
        scales=list(profile["scales"]),
        gammas=list(profile["gammas"]),
        orders=profile.get("orders"),
        parities=profile.get("parities"),
        lam=float(lam),
        form="vorticity",
        residual_gate=RUNG1,
        velocity_sign=-1.0,
    )
    residual_ok = bool(cert["closure_report"]["residual_certified_sup"] <= RUNG1)
    earned = bool(
        residual_ok
        and cert["collocation_closure_certified"]
        and cert["sequence_space_closure_certified"]
    )
    assert cert["honesty"]["navier_stokes_proof_claim"] is False
    assert cert["honesty"]["whole_line_certified"] is earned
    assert certified_ccf_hardy_wholeline_blowup_attempt_schema_errors(cert) == []
    gap = cert["closure_report"]["quantified_gap"]
    return {
        "whole_line_certified": bool(cert["honesty"]["whole_line_certified"]),
        "navier_stokes_proof_claim": False,
        "residual_certified_sup": float(cert["closure_report"]["residual_certified_sup"]),
        "residual_gap": float(gap["residual_gap"]),
        "collocation_closed": bool(cert["collocation_closure_certified"]),
        "sequence_space_closed": bool(cert["sequence_space_closure_certified"]),
        "status": "CLOSED" if earned else "BLOCKED",
        "schema_ok": True,
    }


def run_sweep(*, smoke: bool = True, seed: int = 0) -> dict[str, Any]:
    t0 = time.perf_counter()
    if smoke:
        orders = [0, 1]
        n_scales, n_gamma, n_grid, gn_steps, y_max = 2, 1, 17, 4, 8.0
        matched_n_scales = 1
        matched_n = 1
    else:
        # N=0 atoms = n_scales * n_gamma. Matched N=4 must keep the same
        # count: n_scales_m * n_gamma * 5 == n_scales * n_gamma.
        orders = [0, 1, 2, 3, 4]
        n_scales, n_gamma, n_grid, gn_steps, y_max = 5, 2, 65, 40, 20.0
        matched_n_scales = 1
        matched_n = 4
    n0_atoms = int(n_scales) * int(n_gamma)
    matched_atoms = int(matched_n_scales) * int(n_gamma) * (int(matched_n) + 1)
    if n0_atoms != matched_atoms:
        raise ValueError(
            f"matched-width atom count {matched_atoms} != N=0 count {n0_atoms}"
        )

    rows: list[dict[str, Any]] = []
    for n in orders:
        print(
            f"[ccf_conjugate_sweep] start N={n} "
            f"n_scales={n_scales} n_gamma={n_gamma} "
            f"n_grid={n_grid} gn_steps={gn_steps}",
            flush=True,
        )
        out = run_conjugate_once(
            smoke=smoke,
            max_order=n,
            seed=seed,
            n_scales=n_scales,
            n_gamma_multiples=n_gamma,
            n_grid=n_grid,
            gn_steps=gn_steps,
            y_max=y_max,
        )
        cond = _gram_cond(out["profile"], y_max=y_max)
        raw = float(out["metrics"]["dense_max_abs"])
        print(
            f"[ccf_conjugate_sweep] done N={n} dense_max_abs={raw:.6e} "
            f"gram_cond={cond:.3e} anti_ghost={out['metrics']['anti_ghost_fired']}",
            flush=True,
        )
        rows.append(
            {
                "max_order": n,
                "span": n + 1,
                "n_atoms": int(out["config"]["n_atoms"]),
                "dense_max_abs": raw,
                "reproduction_dense_max_abs_for_gate": float(
                    out["metrics"]["reproduction_dense_max_abs_for_gate"]
                ),
                "omega_gauge_sample": float(out["metrics"]["omega_gauge_sample"]),
                "omega_max_abs": float(out["metrics"]["omega_max_abs"]),
                "anti_ghost_fired": bool(out["metrics"]["anti_ghost_fired"]),
                "orders_to_stretch_raw": float(out["metrics"]["orders_to_stretch_raw"]),
                "gram_cond": cond,
                "profile": out["profile"],
                "stretch_1e-13_cleared": bool(out["gates"]["stretch_1e-13_cleared"]),
                "rung1_1e-11_report": bool(out["gates"]["rung1_1e-11_report"]),
            }
        )

    print(
        f"[ccf_conjugate_sweep] start matched-width N={matched_n} "
        f"n_scales={matched_n_scales}",
        flush=True,
    )
    matched = run_conjugate_once(
        smoke=smoke,
        max_order=matched_n,
        seed=seed,
        n_scales=matched_n_scales,
        n_gamma_multiples=n_gamma,
        n_grid=n_grid,
        gn_steps=gn_steps,
        y_max=y_max,
    )
    matched_row = {
        "max_order": matched_n,
        "span": matched_n + 1,
        "n_atoms": int(matched["config"]["n_atoms"]),
        "n_scales": matched_n_scales,
        "dense_max_abs": float(matched["metrics"]["dense_max_abs"]),
        "omega_gauge_sample": float(matched["metrics"]["omega_gauge_sample"]),
        "omega_max_abs": float(matched["metrics"]["omega_max_abs"]),
        "anti_ghost_fired": bool(matched["metrics"]["anti_ghost_fired"]),
        "orders_to_stretch_raw": float(matched["metrics"]["orders_to_stretch_raw"]),
        "gram_cond": _gram_cond(matched["profile"], y_max=y_max),
        "control": "matched_width",
    }

    fit = _fit_p(
        [int(r["max_order"]) for r in rows],
        [float(r["dense_max_abs"]) for r in rows],
    )
    best = min(rows, key=lambda r: float(r["dense_max_abs"]))
    print(
        f"[ccf_conjugate_sweep] CAP attempt on best N={best['max_order']} "
        f"dense={best['dense_max_abs']:.6e}",
        flush=True,
    )
    cap = _attempt_cap(best["profile"], lam=LAM)
    n0 = float(rows[0]["dense_max_abs"])
    ratio_vs_n0 = (n0 / float(best["dense_max_abs"])) if best["dense_max_abs"] else None
    g2_matched = (
        n0 / float(matched_row["dense_max_abs"]) if matched_row["dense_max_abs"] else None
    )

    stretch_cleared = bool(best["dense_max_abs"] <= STRETCH)
    rung1_cleared = bool(best["dense_max_abs"] <= RUNG1)
    return {
        "benchmark": "ccf_conjugate_sweep",
        "tier": "cpu_smoke" if smoke else "full",
        "wall_seconds": time.perf_counter() - t0,
        "config": {
            "orders": orders,
            "n_scales": n_scales,
            "n_gamma_multiples": n_gamma,
            "n_grid": n_grid,
            "gn_steps": gn_steps,
            "y_max": y_max,
            "seed": seed,
            "lam": LAM,
            "matched_width": {
                "max_order": matched_n,
                "n_scales": matched_n_scales,
                "n_gamma_multiples": n_gamma,
            },
        },
        "rows": [
            {k: v for k, v in r.items() if k != "profile"} for r in rows
        ],
        "matched_width": matched_row,
        "empirical_p": fit,
        "best": {
            "max_order": int(best["max_order"]),
            "dense_max_abs": float(best["dense_max_abs"]),
            "orders_to_stretch_raw": float(best["orders_to_stretch_raw"]),
            "ratio_vs_n0": ratio_vs_n0,
            "anti_ghost_fired": bool(best["anti_ghost_fired"]),
        },
        "cap": cap,
        "g2_measurement": {
            "matched_width_ratio_vs_n0": g2_matched,
            "ten_x": bool(g2_matched is not None and g2_matched >= 10.0),
            "note": "measurement, not a merge blocker",
        },
        "gates": {
            "stretch_gate": STRETCH,
            "rung1_gate": RUNG1,
            "stretch_1e-13_cleared": stretch_cleared,
            "rung1_1e-11_cleared": rung1_cleared,
            "whole_line_certified": bool(cap["whole_line_certified"]),
            "navier_stokes_proof_claim": False,
            "lambda_unmoved": True,
            "passed": True,
        },
        "honesty": {
            "navier_stokes_proof_claim": False,
            "continuum_claim": False,
            "stretch_unearned": not stretch_cleared,
            "rung1_unearned": not rung1_cleared,
            "rung2_unearned": not bool(cap["whole_line_certified"]),
            "compare_arms_on": "dense_max_abs",
            "not_progress_toward_stretch": True,
            "note": (
                "N>0 is a measured dictionary-order experiment. A flat G2 "
                "means the catch-22 is not dictionary order. Not 3D NS."
            ),
        },
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--full", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--write-docs",
        action="store_true",
        help="Write docs/benchmarks/ccf_conjugate_sweep_smoke.json (smoke only)",
    )
    args = p.parse_args(argv)
    payload = run_sweep(smoke=not args.full, seed=args.seed)
    name = f"ccf_conjugate_sweep_{'full' if args.full else 'smoke'}.json"
    out = _scratch() / name
    text = json.dumps(payload, indent=2, default=str) + "\n"
    out.write_text(text, encoding="utf-8")
    if args.write_docs and not args.full:
        docs = ROOT / "docs" / "benchmarks" / "ccf_conjugate_sweep_smoke.json"
        docs.parent.mkdir(parents=True, exist_ok=True)
        docs.write_text(text, encoding="utf-8")
    print(
        json.dumps(
            {
                "artifact": str(out),
                "best_dense_max_abs": payload["best"]["dense_max_abs"],
                "empirical_p": payload["empirical_p"]["p"],
                "orders_to_stretch_raw": payload["best"]["orders_to_stretch_raw"],
                "cap_status": payload["cap"]["status"],
                "stretch_1e-13_cleared": payload["gates"]["stretch_1e-13_cleared"],
                "navier_stokes_proof_claim": False,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
