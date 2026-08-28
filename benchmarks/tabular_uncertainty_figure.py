# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Theory 05-03 Figure-5-style diagnostic panel + Figure-1-style decile curve.

Qualitatively reproduces Figure 5 of Kartashev, Rubachev & Babenko
(arXiv:2509.04430) on the paper's own ``saw_wave_2d`` synthetic set (a dense
per-arm prediction grid, colored on the same ``0..1`` scale, next to the known
noise profile ``s(x2)``), plus the quantitative twin that actually matters: test
rows sorted by the *true* uncertainty, per-row squared-error difference between
the noise-damped (``lam>0``) and plain (``lam=0``) omnibias arms, smoothed with
``scipy.ndimage.gaussian_filter1d`` -- matching the source paper's Figure 1 /
Appendix B exactly.

**This script is a diagnostic, not a gate.** It writes no ``passed`` key and is
never consulted by ``gates["all_passed"]`` anywhere. Gate G1 (the actual
falsifier, theory 05-03 section 8) is evaluated separately, on a full multi-seed
``lam``-selection sweep, by ``benchmarks/tabular_uncertainty.py`` (Phase 2). This
script exists to produce the figure honestly, not to decide anything.

Writes:

* ``docs/benchmarks/saw_wave_figure5_smoke.json`` -- small, committed: per-arm
  top-decile RMSE, the raw + smoothed decile curve, and run metadata.
* ``$OMNIBIAS_SCRATCH/tabular_uncertainty/saw_wave_grid.npz`` -- the dense
  per-arm prediction grid (too large to commit); ``docs/img/generate_figures.py``
  reads this to render the Figure-5-style panel without re-training, so
  re-rendering the PNG never requires re-running this script.

Usage::

    python benchmarks/tabular_uncertainty_figure.py [--n 4000] [--grid 220] \
        [--lam 3.0] [--seed 0] [--steps 80]
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from _common import provenance, write_json  # type: ignore[import-not-found]  # noqa: E402

SCRATCH = Path(os.environ.get("OMNIBIAS_SCRATCH", "artifacts"))
GRID_DIR = SCRATCH / "tabular_uncertainty"

N_DEFAULT = 4000
GRID_N_DEFAULT = 220
LAM_DEFAULT = 3.0
SEED_DEFAULT = 0
STEPS_DEFAULT = 80
N_TREES = 32
DEPTH = 2
BETA_FINAL = 16.0
N_DECILES = 10


def _fit_omnibias(
    Xtr: np.ndarray,
    ytr: np.ndarray,
    *,
    lam: float,
    log_scale: np.ndarray | None,
    seed: int,
    steps: int,
) -> Any:
    import torch
    from omnibias.tab._core.config import SoftTreeConfig
    from omnibias.tab.torch.heteroscedastic import fit_noise_aware
    from omnibias.tab.torch.model import SoftTreeEnsemble
    from omnibias.tab.torch.train import fit_second_order

    cfg = SoftTreeConfig(
        n_features=Xtr.shape[1],
        n_trees=N_TREES,
        depth=DEPTH,
        task="regression",
        n_outputs=1,
        beta_init=1.0,
        beta_final=BETA_FINAL,
        anneal_steps=steps,
        seed=seed,
    )
    torch.manual_seed(seed)
    model = SoftTreeEnsemble(cfg)
    if lam <= 0.0:
        fit_second_order(model, Xtr, ytr, optimizer="trust_region", steps=steps)
    else:
        assert log_scale is not None
        fit_noise_aware(
            model, Xtr, ytr, log_scale=log_scale, lam=lam, optimizer="trust_region", steps=steps
        )
    return model


def _top_decile_rmse(err2: np.ndarray, rank: np.ndarray, frac: float = 0.1) -> float:
    r"""RMSE over the top ``frac`` of rows by ``rank`` (descending -- highest first)."""
    n = err2.shape[0]
    k = max(1, int(round(n * frac)))
    order = np.argsort(-rank)
    return float(np.sqrt(np.mean(err2[order[:k]])))


def _decile_curve(
    err2_lam0: np.ndarray, err2_lamP: np.ndarray, rank: np.ndarray, *, n_bins: int = N_DECILES
) -> dict[str, Any]:
    r"""Binned + smoothed ``err2_lamP - err2_lam0`` vs uncertainty decile (0=lowest, last=highest).

    Negative values mean the noise-damped (``lam>0``) arm has *lower* squared error than
    the plain (``lam=0``) arm in that decile -- matching the source paper's Figure 1 /
    Appendix B sign convention (their curve reads as "gap the DL model closes").
    """
    order = np.argsort(rank)
    e0 = err2_lam0[order]
    eP = err2_lamP[order]
    n = order.shape[0]
    edges = np.linspace(0, n, n_bins + 1).astype(int)
    raw = [
        float(np.mean(eP[edges[i] : edges[i + 1]]) - np.mean(e0[edges[i] : edges[i + 1]]))
        if edges[i + 1] > edges[i]
        else 0.0
        for i in range(n_bins)
    ]
    raw_arr = np.asarray(raw, dtype=np.float64)
    try:
        from scipy.ndimage import gaussian_filter1d

        smoothed = gaussian_filter1d(raw_arr, sigma=1.0, mode="nearest")
    except ImportError:  # pragma: no cover - scipy is a standard dev dependency here
        smoothed = raw_arr
    return {
        "raw_diff_err2_lamP_minus_lam0": raw_arr.tolist(),
        "smoothed_diff_err2_lamP_minus_lam0": smoothed.tolist(),
        "n_bins": n_bins,
        "decile_order": "0 = lowest true uncertainty, n_bins-1 = highest true uncertainty",
        "sign_convention": (
            "negative means lam>0 (noise-damped) has lower squared error than lam=0 "
            "in that decile"
        ),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=N_DEFAULT, help="saw_wave_2d row count")
    parser.add_argument("--grid", type=int, default=GRID_N_DEFAULT, help="grid side length")
    parser.add_argument("--lam", type=float, default=LAM_DEFAULT, help="noise-damping lam>0")
    parser.add_argument("--seed", type=int, default=SEED_DEFAULT)
    parser.add_argument("--steps", type=int, default=STEPS_DEFAULT, help="omnibias training steps")
    args = parser.parse_args(argv)

    from omnibias.tab.bench import (
        fit_predict_catboost_uncertainty,
        fit_predict_lightgbm,
        saw_wave_2d,
        saw_wave_f,
        saw_wave_s,
    )
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    t0 = time.perf_counter()
    ds = saw_wave_2d(args.n, seed=args.seed)
    assert ds.f_clean is not None and ds.s_true is not None
    Xtr_raw, Xte_raw, ytr, yte, _ftr, _fte, _str, ste = train_test_split(
        ds.X, ds.y, ds.f_clean, ds.s_true, test_size=0.3, random_state=args.seed
    )
    scaler = StandardScaler().fit(Xtr_raw)
    Xtr, Xte = scaler.transform(Xtr_raw), scaler.transform(Xte_raw)

    g = int(args.grid)
    x1_lin = np.linspace(0.0, 1.0, g)
    x2_lin = np.linspace(0.0, 10.0, g)
    X1, X2 = np.meshgrid(x1_lin, x2_lin)  # X2 varies down rows -- the "vertical" axis
    grid_raw = np.column_stack([X1.ravel(), X2.ravel()])
    grid_scaled = scaler.transform(grid_raw)
    truth_grid = saw_wave_f(X1, X2)
    s_profile = saw_wave_s(x2_lin)

    # One CatBoost fit, queried on train (for the penalty weight) + test + grid at once.
    query = np.vstack([Xtr, Xte, grid_scaled])
    _, log_scale_all = fit_predict_catboost_uncertainty(Xtr, ytr, query, seed=args.seed)
    n_tr, n_te = Xtr.shape[0], Xte.shape[0]
    log_scale_tr = log_scale_all[:n_tr]
    log_scale_te = log_scale_all[n_tr : n_tr + n_te]

    # One LightGBM fit, queried on test + grid ("tuned GBDT" reference column).
    lgbm_pred, _ = fit_predict_lightgbm(
        Xtr, ytr, np.vstack([Xte, grid_scaled]), task="regression", n_outputs=1, seed=args.seed
    )
    lgbm_pred_grid = lgbm_pred[n_te:]

    model_lam0 = _fit_omnibias(Xtr, ytr, lam=0.0, log_scale=None, seed=args.seed, steps=args.steps)
    model_lamP = _fit_omnibias(
        Xtr, ytr, lam=args.lam, log_scale=log_scale_tr, seed=args.seed, steps=args.steps
    )
    pred_lam0_te = model_lam0.score(Xte, beta=BETA_FINAL)[:, 0]
    pred_lam0_grid = model_lam0.score(grid_scaled, beta=BETA_FINAL)[:, 0]
    pred_lamP_te = model_lamP.score(Xte, beta=BETA_FINAL)[:, 0]
    pred_lamP_grid = model_lamP.score(grid_scaled, beta=BETA_FINAL)[:, 0]

    err2_lam0_te = (pred_lam0_te - yte) ** 2
    err2_lamP_te = (pred_lamP_te - yte) ** 2
    top_decile_lam0 = _top_decile_rmse(err2_lam0_te, ste)
    top_decile_lamP = _top_decile_rmse(err2_lamP_te, ste)
    decile_curve = _decile_curve(err2_lam0_te, err2_lamP_te, ste)

    from scipy.stats import spearmanr

    rho, _ = spearmanr(np.exp(log_scale_te), ste)

    GRID_DIR.mkdir(parents=True, exist_ok=True)
    grid_path = GRID_DIR / "saw_wave_grid.npz"
    np.savez(
        grid_path,
        x1=x1_lin,
        x2=x2_lin,
        truth=truth_grid,
        lightgbm=lgbm_pred_grid.reshape(g, g),
        lam0=pred_lam0_grid.reshape(g, g),
        lamP=pred_lamP_grid.reshape(g, g),
        s_profile=s_profile,
        lam=np.array([args.lam]),
    )

    config = {
        "family": "saw_wave_figure5_diagnostic",
        "dataset": "saw_wave_2d",
        "n": args.n,
        "n_train": int(n_tr),
        "n_test": int(n_te),
        "grid": g,
        "lam": args.lam,
        "seed": args.seed,
        "steps": args.steps,
        "n_trees": N_TREES,
        "depth": DEPTH,
        "beta_final": BETA_FINAL,
    }
    payload = provenance(schema="tabular-uncertainty-figure5-v1", config=config)
    payload.update(
        {
            "spearman_shat_strue_test": float(rho),
            "top_decile_rmse": {"lam0": top_decile_lam0, "lamP": top_decile_lamP},
            "decile_curve": decile_curve,
            # Recorded vendor-neutral (relative to $OMNIBIAS_SCRATCH, never the
            # resolved absolute path -- see AGENTS.md "Do" / test_no_leakage.py).
            "grid_cache_path": "$OMNIBIAS_SCRATCH/tabular_uncertainty/saw_wave_grid.npz",
            "diagnostic": True,
            "note": (
                "Diagnostic panel + decile curve only -- never a gate (theory 05-03 "
                "section 10). Carries no 'passed' key and is not read by any "
                "gates['all_passed']. Gate G1 is a separate, full multi-seed "
                "lam-selection sweep in benchmarks/tabular_uncertainty.py (Phase 2)."
            ),
            "wall_seconds": round(time.perf_counter() - t0, 3),
        }
    )
    out = write_json("saw_wave_figure5_smoke.json", payload)
    print(
        f"wrote {out}  wrote {grid_path}  "
        f"top_decile_rmse(lam=0)={top_decile_lam0:.4f}  top_decile_rmse(lam={args.lam})={top_decile_lamP:.4f}  "
        f"spearman(s_hat,s_true)={rho:.3f}"
    )


if __name__ == "__main__":
    main()
