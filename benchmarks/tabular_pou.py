# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Theory 05-04: TabPOU axis-aligned POU boosting (G0-G6).

Smoke (default) is a wiring gate: only G0 and G5 fail the process. ``--full``
is the scientific experiment (5 seeds, public suite). RealMLP / TabM stay
``--full``-only. Temperature collapse only; no founding ``delta -> 0``.
G0b is a matched-budget report on current ``fit_boosted``, not a TabPOU
pass/fail. G6 never silently replaces the flagship LightGBM table in
``docs/benchmarks.md``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from _common import provenance, write_json  # type: ignore[import-not-found]  # noqa: E402
from _gates import gates_block  # type: ignore[import-not-found]  # noqa: E402

SCRATCH = Path(os.environ.get("OMNIBIAS_SCRATCH", "artifacts"))

SEEDS_FULL: tuple[int, ...] = (0, 1, 2, 3, 4)
SEEDS_SMOKE: tuple[int, ...] = (0,)
G1_MARGIN = 0.05
G3_CLASS = ("breast_cancer", "banknote", "ionosphere")
G3_REG = ("kin8nm", "wine_quality", "energy_efficiency")
G6_SUITE = ("breast_cancer", "wine", "digits", "diabetes")

BUDGET: dict[str, dict[str, int]] = {
    "smoke": {
        "n_g1": 240, "n_stages": 8, "depth": 2, "n_quantiles": 8, "n_bins": 4,
        "inner_steps": 8, "lgbm_estimators": 40, "cat_iterations": 40,
        "g4_max_rows": 400, "residual_steps": 12, "residual_k": 4,
        "g0b_n": 80, "g2_n": 200, "g2_stages": 6,
    },
    "full": {
        "n_g1": 2000, "n_stages": 60, "depth": 3, "n_quantiles": 16, "n_bins": 16,
        "inner_steps": 40, "lgbm_estimators": 200, "cat_iterations": 200,
        "g4_max_rows": 2000, "residual_steps": 40, "residual_k": 8,
        "g0b_n": 200, "g2_n": 800, "g2_stages": 20,
    },
}


def _pin_worker_threads(n_threads: int = 1) -> None:
    import torch

    torch.set_num_threads(max(1, int(n_threads)))


def _parallel_map(fn: Any, units: list[tuple[Any, ...]], *, workers: int) -> list[Any]:
    if workers <= 1 or len(units) <= 1:
        return [fn(*unit) for unit in units]
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor

    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=workers, mp_context=ctx, initializer=_pin_worker_threads, initargs=(1,)
    ) as ex:
        futures = [ex.submit(fn, *unit) for unit in units]
        return [f.result() for f in futures]


def _mean_std(xs: list[float]) -> tuple[float, float]:
    arr = np.asarray(xs, dtype=np.float64)
    if arr.size == 0:
        return float("nan"), float("nan")
    return float(arr.mean()), float(arr.std(ddof=1) if arr.size > 1 else 0.0)


def _not_worse(ours: float, base: float, base_std: float) -> bool:
    return bool(ours >= base - base_std - 1e-12)


def _strict_win(ours: float, base: float, base_std: float) -> bool:
    return bool(ours > base + 1e-12)


def _check_g0() -> dict[str, Any]:
    from omnibias.tab import SoftTreeConfig, init_params
    from omnibias.tab._core.forward import leaf_memberships
    from omnibias.tab._core.leaves import closed_form_leaves, newton_leaf_loss
    from omnibias.tab.torch.boosting import fit_boosted
    from omnibias.tab.torch.model import SoftTreeEnsemble

    rng = np.random.default_rng(0)
    cfg = SoftTreeConfig(
        n_features=4, n_trees=1, depth=2, task="regression", n_outputs=1,
        seed=0, beta_final=6.0, leaf_l2=1e-4,
    )
    X = rng.standard_normal((60, 4))
    residual = rng.standard_normal((60, 1))
    weight = np.full((60, 1), 2.0)
    params = init_params(cfg, rng)
    P = leaf_memberships(params, X, cfg.beta_final)
    leaves_cf = closed_form_leaves(P, residual, weight, leaf_l2=cfg.leaf_l2)
    loss_cf = newton_leaf_loss(P, residual, weight, leaves_cf)

    import torch

    model = SoftTreeEnsemble(cfg, params)
    model.set_beta(cfg.beta_final)
    model.W.requires_grad_(False)
    model.t.requires_grad_(False)
    model.b0.requires_grad_(False)
    Xt = torch.as_tensor(X)
    rt = torch.as_tensor(residual)
    wt = torch.as_tensor(weight)
    opt = torch.optim.Adam([model.leaves], lr=0.05)
    for _ in range(250):
        opt.zero_grad(set_to_none=True)
        loss = (wt * (model(Xt) - rt) ** 2).mean()
        loss.backward()
        opt.step()
    loss_ad = newton_leaf_loss(P, residual, weight, model.to_params().leaves)
    ratio = float(loss_cf / max(loss_ad, 1e-30))

    y = (X[:, 0] > 0).astype(np.float64)
    cfg_b = SoftTreeConfig(n_features=4, n_trees=1, depth=1, task="binary", seed=1, beta_final=4.0)
    pa = fit_boosted(X, y, cfg_b, n_stages=3, inner_steps=5)[0].to_params()
    pb = fit_boosted(X, y, cfg_b, n_stages=3, inner_steps=5, leaf_solver="adam")[0].to_params()
    adam_ident = bool(np.array_equal(pa.W, pb.W) and np.array_equal(pa.leaves, pb.leaves))

    cfg_ax = SoftTreeConfig(n_features=6, n_trees=2, depth=2, split_kind="axis", task="binary", seed=2)
    W = init_params(cfg_ax, 2).W
    axis_ok = all(
        int(np.count_nonzero(np.abs(W[m, j]) > 1e-15)) == 1 for m in range(2) for j in range(2)
    )
    passed = bool(ratio <= 1.01 and adam_ident and axis_ok)
    return {
        "name": "G0",
        "passed": passed,
        "loss_cf_over_adam": ratio,
        "adam_bit_identity": adam_ident,
        "axis_one_hot": axis_ok,
    }


def _g0b_one_arm(Xtr: np.ndarray, ytr: np.ndarray, Xte: np.ndarray, yte: np.ndarray, *, n_stages: int, inner_steps: int, seed: int) -> float:
    from omnibias.tab import SoftTreeConfig
    from omnibias.tab.torch.boosting import fit_boosted

    cfg = SoftTreeConfig(
        n_features=int(Xtr.shape[1]), n_trees=1, depth=2, task="regression",
        n_outputs=1, seed=seed, beta_final=8.0,
    )
    model, _ = fit_boosted(
        Xtr, ytr, cfg, n_stages=n_stages, inner_steps=inner_steps, learning_rate=0.3,
    )
    pred = model.predict(Xte).reshape(-1)
    return float(np.sqrt(np.mean((pred - yte.reshape(-1)) ** 2)))


def _check_g0b(budget: dict[str, int], seeds: tuple[int, ...]) -> dict[str, Any]:
    """Report-only: flagship ``fit_boosted`` vs the 05-03 under-budget. Not a TabPOU gate."""
    rng = np.random.default_rng(0)
    n = int(budget["g0b_n"])
    X = rng.standard_normal((n, 4))
    y = np.sin(X[:, 0]) + 0.4 * X[:, 1]
    n_tr = int(0.75 * n)
    under: list[float] = []
    flag: list[float] = []
    for seed in seeds:
        rng_s = np.random.default_rng(int(seed))
        idx = rng_s.permutation(n)
        tr, te = idx[:n_tr], idx[n_tr:]
        under.append(_g0b_one_arm(X[tr], y[tr], X[te], y[te], n_stages=10, inner_steps=int(budget["inner_steps"]), seed=int(seed)))
        flag.append(_g0b_one_arm(X[tr], y[tr], X[te], y[te], n_stages=60, inner_steps=int(budget["inner_steps"]), seed=int(seed)))
    um, us = _mean_std(under)
    fm, fs = _mean_std(flag)
    return {
        "name": "G0b",
        "passed": True,
        "report_only": True,
        "under_budget_rmse_mean": um,
        "under_budget_rmse_std": us,
        "flagship_rmse_mean": fm,
        "flagship_rmse_std": fs,
        "n_stages_under": 10,
        "n_stages_flagship": 60,
        "inner_steps": int(budget["inner_steps"]),
        "note": "Current fit_boosted (adam, oblique) at TabConfig vs 05-03 n_stages=10; not a TabPOU pass/fail.",
    }


def _check_g5() -> dict[str, Any]:
    from omnibias.tab import certify_tab_gap, make_axis_rule
    from omnibias.tab.pou import TabPOUConfig, fit_tabpou

    X, y, _ = make_axis_rule(n_samples=180, n_features=8, seed=0)
    cfg1 = TabPOUConfig(
        n_features=8, task="binary", depth=1, n_stages=6, n_quantiles=8,
        use_embed=False, robust_scale=False, onehot_max_card=1, seed=0, patience=None,
    )
    model1, _ = fit_tabpou(X, y, cfg1)
    cert1 = certify_tab_gap(model1.to_params(), model1.transform(X[:80]))
    cfg2 = TabPOUConfig(
        n_features=8, task="binary", depth=2, n_stages=4, n_quantiles=8,
        use_embed=False, robust_scale=False, onehot_max_card=1, seed=1, patience=None,
    )
    model2, _ = fit_tabpou(X, y, cfg2)
    cert2 = certify_tab_gap(model2.to_params(), model2.transform(X[:80]))
    passed = bool(cert1.is_sound and cert2.is_sound)
    return {
        "name": "G5",
        "passed": passed,
        "depth1_is_sound": bool(cert1.is_sound),
        "depth2_is_sound": bool(cert2.is_sound),
        "depth1_max_gap": float(cert1.max_gap),
        "depth1_measured_max": float(cert1.measured_max),
        "depth2_max_gap": float(cert2.max_gap),
        "depth2_measured_max": float(cert2.measured_max),
    }


def _g1_one_seed(seed: int, n: int, n_stages: int, depth: int, n_quantiles: int) -> dict[str, Any]:
    from omnibias.tab import SoftTreeConfig, make_axis_rule
    from omnibias.tab.pou import TabPOUConfig, fit_tabpou
    from omnibias.tab.torch.boosting import fit_boosted

    X, y, _ = make_axis_rule(n_samples=n, n_features=8, seed=seed)
    n_tr = int(0.75 * n)
    Xtr, Xte, ytr, yte = X[:n_tr], X[n_tr:], y[:n_tr], y[n_tr:]
    cfg = TabPOUConfig(
        n_features=8, task="binary", depth=depth, n_stages=n_stages,
        n_quantiles=n_quantiles, use_embed=False, robust_scale=False,
        onehot_max_card=1, seed=seed, patience=None, learning_rate=0.4,
    )
    axis_model, _ = fit_tabpou(Xtr, ytr, cfg)
    acc_axis = float(np.mean(axis_model.predict(Xte) == yte))
    obl = SoftTreeConfig(
        n_features=8, n_trees=1, depth=depth, split_kind="oblique",
        task="binary", seed=seed, beta_final=8.0,
    )
    obl_model, _ = fit_boosted(Xtr, ytr, obl, n_stages=n_stages, inner_steps=8, learning_rate=0.4)
    acc_obl = float(np.mean(obl_model.predict(Xte) == yte))
    return {
        "seed": seed, "acc_axis": acc_axis, "acc_oblique": acc_obl,
        "margin": acc_axis - acc_obl,
    }


def _g2_synth_one(seed: int, n: int, n_stages: int, use_embed: bool, n_bins: int) -> dict[str, Any]:
    from omnibias.tab.bench import mlp_heteroscedastic_20d, train_test_split
    from omnibias.tab.pou import TabPOUConfig, fit_tabpou

    ds = mlp_heteroscedastic_20d(n=n, seed=seed)
    Xtr, Xte, ytr, yte = train_test_split(ds, seed=seed)
    cfg = TabPOUConfig(
        n_features=Xtr.shape[1], task="regression", depth=2, n_stages=n_stages,
        n_quantiles=8, use_embed=use_embed, concat_raw=True, n_bins=n_bins,
        robust_scale=True, onehot_max_card=1, seed=seed, patience=None,
    )
    model, _ = fit_tabpou(Xtr, ytr, cfg)
    pred = model.predict(Xte).reshape(-1)
    rmse = float(np.sqrt(np.mean((pred - yte.reshape(-1)) ** 2)))
    mean_pred = float(np.mean(ytr))
    rmse_mean = float(np.sqrt(np.mean((mean_pred - yte.reshape(-1)) ** 2)))
    skill = 1.0 - (rmse ** 2) / max(rmse_mean ** 2, 1e-30)
    return {"seed": seed, "use_embed": use_embed, "rmse": rmse, "skill": skill, "set": "mlp_heteroscedastic_20d"}


def _g2_public_one(seed: int, n_stages: int, use_embed: bool, n_bins: int) -> dict[str, Any]:
    from omnibias.tab.bench import load_dataset, train_test_split
    from omnibias.tab.pou import TabPOUConfig, fit_tabpou

    ds = load_dataset("diabetes", seed=seed)
    Xtr, Xte, ytr, yte = train_test_split(ds, seed=seed)
    cfg = TabPOUConfig(
        n_features=Xtr.shape[1], task="regression", depth=2, n_stages=n_stages,
        n_quantiles=8, use_embed=use_embed, concat_raw=True, n_bins=n_bins,
        robust_scale=True, onehot_max_card=1, seed=seed, patience=None,
    )
    model, _ = fit_tabpou(Xtr, ytr, cfg)
    pred = model.predict(Xte).reshape(-1)
    rmse = float(np.sqrt(np.mean((pred - yte.reshape(-1)) ** 2)))
    mean_pred = float(np.mean(ytr))
    rmse_mean = float(np.sqrt(np.mean((mean_pred - yte.reshape(-1)) ** 2)))
    skill = 1.0 - (rmse ** 2) / max(rmse_mean ** 2, 1e-30)
    return {"seed": seed, "use_embed": use_embed, "rmse": rmse, "skill": skill, "set": "diabetes"}


def _fit_tabpou_split(
    Xtr: np.ndarray, ytr: np.ndarray, Xte: np.ndarray, yte: np.ndarray,
    *, task: str, n_outputs: int, budget: dict[str, int], seed: int,
    use_residual: bool,
) -> dict[str, float]:
    from omnibias.tab.bench import score_predictions
    from omnibias.tab.pou import TabPOUConfig, fit_tabpou

    cfg = TabPOUConfig(
        n_features=int(Xtr.shape[1]), task=task, n_outputs=n_outputs,
        depth=int(budget["depth"]), n_stages=int(budget["n_stages"]),
        n_quantiles=int(budget["n_quantiles"]), n_bins=int(budget["n_bins"]),
        use_embed=True, concat_raw=True, robust_scale=True, seed=seed,
        patience=8, use_residual=use_residual,
        residual_k=int(budget["residual_k"]), residual_steps=int(budget["residual_steps"]),
    )
    model, _ = fit_tabpou(Xtr, ytr, cfg)
    pred = model.predict(Xte)
    prob = model.predict_proba(Xte) if task != "regression" else None
    return score_predictions(yte, pred, prob, task)


def _g3_dataset(name: str, seeds: tuple[int, ...], budget: dict[str, int], max_rows: int) -> dict[str, Any]:
    from omnibias.tab.bench import (
        fit_predict_catboost,
        fit_predict_lightgbm,
        load_dataset,
        score_predictions,
        train_test_split,
    )

    try:
        ds = load_dataset(name, max_rows=max_rows, seed=0)
    except (RuntimeError, ValueError) as exc:
        return {"name": name, "skipped": True, "reason": str(exc)}
    ours: list[float] = []
    lgbm: list[float] = []
    cat: list[float] = []
    cat_ok = True
    cat_reason = ""
    for seed in seeds:
        Xtr, Xte, ytr, yte = train_test_split(ds, seed=seed)
        s_ours = _fit_tabpou_split(
            Xtr, ytr, Xte, yte, task=ds.task, n_outputs=ds.n_outputs,
            budget=budget, seed=seed, use_residual=False,
        )
        ours.append(float(s_ours["primary"]))
        p_lgbm, pr = fit_predict_lightgbm(
            Xtr, ytr, Xte, task=ds.task, n_outputs=ds.n_outputs,
            n_estimators=int(budget["lgbm_estimators"]), seed=seed,
        )
        lgbm.append(float(score_predictions(yte, p_lgbm, pr, ds.task)["primary"]))
        try:
            p_cat, prc = fit_predict_catboost(
                Xtr, ytr, Xte, task=ds.task, n_outputs=ds.n_outputs,
                iterations=int(budget["cat_iterations"]), seed=seed,
            )
            cat.append(float(score_predictions(yte, p_cat, prc, ds.task)["primary"]))
        except Exception as exc:  # noqa: BLE001 -- optional extra
            cat_ok = False
            cat_reason = str(exc)
            break
    if not cat_ok:
        return {"name": name, "skipped": True, "reason": f"catboost: {cat_reason}"}
    om, os_ = _mean_std(ours)
    lm, ls = _mean_std(lgbm)
    cm, cs = _mean_std(cat)
    return {
        "name": name, "skipped": False, "task": ds.task,
        "tabpou_mean": om, "tabpou_std": os_,
        "lgbm_mean": lm, "lgbm_std": ls,
        "cat_mean": cm, "cat_std": cs,
        "not_worse_lgbm": _not_worse(om, lm, ls),
        "not_worse_cat": _not_worse(om, cm, cs),
        "win_lgbm": _strict_win(om, lm, ls),
        "win_cat": _strict_win(om, cm, cs),
    }


def _g3_dataset_unit(name: str, seeds: tuple[int, ...], budget: dict[str, int], max_rows: int) -> dict[str, Any]:
    return _g3_dataset(name, seeds, budget, max_rows)


def _g4_dataset(name: str, seeds: tuple[int, ...], budget: dict[str, int]) -> dict[str, Any]:
    from omnibias.tab.bench import (
        fit_predict_realmlp,
        fit_predict_tabm,
        load_dataset,
        score_predictions,
        train_test_split,
    )

    try:
        ds = load_dataset(name, max_rows=int(budget["g4_max_rows"]), seed=0)
    except (RuntimeError, ValueError) as exc:
        return {"name": name, "skipped": True, "reason": str(exc)}
    if ds.task != "regression":
        return {
            "name": name, "skipped": True,
            "reason": "RealMLP/TabM harness is regression-only (theory 05-03)",
        }
    ours: list[float] = []
    real: list[float] = []
    tabm: list[float] = []
    for seed in seeds:
        Xtr, Xte, ytr, yte = train_test_split(ds, seed=seed)
        s_ours = _fit_tabpou_split(
            Xtr, ytr, Xte, yte, task=ds.task, n_outputs=ds.n_outputs,
            budget=budget, seed=seed, use_residual=True,
        )
        ours.append(float(s_ours["primary"]))
        try:
            p_r, _ = fit_predict_realmlp(Xtr, ytr, Xte, n_epochs=30, seed=seed)
            real.append(float(score_predictions(yte, p_r, None, ds.task)["primary"]))
            p_t, _ = fit_predict_tabm(Xtr, ytr, Xte, n_epochs=30, seed=seed)
            tabm.append(float(score_predictions(yte, p_t, None, ds.task)["primary"]))
        except Exception as exc:  # noqa: BLE001
            return {"name": name, "skipped": True, "reason": str(exc)}
    om, os_ = _mean_std(ours)
    rm, rs = _mean_std(real)
    tm, ts = _mean_std(tabm)
    return {
        "name": name, "skipped": False,
        "tabpou_mean": om, "tabpou_std": os_,
        "realmlp_mean": rm, "realmlp_std": rs,
        "tabm_mean": tm, "tabm_std": ts,
        "not_worse_realmlp": _not_worse(om, rm, rs),
        "not_worse_tabm": _not_worse(om, tm, ts),
        "vs_realmlp": "win" if _strict_win(om, rm, rs) else ("tie" if _not_worse(om, rm, rs) else "loss"),
        "vs_tabm": "win" if _strict_win(om, tm, ts) else ("tie" if _not_worse(om, tm, ts) else "loss"),
    }


def _g6_dataset(name: str, seeds: tuple[int, ...], budget: dict[str, int]) -> dict[str, Any]:
    from omnibias.tab.bench import (
        fit_predict_lightgbm,
        load_dataset,
        score_predictions,
        train_test_split,
    )

    ds = load_dataset(name, seed=0)
    ours: list[float] = []
    lgbm: list[float] = []
    for seed in seeds:
        Xtr, Xte, ytr, yte = train_test_split(ds, seed=seed)
        s_ours = _fit_tabpou_split(
            Xtr, ytr, Xte, yte, task=ds.task, n_outputs=ds.n_outputs,
            budget=budget, seed=seed, use_residual=False,
        )
        ours.append(float(s_ours["primary"]))
        p_lgbm, pr = fit_predict_lightgbm(
            Xtr, ytr, Xte, task=ds.task, n_outputs=ds.n_outputs,
            n_estimators=int(budget["lgbm_estimators"]), seed=seed,
        )
        lgbm.append(float(score_predictions(yte, p_lgbm, pr, ds.task)["primary"]))
    om, os_ = _mean_std(ours)
    lm, ls = _mean_std(lgbm)
    return {
        "name": name, "tabpou_mean": om, "lgbm_mean": lm,
        "not_worse": _not_worse(om, lm, ls),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="TabPOU 05-04 gates")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--skip-g4", action="store_true",
        help="Skip RealMLP/TabM (pytabkit) even on --full; leftover-record G4.",
    )
    args = parser.parse_args()
    tier = "full" if args.full else "smoke"
    budget = BUDGET[tier]
    seeds = SEEDS_FULL if args.full else SEEDS_SMOKE
    workers = max(1, int(args.workers))
    t0 = time.perf_counter()

    g0 = _check_g0()
    g5 = _check_g5()
    g0b: dict[str, Any]
    if args.full:
        g0b = _check_g0b(budget, seeds[:1])
    else:
        g0b = {"name": "G0b", "passed": True, "report_only": True, "skipped": True, "wiring_only": True}

    g1_units = _parallel_map(
        _g1_one_seed,
        [(s, budget["n_g1"], 8 if not args.full else int(budget["n_stages"]), 2, int(budget["n_quantiles"])) for s in seeds],
        workers=workers,
    )
    g1_margins = [u["margin"] for u in g1_units]
    g1_worst = float(min(g1_margins)) if g1_margins else float("nan")
    g1 = {
        "name": "G1",
        "passed": bool(g1_worst >= G1_MARGIN) if args.full else True,
        "wiring_only": not args.full,
        "worst_margin": g1_worst,
        "units": g1_units,
    }

    g2_raw = _parallel_map(
        _g2_synth_one,
        [(s, int(budget["g2_n"]), int(budget["g2_stages"]), False, int(budget["n_bins"])) for s in seeds],
        workers=workers,
    )
    g2_band = _parallel_map(
        _g2_synth_one,
        [(s, int(budget["g2_n"]), int(budget["g2_stages"]), True, int(budget["n_bins"])) for s in seeds],
        workers=workers,
    )
    g2_pub_raw = _parallel_map(
        _g2_public_one,
        [(s, int(budget["g2_stages"]), False, int(budget["n_bins"])) for s in seeds],
        workers=workers,
    )
    g2_pub_band = _parallel_map(
        _g2_public_one,
        [(s, int(budget["g2_stages"]), True, int(budget["n_bins"])) for s in seeds],
        workers=workers,
    )
    raw_rmse = float(max(u["rmse"] for u in g2_raw))
    band_rmse = float(max(u["rmse"] for u in g2_band))
    pub_raw = float(max(u["rmse"] for u in g2_pub_raw))
    pub_band = float(max(u["rmse"] for u in g2_pub_band))
    skill_ok = all(u["skill"] > 0.0 for u in g2_raw + g2_band + g2_pub_raw + g2_pub_band)
    band_ok = bool(band_rmse <= raw_rmse + 1e-12 and pub_band <= pub_raw + 1e-12)
    g2 = {
        "name": "G2",
        "passed": bool(skill_ok and band_ok) if args.full else True,
        "wiring_only": not args.full,
        "skill_ok": skill_ok,
        "worst_rmse_raw_synth": raw_rmse,
        "worst_rmse_band_synth": band_rmse,
        "worst_rmse_raw_diabetes": pub_raw,
        "worst_rmse_band_diabetes": pub_band,
    }

    g3_names = list(G3_CLASS[:1]) if not args.full else list(G3_CLASS + G3_REG)
    g3_rows = _parallel_map(
        _g3_dataset_unit,
        [(n, seeds, budget, 800 if not args.full else 3000) for n in g3_names],
        workers=workers,
    )
    scored = [r for r in g3_rows if not r.get("skipped")]
    n_scored = len(scored)
    nw = sum(1 for r in scored if r.get("not_worse_lgbm") and r.get("not_worse_cat"))
    wins = sum(1 for r in scored if r.get("win_lgbm") or r.get("win_cat"))
    g3_pass = bool(n_scored >= 5 and nw >= 5 and wins >= 3)
    g3 = {
        "name": "G3",
        "passed": g3_pass if args.full else True,
        "wiring_only": not args.full,
        "n_scored": n_scored,
        "not_worse_both": nw,
        "strict_wins": wins,
        "rows": g3_rows,
    }

    g4: dict[str, Any]
    if args.full and not args.skip_g4:
        g4_rows = _parallel_map(
            _g4_dataset,
            [(n, seeds, budget) for n in list(G3_CLASS + G3_REG)],
            workers=workers,
        )
        scored4 = [r for r in g4_rows if not r.get("skipped")]
        nw_r = sum(1 for r in scored4 if r.get("not_worse_realmlp"))
        nw_t = sum(1 for r in scored4 if r.get("not_worse_tabm"))
        leftover = None
        if len(scored4) < 4:
            leftover = (
                "G4 bar is not-worse on >=4/6 vs named RealMLP/TabM; the harness is "
                "regression-only so classification rows are incomparable (05-03)."
            )
        g4 = {
            "name": "G4",
            "passed": bool(len(scored4) >= 4 and nw_r >= 4 and nw_t >= 4),
            "not_worse_realmlp": nw_r,
            "not_worse_tabm": nw_t,
            "n_comparable": len(scored4),
            "leftover_recorded": leftover,
            "rows": g4_rows,
        }
    else:
        g4 = {
            "name": "G4",
            "passed": True,
            "wiring_only": not args.full,
            "skipped": True,
            "leftover_recorded": "G4 skipped on smoke (no pytabkit in CI)" if not args.full else "G4 skipped (--skip-g4)",
        }

    g6: dict[str, Any]
    if args.full:
        g6_rows = _parallel_map(
            _g6_dataset,
            [(n, seeds[:1], budget) for n in G6_SUITE],
            workers=workers,
        )
        nw6 = sum(1 for r in g6_rows if r.get("not_worse"))
        g6 = {
            "name": "G6",
            "passed": True,
            "flagship_table_replaced": False,
            "optional_not_worse_4": nw6,
            "rows": g6_rows,
            "note": "Flagship LightGBM table in docs/benchmarks.md is not replaced; TabPOU is reported separately.",
        }
    else:
        g6 = {
            "name": "G6",
            "passed": True,
            "flagship_table_replaced": False,
            "note": "Flagship LightGBM table in docs/benchmarks.md is not replaced; TabPOU is reported separately.",
        }

    entries = [g0, g0b, g1, g2, g3, g4, g5, g6]
    smoke_ok = bool(g0["passed"] and g5["passed"])
    full_ok = bool(g0["passed"] and g1["passed"] and g3["passed"] and g5["passed"])
    payload = {
        **provenance(
            schema="omnibias.tab.pou.v1",
            config={"tier": tier, "budget": budget, "seeds": list(seeds), "workers": workers},
        ),
        "tier": tier,
        "elapsed_seconds": round(time.perf_counter() - t0, 3),
        "g0": g0, "g0b": g0b, "g1": g1, "g2": g2, "g3": g3, "g4": g4, "g5": g5, "g6": g6,
        "gates": gates_block(entries),
        "smoke_process_passed": smoke_ok,
        "licensed_sentence": (
            "TabPOU is a from-scratch axis-aligned POU booster; G0/G5 are wiring+certificate; "
            "G1-G4 are hypothesis gates earned only on --full; G0b is a budget-control report; "
            "G6 does not replace the flagship LightGBM table."
        ),
    }
    if args.full:
        SCRATCH.mkdir(parents=True, exist_ok=True)
        out = SCRATCH / "tabular_pou_full.json"
        out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        write_json("tabular_pou.json", payload)
        print(f"wrote {out}")
    else:
        path = write_json("tabular_pou_smoke.json", payload)
        print(f"wrote {path}")
    ok = full_ok if args.full else smoke_ok
    print(json.dumps({"ok": ok, "tier": tier, "g0": g0["passed"], "g5": g5["passed"]}, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
