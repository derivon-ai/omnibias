# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Theory 05-05: TabPOU joint worlds (H0-H7, then G-tree / G-net / G-hybrid / G5).

Smoke (default) is a wiring gate: H0 identity + tiny H1 + G5. ``--full`` is
Phase A (3-seed probes, val-only accept/reject). ``--lock`` is Phase B
(5-seed family bars). Temperature collapse only; 05-04 G3 stays report-only.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from _common import provenance, write_json  # type: ignore[import-not-found]  # noqa: E402
from _gates import gates_block  # type: ignore[import-not-found]  # noqa: E402

SCRATCH = Path(os.environ.get("OMNIBIAS_SCRATCH", "artifacts"))

SEEDS_FULL: tuple[int, ...] = (0, 1, 2)
SEEDS_LOCK: tuple[int, ...] = (0, 1, 2, 3, 4)
SEEDS_SMOKE: tuple[int, ...] = (0,)
TREE_PUBLIC = ("breast_cancer", "banknote", "ionosphere")
NET_PUBLIC = ("kin8nm", "energy_efficiency")
HYBRID_PUBLIC = TREE_PUBLIC + ("kin8nm", "energy_efficiency", "wine_quality")
ALL_HYP = ("H0", "H1", "H2", "H3", "H4", "H5", "H6", "H7")

BUDGET: dict[str, dict[str, int]] = {
    "smoke": {
        "n_stages": 4, "depth": 1, "n_quantiles": 6, "n_bins": 4,
        "inner_steps": 8, "residual_steps": 6, "residual_k": 2,
        "joint_steps": 6, "polish_sweeps": 2, "g1_n": 160, "g2_n": 80,
        "lgbm_estimators": 40, "cat_iterations": 40, "public_max_rows": 400,
    },
    "full": {
        "n_stages": 20, "depth": 2, "n_quantiles": 8, "n_bins": 8,
        "inner_steps": 20, "residual_steps": 20, "residual_k": 4,
        "joint_steps": 20, "polish_sweeps": 3, "g1_n": 800, "g2_n": 400,
        "lgbm_estimators": 100, "cat_iterations": 100, "public_max_rows": 2000,
    },
    "lock": {
        "n_stages": 60, "depth": 3, "n_quantiles": 16, "n_bins": 16,
        "inner_steps": 40, "residual_steps": 40, "residual_k": 8,
        "joint_steps": 40, "polish_sweeps": 4, "g1_n": 2000, "g2_n": 800,
        "lgbm_estimators": 200, "cat_iterations": 200, "public_max_rows": 3000,
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


def _status(accepted: bool, rejected: bool) -> str:
    if accepted:
        return "accepted"
    if rejected:
        return "rejected"
    return "inconclusive"


def _wiring(name: str, **extra: Any) -> dict[str, Any]:
    out: dict[str, Any] = {
        "name": name, "passed": True, "status": "wiring_only", "wiring_only": True,
    }
    out.update(extra)
    return out


def _skipped(name: str, reason: str) -> dict[str, Any]:
    return {
        "name": name, "passed": True, "status": "inconclusive",
        "skipped": True, "reason": reason, "wiring_only": False,
    }


def _base_cfg(
    n_features: int,
    task: str,
    budget: dict[str, int],
    seed: int,
    *,
    use_embed: bool,
    use_residual: bool,
    n_stages: int | None = None,
    colsample: float = 1.0,
    embed_role: str = "band",
) -> Any:
    from omnibias.tab.pou import TabPOUConfig

    return TabPOUConfig(
        n_features=n_features, task=task, n_outputs=1,
        depth=int(budget["depth"]), n_stages=int(n_stages or budget["n_stages"]),
        n_quantiles=int(budget["n_quantiles"]), n_bins=int(budget["n_bins"]),
        use_embed=use_embed, concat_raw=True, robust_scale=True,
        onehot_max_card=1, seed=seed, patience=6, colsample=float(colsample),
        embed_role=embed_role,
        use_residual=use_residual, residual_k=int(budget["residual_k"]),
        residual_steps=int(budget["residual_steps"]), residual_hidden=32,
    )


def _score_model(model: Any, Xte: np.ndarray, yte: np.ndarray, task: str) -> float:
    from omnibias.tab.bench import score_predictions

    pred = model.predict(Xte)
    prob = model.predict_proba(Xte) if task != "regression" else None
    return float(score_predictions(yte, pred, prob, task)["primary"])


def _rmse(model: Any, X: np.ndarray, y: np.ndarray) -> float:
    pred = model.predict(X).reshape(-1)
    return float(np.sqrt(np.mean((pred - np.asarray(y).reshape(-1)) ** 2)))


def _array_split(
    X: np.ndarray, y: np.ndarray, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(int(seed) + 17)
    n = int(X.shape[0])
    perm = rng.permutation(n)
    n_va = max(1, int(round(0.25 * n)))
    va, tr = perm[:n_va], perm[n_va:]
    return X[tr], X[va], y[tr], y[va]


def _probe_val(
    name: str, seed: int, max_rows: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    from omnibias.tab.bench import load_dataset, train_val_test_split

    ds = load_dataset(name, max_rows=max_rows, seed=0)
    parts = train_val_test_split(ds, seed=int(seed))
    return parts["Xtr"], parts["Xva"], parts["ytr"], parts["yva"], ds.task


def _check_g5() -> dict[str, Any]:
    from omnibias.tab import certify_tab_gap, make_axis_rule
    from omnibias.tab.pou import TabPOUConfig
    from omnibias.tab.pou.joint import TabPOUJointConfig, fit_tabpou_joint

    X, y, _ = make_axis_rule(n_samples=140, n_features=8, seed=0)
    base = TabPOUConfig(
        n_features=8, task="binary", depth=1, n_stages=4, n_quantiles=6,
        use_embed=False, robust_scale=False, onehot_max_card=1, seed=0, patience=None,
    )
    model, _ = fit_tabpou_joint(
        X, y, TabPOUJointConfig(base=base, polish_thresholds=True, polish_sweeps=2)
    )
    cert = certify_tab_gap(model.to_params(), model.transform(X[:70]))
    return {
        "name": "G5",
        "passed": bool(cert.is_sound),
        "is_sound": bool(cert.is_sound),
        "max_gap": float(cert.max_gap),
        "measured_max": float(cert.measured_max),
    }


def _h0_identity() -> dict[str, Any]:
    from omnibias.tab import SoftTreeConfig, init_params
    from omnibias.tab._core.forward import leaf_memberships
    from omnibias.tab._core.leaves import closed_form_leaves, newton_leaf_loss
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
    passed = bool(ratio <= 1.01)
    return {
        "name": "H0",
        "passed": passed,
        "status": "wiring_only",
        "wiring_only": True,
        "loss_cf_over_adam": ratio,
        "identity_ok": passed,
    }


def _h0(budget: dict[str, int], seeds: tuple[int, ...], full: bool) -> dict[str, Any]:
    if not full:
        return _h0_identity()
    from omnibias.tab import SoftTreeConfig
    from omnibias.tab.pou import fit_tabpou
    from omnibias.tab.torch.boosting import fit_boosted

    names = ("breast_cancer", "kin8nm")
    cf_s: list[float] = []
    ad_s: list[float] = []
    t_cf = 0.0
    t_ad = 0.0
    try:
        for name in names:
            for seed in seeds:
                Xtr, Xva, ytr, yva, task = _probe_val(name, int(seed), int(budget["public_max_rows"]))
                cfg = _base_cfg(
                    int(Xtr.shape[1]), task, budget, int(seed),
                    use_embed=False, use_residual=False,
                )
                cfg = replace(cfg, patience=None)
                t0 = time.perf_counter()
                m_cf, _ = fit_tabpou(Xtr, ytr, cfg)
                t_cf += time.perf_counter() - t0
                cf_s.append(_score_model(m_cf, Xva, yva, task))
                obl = SoftTreeConfig(
                    n_features=int(Xtr.shape[1]), n_trees=1, depth=int(budget["depth"]),
                    split_kind="axis", task=task, seed=int(seed), beta_final=8.0,
                )
                t0 = time.perf_counter()
                m_ad, _ = fit_boosted(
                    Xtr, ytr, obl, n_stages=int(budget["n_stages"]),
                    inner_steps=int(budget["inner_steps"]), learning_rate=0.3,
                    leaf_solver="adam",
                )
                t_ad += time.perf_counter() - t0
                ad_s.append(_score_model(m_ad, Xva, yva, task))
    except (RuntimeError, ValueError) as exc:
        return _skipped("H0", str(exc))
    cm, cs = _mean_std(cf_s)
    am, as_ = _mean_std(ad_s)
    nw = _not_worse(cm, am, as_)
    faster = t_cf <= t_ad + 1e-9
    within = abs(cm - am) <= max(as_, 1e-6) + 0.01
    accepted = bool(nw and faster)
    rejected = bool((not accepted) and within)
    ident = _h0_identity()
    return {
        "name": "H0",
        "passed": bool(ident["passed"]),
        "status": _status(accepted, rejected),
        "cf_mean": cm, "adam_mean": am, "seconds_cf": round(t_cf, 3),
        "seconds_adam": round(t_ad, 3), "not_worse": nw, "faster": faster,
        "loss_cf_over_adam": ident["loss_cf_over_adam"],
        "wiring_only": False,
    }


def _h1(budget: dict[str, int], seeds: tuple[int, ...], full: bool) -> dict[str, Any]:
    from omnibias.tab import make_axis_rule
    from omnibias.tab.bench import mlp_heteroscedastic_20d
    from omnibias.tab.pou import fit_tabpou

    tree_drop: list[float] = []
    net_ratio: list[float] = []
    for seed in seeds:
        if full:
            try:
                Xtr, Xva, ytr, yva, task = _probe_val(
                    "breast_cancer", int(seed), int(budget["public_max_rows"])
                )
            except (RuntimeError, ValueError) as exc:
                return _skipped("H1", str(exc))
        else:
            X, y, _ = make_axis_rule(n_samples=int(budget["g1_n"]), n_features=8, seed=int(seed))
            Xtr, Xva, ytr, yva = _array_split(X, y, int(seed))
            task = "binary"
        cfg0 = _base_cfg(int(Xtr.shape[1]), task, budget, int(seed), use_embed=False, use_residual=False)
        m0, _ = fit_tabpou(Xtr, ytr, cfg0)
        mr, _ = fit_tabpou(Xtr, ytr, replace(cfg0, use_residual=True))
        tree_drop.append(_score_model(m0, Xva, yva, task) - _score_model(mr, Xva, yva, task))
        if full:
            try:
                Xtr2, Xva2, ytr2, yva2, task2 = _probe_val(
                    "kin8nm", int(seed), int(budget["public_max_rows"])
                )
            except (RuntimeError, ValueError) as exc:
                return _skipped("H1", str(exc))
        else:
            ds = mlp_heteroscedastic_20d(n=int(budget["g2_n"]), seed=int(seed))
            Xtr2, Xva2, ytr2, yva2 = _array_split(ds.X, ds.y, int(seed))
            task2 = "regression"
        c2 = _base_cfg(int(Xtr2.shape[1]), task2, budget, int(seed), use_embed=False, use_residual=False)
        m0n, _ = fit_tabpou(Xtr2, ytr2, c2)
        mrn, _ = fit_tabpou(Xtr2, ytr2, replace(c2, use_residual=True))
        r0 = _rmse(m0n, Xva2, yva2)
        rr = _rmse(mrn, Xva2, yva2)
        net_ratio.append(rr / max(r0, 1e-12))
    drop_m, _ = _mean_std(tree_drop)
    ratio_m, _ = _mean_std(net_ratio)
    accepted = bool(ratio_m <= 0.95 and drop_m <= 0.01) if full else True
    rejected = bool(full and ratio_m > 0.99)
    return {
        "name": "H1",
        "passed": True,
        "status": _status(accepted, rejected) if full else "wiring_only",
        "tree_acc_drop_mean": drop_m, "kin8nm_rmse_ratio": ratio_m,
        "wiring_only": not full,
    }


def _h2(budget: dict[str, int], seeds: tuple[int, ...], full: bool) -> dict[str, Any]:
    from omnibias.tab import make_axis_rule
    from omnibias.tab.pou import fit_tabpou
    from omnibias.tab.pou.joint import TabPOUJointConfig, fit_tabpou_joint

    margins: list[float] = []
    for seed in seeds:
        X, y, _ = make_axis_rule(n_samples=int(budget["g1_n"]), n_features=8, seed=int(seed))
        Xtr, Xva, ytr, yva = _array_split(X, y, int(seed))
        base = _base_cfg(8, "binary", budget, int(seed), use_embed=False, use_residual=False)
        base = replace(base, robust_scale=False, patience=None)
        m0, _ = fit_tabpou(Xtr, ytr, base)
        mj, _ = fit_tabpou_joint(
            Xtr, ytr,
            TabPOUJointConfig(base=base, polish_thresholds=True, polish_sweeps=int(budget["polish_sweeps"])),
        )
        margins.append(_score_model(mj, Xva, yva, "binary") - _score_model(m0, Xva, yva, "binary"))
    worst = float(min(margins)) if margins else float("nan")
    probe_ok = True
    probe_win = False
    rels: list[float] = []
    if full:
        for name in ("breast_cancer", "kin8nm"):
            try:
                for seed in seeds:
                    Xtr, Xva, ytr, yva, task = _probe_val(
                        name, int(seed), int(budget["public_max_rows"])
                    )
                    base = _base_cfg(
                        int(Xtr.shape[1]), task, budget, int(seed),
                        use_embed=False, use_residual=False,
                    )
                    m0, _ = fit_tabpou(Xtr, ytr, base)
                    mj, _ = fit_tabpou_joint(
                        Xtr, ytr,
                        TabPOUJointConfig(
                            base=base, polish_thresholds=True,
                            polish_sweeps=int(budget["polish_sweeps"]),
                        ),
                    )
                    s0 = _score_model(m0, Xva, yva, task)
                    sj = _score_model(mj, Xva, yva, task)
                    rels.append(abs(sj - s0) / max(abs(s0), 1e-12))
                    if sj < s0 - 1e-12:
                        probe_ok = False
                    if sj > s0 + 1e-12:
                        probe_win = True
            except (RuntimeError, ValueError) as exc:
                return _skipped("H2", str(exc))
    axis_win = worst >= 0.02
    accepted = bool(axis_win or (probe_ok and probe_win)) if full else True
    rejected = bool(full and (not accepted) and (not rels or float(max(rels)) < 0.01) and abs(worst) < 0.01)
    return {
        "name": "H2",
        "passed": True,
        "status": _status(accepted, rejected) if full else "wiring_only",
        "worst_margin_vs_v0": worst, "probe_not_worse": probe_ok, "probe_strict_win": probe_win,
        "wiring_only": not full,
    }


def _h3(budget: dict[str, int], seeds: tuple[int, ...], full: bool) -> dict[str, Any]:
    from omnibias.tab.bench import mlp_heteroscedastic_20d
    from omnibias.tab.pou import fit_tabpou
    from omnibias.tab.pou.joint import TabPOUJointConfig, fit_tabpou_joint

    def _fit_rmse(Xtr: np.ndarray, ytr: np.ndarray, Xva: np.ndarray, yva: np.ndarray, seed: int, arm: str) -> float:
        use_embed = arm != "none"
        role = "integral" if arm == "trained_integral" else "band"
        cfg = _base_cfg(
            int(Xtr.shape[1]), "regression", budget, seed,
            use_embed=use_embed, use_residual=False, embed_role=role,
        )
        if arm in ("trained_band", "trained_integral"):
            model, _ = fit_tabpou_joint(
                Xtr, ytr,
                TabPOUJointConfig(base=cfg, train_embed=True, joint_steps=int(budget["joint_steps"])),
            )
        else:
            model, _ = fit_tabpou(Xtr, ytr, cfg, freeze_embed=True)
        return _rmse(model, Xva, yva)

    arms = ("none", "frozen", "trained_band", "trained_integral")
    worst: dict[str, float] = {}
    datasets: list[tuple[str, Any]] = [("mlp_heteroscedastic_20d", None)]
    if full:
        datasets.append(("kin8nm", None))
    for ds_name, _ in datasets:
        per_arm: dict[str, list[float]] = {a: [] for a in arms}
        for seed in seeds:
            if ds_name == "mlp_heteroscedastic_20d":
                ds = mlp_heteroscedastic_20d(n=int(budget["g2_n"]), seed=int(seed))
                Xtr, Xva, ytr, yva = _array_split(ds.X, ds.y, int(seed))
            else:
                try:
                    Xtr, Xva, ytr, yva, _task = _probe_val(
                        "kin8nm", int(seed), int(budget["public_max_rows"])
                    )
                except (RuntimeError, ValueError) as exc:
                    return _skipped("H3", str(exc))
            for arm in arms:
                per_arm[arm].append(_fit_rmse(Xtr, ytr, Xva, yva, int(seed), arm))
        for arm in arms:
            key = f"{ds_name}:{arm}"
            worst[key] = float(max(per_arm[arm]))
    trained_ok = True
    for ds_name, _ in datasets:
        if worst[f"{ds_name}:trained_band"] > worst[f"{ds_name}:none"] + 1e-12:
            trained_ok = False
    frozen_or_none = False
    for ds_name, _ in datasets:
        tb = worst[f"{ds_name}:trained_band"]
        if min(worst[f"{ds_name}:frozen"], worst[f"{ds_name}:none"]) < tb - 1e-12:
            frozen_or_none = True
    winner = "trained_band" if trained_ok else ("frozen" if not trained_ok else "none")
    if frozen_or_none:
        winner = "frozen_or_none"
    accepted = bool(trained_ok) if full else True
    rejected = bool(full and frozen_or_none)
    return {
        "name": "H3",
        "passed": True,
        "status": _status(accepted, rejected) if full else "wiring_only",
        "worst_rmse": worst, "winner": winner, "wiring_only": not full,
    }


def _h4(budget: dict[str, int], seeds: tuple[int, ...], full: bool) -> dict[str, Any]:
    from omnibias.tab.bench import mlp_heteroscedastic_20d
    from omnibias.tab.pou import fit_tabpou

    naive: list[float] = []
    grouped: list[float] = []
    col_s: list[float] = []
    naive_tr: list[float] = []
    grouped_tr: list[float] = []
    for seed in seeds:
        if full:
            try:
                Xtr, Xva, ytr, yva, task = _probe_val(
                    "kin8nm", int(seed), int(budget["public_max_rows"])
                )
            except (RuntimeError, ValueError) as exc:
                return _skipped("H4", str(exc))
        else:
            ds = mlp_heteroscedastic_20d(n=int(budget["g2_n"]), seed=int(seed))
            Xtr, Xva, ytr, yva = _array_split(ds.X, ds.y, int(seed))
            task = "regression"
        cfg = _base_cfg(int(Xtr.shape[1]), task, budget, int(seed), use_embed=True, use_residual=False)
        m0, r0 = fit_tabpou(Xtr, ytr, cfg, grouped_splits=False)
        mg, rg = fit_tabpou(Xtr, ytr, cfg, grouped_splits=True)
        mc, _rc = fit_tabpou(Xtr, ytr, replace(cfg, colsample=0.5), grouped_splits=False)
        naive.append(_rmse(m0, Xva, yva))
        grouped.append(_rmse(mg, Xva, yva))
        col_s.append(_rmse(mc, Xva, yva))
        naive_tr.append(float(r0.train_loss))
        grouped_tr.append(float(rg.train_loss))
    nm, _ = _mean_std(naive)
    gm, _ = _mean_std(grouped)
    cm, _ = _mean_std(col_s)
    improved = bool(gm < nm - 1e-12 or cm < nm - 1e-12 or float(np.mean(grouped_tr)) < float(np.mean(naive_tr)))
    accepted = bool(improved) if full else True
    rejected = bool(full and not improved)
    return {
        "name": "H4",
        "passed": True,
        "status": _status(accepted, rejected) if full else "wiring_only",
        "naive_rmse": nm, "grouped_rmse": gm, "colsample_rmse": cm,
        "wiring_only": not full,
    }


def _h5(budget: dict[str, int], seeds: tuple[int, ...], full: bool) -> dict[str, Any]:
    from omnibias.tab import make_axis_rule, make_oblique_xor
    from omnibias.tab.pou import TabPOUConfig, fit_tabpou

    greedy_m: list[float] = []
    pair_m: list[float] = []
    makers = (make_oblique_xor, make_axis_rule)
    for maker in makers:
        for seed in seeds:
            X, y, _ = maker(n_samples=int(budget["g1_n"]), seed=int(seed))
            Xtr, Xva, ytr, yva = _array_split(X, y, int(seed))
            majority = float(max(yva.mean(), 1.0 - yva.mean()))
            cfg = TabPOUConfig(
                n_features=int(Xtr.shape[1]), task="binary", depth=2,
                n_stages=int(min(8, budget["n_stages"])), n_quantiles=int(budget["n_quantiles"]),
                use_embed=False, robust_scale=False, onehot_max_card=1, seed=int(seed), patience=None,
            )
            mg, _ = fit_tabpou(Xtr, ytr, cfg, pairwise=False)
            mp, _ = fit_tabpou(Xtr, ytr, cfg, pairwise=True)
            greedy_m.append(float(np.mean(mg.predict(Xva) == yva)) - majority)
            pair_m.append(float(np.mean(mp.predict(Xva) == yva)) - majority)
    g_w = float(min(greedy_m)) if greedy_m else float("nan")
    p_w = float(min(pair_m)) if pair_m else float("nan")
    greedy_fails = g_w < 0.05
    pair_recovers = p_w >= 0.05
    accepted = bool(greedy_fails and pair_recovers) if full else True
    rejected = bool(full and not greedy_fails)
    return {
        "name": "H5",
        "passed": True,
        "status": _status(accepted, rejected) if full else "wiring_only",
        "greedy_worst_margin": g_w, "pairwise_worst_margin": p_w,
        "wiring_only": not full,
    }


def _h6(budget: dict[str, int], seeds: tuple[int, ...], full: bool) -> dict[str, Any]:
    from omnibias.tab import make_axis_rule
    from omnibias.tab.bench import mlp_heteroscedastic_20d
    from omnibias.tab.pou import fit_tabpou
    from omnibias.tab.pou.joint import TabPOUJointConfig, fit_tabpou_joint

    tree_ok = True
    net_win = True
    for seed in seeds:
        if full:
            try:
                Xtr, Xva, ytr, yva, task = _probe_val(
                    "breast_cancer", int(seed), int(budget["public_max_rows"])
                )
            except (RuntimeError, ValueError) as exc:
                return _skipped("H6", str(exc))
        else:
            X, y, _ = make_axis_rule(n_samples=int(budget["g1_n"]), n_features=8, seed=int(seed))
            Xtr, Xva, ytr, yva = _array_split(X, y, int(seed))
            task = "binary"
        base = _base_cfg(int(Xtr.shape[1]), task, budget, int(seed), use_embed=False, use_residual=False)
        m_seq, _ = fit_tabpou(Xtr, ytr, replace(base, use_residual=True))
        m_j, _ = fit_tabpou_joint(
            Xtr, ytr,
            TabPOUJointConfig(base=base, joint_residual=True, joint_steps=int(budget["joint_steps"])),
        )
        if _score_model(m_j, Xva, yva, task) < _score_model(m_seq, Xva, yva, task) - 0.01:
            tree_ok = False
        if full:
            try:
                Xtr2, Xva2, ytr2, yva2, task2 = _probe_val(
                    "kin8nm", int(seed), int(budget["public_max_rows"])
                )
            except (RuntimeError, ValueError) as exc:
                return _skipped("H6", str(exc))
        else:
            ds = mlp_heteroscedastic_20d(n=int(budget["g2_n"]), seed=int(seed))
            Xtr2, Xva2, ytr2, yva2 = _array_split(ds.X, ds.y, int(seed))
            task2 = "regression"
        b2 = _base_cfg(int(Xtr2.shape[1]), task2, budget, int(seed), use_embed=False, use_residual=False)
        ms, _ = fit_tabpou(Xtr2, ytr2, replace(b2, use_residual=True))
        mj, _ = fit_tabpou_joint(
            Xtr2, ytr2,
            TabPOUJointConfig(base=b2, joint_residual=True, joint_steps=int(budget["joint_steps"])),
        )
        if _rmse(mj, Xva2, yva2) > _rmse(ms, Xva2, yva2):
            net_win = False
    accepted = bool(tree_ok and net_win) if full else True
    rejected = bool(full and not tree_ok)
    return {
        "name": "H6",
        "passed": True,
        "status": _status(accepted, rejected) if full else "wiring_only",
        "hybrid_holds_on_tree_probe": tree_ok, "joint_beats_seq_on_net": net_win,
        "wiring_only": not full,
    }


def _h7(budget: dict[str, int], seeds: tuple[int, ...], full: bool) -> dict[str, Any]:
    from omnibias.tab import make_axis_rule
    from omnibias.tab.pou import fit_tabpou

    X, y, _ = make_axis_rule(n_samples=int(budget["g1_n"]), n_features=8, seed=int(seeds[0]))
    Xtr, Xva, ytr, yva = _array_split(X, y, int(seeds[0]))
    n_lo = 60 if full else int(budget["n_stages"])
    n_hi = 200 if full else int(budget["n_stages"])
    c_lo = _base_cfg(8, "binary", budget, int(seeds[0]), use_embed=False, use_residual=False, n_stages=n_lo)
    c_lo = replace(c_lo, robust_scale=False, patience=None)
    m_lo, _ = fit_tabpou(Xtr, ytr, c_lo)
    m_cs, _ = fit_tabpou(Xtr, ytr, replace(c_lo, colsample=0.5))
    acc_hi = float("nan")
    if full:
        m_hi, _ = fit_tabpou(Xtr, ytr, replace(c_lo, n_stages=n_hi))
        acc_hi = float(np.mean(m_hi.predict(Xva) == yva))
    return {
        "name": "H7",
        "passed": True,
        "status": "report_only",
        "acc_n_stages_lo": float(np.mean(m_lo.predict(Xva) == yva)),
        "acc_n_stages_hi": acc_hi,
        "acc_colsample_0_5": float(np.mean(m_cs.predict(Xva) == yva)),
        "n_stages_lo": n_lo, "n_stages_hi": n_hi,
        "note": "Do not promote extra-stage boost-only as 05-05 shipped.",
        "wiring_only": not full,
    }


def freeze_from_hypotheses(hs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    r"""Val-only constructive pick. H7 never promotes extra stages."""

    def st(name: str) -> str:
        return str(hs.get(name, {}).get("status", "inconclusive"))

    h3 = st("H3")
    h1 = st("H1")
    h6 = st("H6")
    h2 = st("H2")
    return {
        "polish_thresholds": h2 == "accepted",
        "train_embed": h3 == "accepted",
        "use_embed": h3 == "accepted",
        "grouped_splits": st("H4") == "accepted",
        "joint_residual": h6 == "accepted",
        "use_residual": h1 == "accepted" and h6 != "accepted",
        "pairwise_warmstart": st("H5") == "accepted",
        "binarize_eval": h2 == "accepted",
        "n_stages": 60,
        "note": (
            "H7 is report-only; lock keeps n_stages=60. "
            "H3 reject forbids 17x d (use_embed stays false)."
        ),
    }


def _joint_from_frozen(
    n_features: int, task: str, budget: dict[str, int], seed: int, frozen: dict[str, Any]
) -> Any:
    from omnibias.tab.pou.joint import TabPOUJointConfig

    base = _base_cfg(
        n_features, task, budget, seed,
        use_embed=bool(frozen.get("use_embed", False)),
        use_residual=bool(frozen.get("use_residual", False)),
        n_stages=int(frozen.get("n_stages", budget["n_stages"])),
    )
    return TabPOUJointConfig(
        base=base,
        polish_thresholds=bool(frozen.get("polish_thresholds", False)),
        polish_sweeps=int(budget["polish_sweeps"]),
        train_embed=bool(frozen.get("train_embed", False)),
        grouped_splits=bool(frozen.get("grouped_splits", False)),
        joint_residual=bool(frozen.get("joint_residual", False)),
        joint_steps=int(budget["joint_steps"]),
        binarize_eval=bool(frozen.get("binarize_eval", False)),
        pairwise_warmstart=bool(frozen.get("pairwise_warmstart", False)),
    )


def _lock_row(
    name: str,
    seeds: tuple[int, ...],
    budget: dict[str, int],
    frozen: dict[str, Any],
    vs_dl: bool,
    vs_v0: bool,
) -> dict[str, Any]:
    from omnibias.tab.bench import (
        fit_predict_catboost,
        fit_predict_lightgbm,
        fit_predict_realmlp,
        fit_predict_tabm,
        load_dataset,
        score_predictions,
        train_test_split,
    )
    from omnibias.tab.pou import fit_tabpou
    from omnibias.tab.pou.joint import fit_tabpou_joint

    try:
        ds = load_dataset(name, max_rows=int(budget["public_max_rows"]), seed=0)
    except (RuntimeError, ValueError) as exc:
        return {"name": name, "skipped": True, "reason": str(exc)}
    ours: list[float] = []
    v0s: list[float] = []
    lgbm: list[float] = []
    cat: list[float] = []
    real: list[float] = []
    tabm: list[float] = []
    for seed in seeds:
        Xtr, Xte, ytr, yte = train_test_split(ds, seed=int(seed))
        jcfg = _joint_from_frozen(int(Xtr.shape[1]), ds.task, budget, int(seed), frozen)
        model, _ = fit_tabpou_joint(Xtr, ytr, jcfg)
        ours.append(_score_model(model, Xte, yte, ds.task))
        if vs_v0:
            cfg0 = _base_cfg(
                int(Xtr.shape[1]), ds.task, budget, int(seed),
                use_embed=False, use_residual=False,
                n_stages=int(frozen.get("n_stages", budget["n_stages"])),
            )
            m0, _ = fit_tabpou(Xtr, ytr, cfg0)
            v0s.append(_score_model(m0, Xte, yte, ds.task))
        p_lgbm, pr = fit_predict_lightgbm(
            Xtr, ytr, Xte, task=ds.task, n_outputs=ds.n_outputs,
            n_estimators=int(budget["lgbm_estimators"]), seed=int(seed),
        )
        lgbm.append(float(score_predictions(yte, p_lgbm, pr, ds.task)["primary"]))
        try:
            p_cat, prc = fit_predict_catboost(
                Xtr, ytr, Xte, task=ds.task, n_outputs=ds.n_outputs,
                iterations=int(budget["cat_iterations"]), seed=int(seed),
            )
            cat.append(float(score_predictions(yte, p_cat, prc, ds.task)["primary"]))
        except Exception as exc:  # noqa: BLE001
            return {"name": name, "skipped": True, "reason": f"catboost unavailable: {exc}"}
        if vs_dl and ds.task == "regression":
            try:
                p_r, _ = fit_predict_realmlp(Xtr, ytr, Xte, n_epochs=30, seed=int(seed))
                real.append(float(score_predictions(yte, p_r, None, ds.task)["primary"]))
                p_t, _ = fit_predict_tabm(Xtr, ytr, Xte, n_epochs=30, seed=int(seed))
                tabm.append(float(score_predictions(yte, p_t, None, ds.task)["primary"]))
            except Exception as exc:  # noqa: BLE001
                vs_dl = False
                _ = str(exc)
    om, os_ = _mean_std(ours)
    lm, ls = _mean_std(lgbm)
    cm, cs = _mean_std(cat)
    row: dict[str, Any] = {
        "name": name, "skipped": False, "task": ds.task,
        "joint_mean": om, "lgbm_mean": lm, "cat_mean": cm,
        "not_worse_lgbm": _not_worse(om, lm, ls),
        "not_worse_cat": _not_worse(om, cm, cs),
    }
    if v0s:
        vm, vs = _mean_std(v0s)
        row["v0_mean"] = vm
        row["not_worse_v0"] = _not_worse(om, vm, vs)
    if real:
        rm, rs = _mean_std(real)
        tm, ts = _mean_std(tabm)
        row["realmlp_mean"] = rm
        row["tabm_mean"] = tm
        row["not_worse_realmlp"] = _not_worse(om, rm, rs)
        row["not_worse_tabm"] = _not_worse(om, tm, ts)
    return row


def _lock_synth(
    seeds: tuple[int, ...], budget: dict[str, int], frozen: dict[str, Any], vs_dl: bool
) -> dict[str, Any]:
    from omnibias.tab.bench import (
        fit_predict_realmlp,
        fit_predict_tabm,
        mlp_heteroscedastic_20d,
        score_predictions,
        train_test_split,
    )
    from omnibias.tab.pou.joint import fit_tabpou_joint

    ours: list[float] = []
    real: list[float] = []
    tabm: list[float] = []
    for seed in seeds:
        ds = mlp_heteroscedastic_20d(n=int(budget["g2_n"]), seed=int(seed))
        Xtr, Xte, ytr, yte = train_test_split(ds, seed=int(seed))
        jcfg = _joint_from_frozen(int(Xtr.shape[1]), "regression", budget, int(seed), frozen)
        model, _ = fit_tabpou_joint(Xtr, ytr, jcfg)
        ours.append(float(score_predictions(yte, model.predict(Xte), None, "regression")["primary"]))
        if vs_dl:
            try:
                p_r, _ = fit_predict_realmlp(Xtr, ytr, Xte, n_epochs=30, seed=int(seed))
                real.append(float(score_predictions(yte, p_r, None, "regression")["primary"]))
                p_t, _ = fit_predict_tabm(Xtr, ytr, Xte, n_epochs=30, seed=int(seed))
                tabm.append(float(score_predictions(yte, p_t, None, "regression")["primary"]))
            except Exception as exc:  # noqa: BLE001
                return {"name": "mlp_heteroscedastic_20d", "skipped": True, "reason": str(exc)}
    om, os_ = _mean_std(ours)
    row: dict[str, Any] = {
        "name": "mlp_heteroscedastic_20d", "skipped": False, "task": "regression",
        "joint_mean": om,
    }
    if real:
        rm, rs = _mean_std(real)
        tm, ts = _mean_std(tabm)
        row["realmlp_mean"] = rm
        row["tabm_mean"] = tm
        row["not_worse_realmlp"] = _not_worse(om, rm, rs)
        row["not_worse_tabm"] = _not_worse(om, tm, ts)
    return row


def _g1_vs_v0(budget: dict[str, int], seeds: tuple[int, ...], frozen: dict[str, Any]) -> dict[str, Any]:
    from omnibias.tab import make_axis_rule
    from omnibias.tab.pou import fit_tabpou
    from omnibias.tab.pou.joint import fit_tabpou_joint

    margins: list[float] = []
    for seed in seeds:
        X, y, _ = make_axis_rule(n_samples=int(budget["g1_n"]), n_features=8, seed=int(seed))
        n_tr = int(0.75 * X.shape[0])
        Xtr, Xte, ytr, yte = X[:n_tr], X[n_tr:], y[:n_tr], y[n_tr:]
        frozen_ax = {**frozen, "use_embed": False, "train_embed": False}
        jcfg = _joint_from_frozen(8, "binary", budget, int(seed), frozen_ax)
        jcfg = replace(jcfg, base=replace(jcfg.base, use_embed=False, robust_scale=False, patience=None))
        mj, _ = fit_tabpou_joint(Xtr, ytr, jcfg)
        m0, _ = fit_tabpou(Xtr, ytr, jcfg.base)
        margins.append(float(np.mean(mj.predict(Xte) == yte)) - float(np.mean(m0.predict(Xte) == yte)))
    worst = float(min(margins)) if margins else float("nan")
    return {"name": "G1_vs_v0", "worst_margin_vs_v0": worst, "passed": bool(worst >= -1e-12)}


def _leftovers(g_tree: dict[str, Any], g_net: dict[str, Any], g_hyb: dict[str, Any]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    if not g_tree.get("passed"):
        out.append({
            "family": "G-tree",
            "note": (
                "Unique split curvature was not enough vs CatBoost / LightGBM "
                "on tree-shaped public rows. Do not add ordered target statistics."
            ),
        })
    if not g_net.get("passed"):
        out.append({
            "family": "G-net",
            "note": (
                "Sequential/joint TabM-on-POU is not RealMLP/TabM on net-shaped "
                "rows. Certified v0 booster remains 05-04."
            ),
        })
    if not g_hyb.get("passed"):
        out.append({
            "family": "G-hybrid",
            "note": "Reject best-of-both-worlds: the mix destroyed the tree half vs v0.",
        })
    return out


def _run_named_h(
    name: str, budget: dict[str, int], seeds: tuple[int, ...], full: bool
) -> tuple[str, dict[str, Any]]:
    runners = {
        "H0": _h0,
        "H1": _h1,
        "H2": _h2,
        "H3": _h3,
        "H4": _h4,
        "H5": _h5,
        "H6": _h6,
        "H7": _h7,
    }
    return name, runners[name](budget, seeds if name != "H7" else seeds[:1], full)


def main() -> int:
    parser = argparse.ArgumentParser(description="TabPOU joint worlds 05-05")
    parser.add_argument("--full", action="store_true", help="Phase A hypothesis battery")
    parser.add_argument("--lock", action="store_true", help="Phase B family lock")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--frozen", type=str, default="", help="Path to frozen joint JSON")
    parser.add_argument("--hypotheses", type=str, default="", help="Comma list, e.g. H0,H1")
    parser.add_argument("--skip-g4", action="store_true", help="Skip RealMLP/TabM on --lock")
    args = parser.parse_args()
    if args.lock:
        tier = "lock"
    elif args.full:
        tier = "full"
    else:
        tier = "smoke"
    budget = BUDGET[tier]
    seeds = SEEDS_LOCK if args.lock else (SEEDS_FULL if args.full else SEEDS_SMOKE)
    if args.hypotheses:
        wanted = {h.strip() for h in args.hypotheses.split(",") if h.strip()}
    elif args.lock:
        wanted = set()
    elif args.full:
        wanted = set(ALL_HYP)
    else:
        wanted = {"H0", "H1"}
    t0 = time.perf_counter()
    g5 = _check_g5()
    hs: dict[str, dict[str, Any]] = {}
    selected = [n for n in ALL_HYP if n in wanted]
    if selected and (args.full or args.lock) and int(args.workers) > 1:
        pairs = _parallel_map(
            _run_named_h,
            [(n, budget, seeds, args.full) for n in selected],
            workers=max(1, int(args.workers)),
        )
        for name, row in pairs:
            hs[name] = row
    else:
        for name in selected:
            _, row = _run_named_h(name, budget, seeds, args.full)
            hs[name] = row
    for name in ALL_HYP:
        if name not in hs:
            hs[name] = _wiring(name, skipped=True, reason="not selected")
    frozen = freeze_from_hypotheses(hs)
    if args.frozen:
        frozen = json.loads(Path(args.frozen).read_text(encoding="utf-8"))
    elif args.lock:
        cand = SCRATCH / "tabular_pou_joint_frozen.json"
        if cand.is_file():
            frozen = json.loads(cand.read_text(encoding="utf-8"))
        else:
            print("warning: no frozen joint JSON; lock uses hypothesis-derived defaults", flush=True)

    g_tree: dict[str, Any]
    g_net: dict[str, Any]
    g_hyb: dict[str, Any]
    g1 = {"name": "G1_vs_v0", "passed": True, "wiring_only": True}
    leftovers: list[dict[str, str]] = []
    if args.lock:
        workers = max(1, int(args.workers))
        tree_rows = _parallel_map(
            _lock_row,
            [(n, seeds, budget, frozen, False, True) for n in TREE_PUBLIC],
            workers=workers,
        )
        scored_t = [r for r in tree_rows if not r.get("skipped")]
        nw_t = sum(1 for r in scored_t if r.get("not_worse_lgbm") and r.get("not_worse_cat"))
        g1 = _g1_vs_v0(budget, seeds, frozen)
        g_tree = {
            "name": "G-tree",
            "passed": bool(len(scored_t) >= 3 and nw_t >= 3 and g1["passed"]),
            "not_worse_both": nw_t, "n_scored": len(scored_t), "rows": tree_rows, "g1": g1,
        }
        if args.skip_g4:
            g_net = {
                "name": "G-net", "passed": False, "skipped": True,
                "reason": "RealMLP/TabM skipped; G-net is unearned",
                "rows": [],
            }
        else:
            net_rows = _parallel_map(
                _lock_row,
                [(n, seeds, budget, frozen, True, True) for n in NET_PUBLIC],
                workers=workers,
            )
            net_rows.append(_lock_synth(seeds, budget, frozen, True))
            scored_n = [r for r in net_rows if not r.get("skipped") and "not_worse_realmlp" in r]
            nw_n = sum(1 for r in scored_n if r.get("not_worse_realmlp") and r.get("not_worse_tabm"))
            g_net = {
                "name": "G-net",
                "passed": bool(len(scored_n) >= 2 and nw_n >= 2),
                "not_worse_both": nw_n, "n_comparable": len(scored_n), "rows": net_rows,
            }
        hyb_rows = _parallel_map(
            _lock_row,
            [(n, seeds, budget, frozen, False, True) for n in HYBRID_PUBLIC],
            workers=workers,
        )
        scored_h = [r for r in hyb_rows if not r.get("skipped") and "not_worse_v0" in r]
        nw_h = sum(1 for r in scored_h if r.get("not_worse_v0"))
        g_hyb = {
            "name": "G-hybrid",
            "passed": bool(len(scored_h) >= 5 and nw_h == len(scored_h)),
            "not_worse_v0": nw_h, "n_scored": len(scored_h), "rows": hyb_rows,
        }
        leftovers = _leftovers(g_tree, g_net, g_hyb)
    else:
        g_tree = {"name": "G-tree", "passed": True, "wiring_only": True}
        g_net = {"name": "G-net", "passed": True, "wiring_only": True}
        g_hyb = {"name": "G-hybrid", "passed": True, "wiring_only": True}

    h0, h1, h2, h3, h4, h5, h6, h7 = (hs[k] for k in ALL_HYP)
    entries = [h0, h1, h2, h3, h4, h5, h6, h7, g_tree, g_net, g_hyb, g5]
    smoke_ok = bool(g5["passed"] and h0.get("passed", True))
    lock_ok = bool(
        g5["passed"] and g_tree.get("passed") and g_net.get("passed") and g_hyb.get("passed")
    )
    licensed = (
        "shipped" if (args.lock and lock_ok) else ("gated" if args.lock else "designed")
    )
    payload = {
        **provenance(
            schema="omnibias.tab.pou.joint.v1",
            config={
                "tier": tier, "budget": budget, "seeds": list(seeds),
                "workers": int(args.workers), "hypotheses": sorted(wanted),
            },
        ),
        "tier": tier,
        "elapsed_seconds": round(time.perf_counter() - t0, 3),
        "h0": h0, "h1": h1, "h2": h2, "h3": h3, "h4": h4, "h5": h5, "h6": h6, "h7": h7,
        "frozen": frozen,
        "g_tree": g_tree, "g_net": g_net, "g_hybrid": g_hyb, "g5": g5,
        "leftovers": leftovers,
        "licensed_status": licensed,
        "gates": gates_block(entries),
        "smoke_process_passed": smoke_ok,
        "licensed_sentence": (
            "05-05 joint TabPOU: H0-H7 are hypothesis arms earned on --full; "
            "G-tree/G-net/G-hybrid/G5 are lock bars; 05-04 G3 stays report-only v0; "
            "temperature collapse only; no TabPFN-3 claim."
        ),
    }
    SCRATCH.mkdir(parents=True, exist_ok=True)
    if args.lock:
        out = SCRATCH / "tabular_pou_joint_lock.json"
        out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {out}")
    elif args.full:
        out = SCRATCH / "tabular_pou_joint_full.json"
        out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        fz = SCRATCH / "tabular_pou_joint_frozen.json"
        fz.write_text(json.dumps(frozen, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {out} and {fz}")
    else:
        path = write_json("tabular_pou_joint_smoke.json", payload)
        print(f"wrote {path}")
    ok = lock_ok if args.lock else smoke_ok
    print(json.dumps({"ok": ok, "tier": tier, "g5": g5["passed"], "frozen": frozen, "status": licensed}, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
