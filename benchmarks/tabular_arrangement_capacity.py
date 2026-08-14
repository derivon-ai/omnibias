# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Theory 05-02 G3b: capacity / optimizer ablations vs the same fair LightGBM.

G3 (single H=2 arrangement) stays frozen. This suite runs enhancement arms in
parallel against **one** LightGBM fit per dataset×seed under the same fair
protocol: train on ``Xtr``, early-stop + restore best on ``Xva``, score ``Xte``.

Primary licensed arm is predeclared: ``boost_h2`` (Newton-boosted H=2
arrangements). G3b is earned only if that arm is not-worse (win or 0.5-pt tie)
on >=6 of 8 completed public binary datasets. Other arms are ablations; do not
relicense after seeing test. If ``tab_boost`` would have earned G3b and
``boost_h2`` did not, record that as a finding.
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
from _common import (  # type: ignore[import-not-found]  # noqa: E402
    provenance,
    write_json,
)
from tabular_arrangement_public import (  # type: ignore[import-not-found]  # noqa: E402
    ARRANGEMENT_BUDGET,
    LGBM_GRID_FULL,
    LGBM_GRID_SMOKE,
    TIE_BAND,
    _accuracy,
    _auc,
    _fit_arrangement_arms,
    _fit_lightgbm,
    _winner,
)

SCRATCH = Path(os.environ.get("OMNIBIAS_SCRATCH", "artifacts"))

PRIMARY_ARM = "boost_h2"
G3B_MIN_DATASETS = 6
SEEDS_FULL = (0, 1, 2, 3, 4)
SEEDS_SMOKE = (0, 1)
ARM_ORDER = (
    "h2_adam",
    "h2_newton",
    "h3_adam",
    "h4_adam",
    "boost_h2",
    "tab_boost",
    "tab_joint",
)

# Full boosting caps are LightGBM-style n_estimators ceilings (G3 uses 4000
# trees). Weak learners here are H=2 arrangements / depth-2 soft trees, so the
# cap is 400; --full still INVALID EXPERIMENT if boost_h2 never plateaus.
CAPACITY_BUDGET = {
    "full": {
        "h2_newton": {
            "restarts": 4,
            "steps": 80,
            "patience": 15,
            "beta_final": 64.0,
            "optimizer": "trust_region",
        },
        "h3_adam": {
            "n_hyperplanes": 3,
            "restarts": 4,
            "steps": 2000,
            "patience": 50,
            "beta_final": 64.0,
        },
        "h4_adam": {
            "n_hyperplanes": 4,
            "restarts": 3,
            "steps": 2000,
            "patience": 50,
            "beta_final": 64.0,
        },
        "boost_h2": {
            "n_stages_max": 400,
            "learning_rate": 0.3,
            "stage_patience": 5,
            "weak_restarts": 2,
            "weak_steps": 200,
            "weak_patience": 20,
            "beta_final": 64.0,
        },
        "tab_boost": {
            "n_stages": 400,
            "learning_rate": 0.3,
            "inner_steps": 60,
            "patience": 5,
            "n_trees": 1,
            "depth": 2,
            "beta_final": 32.0,
        },
        "tab_joint": {
            "n_trees": 16,
            "depth": 2,
            "steps": 80,
            "patience": 15,
            "beta_final": 32.0,
        },
    },
    "smoke": {
        "h2_newton": {
            "restarts": 1,
            "steps": 12,
            "patience": 8,
            "beta_final": 16.0,
            "optimizer": "trust_region",
        },
        "h3_adam": {
            "n_hyperplanes": 3,
            "restarts": 1,
            "steps": 40,
            "patience": 15,
            "beta_final": 16.0,
        },
        "h4_adam": {
            "n_hyperplanes": 4,
            "restarts": 1,
            "steps": 40,
            "patience": 15,
            "beta_final": 16.0,
        },
        "boost_h2": {
            "n_stages_max": 3,
            "learning_rate": 0.4,
            "stage_patience": 2,
            "weak_restarts": 1,
            "weak_steps": 30,
            "weak_patience": 12,
            "beta_final": 16.0,
        },
        "tab_boost": {
            "n_stages": 3,
            "learning_rate": 0.4,
            "inner_steps": 15,
            "patience": 2,
            "n_trees": 1,
            "depth": 2,
            "beta_final": 16.0,
        },
        "tab_joint": {
            "n_trees": 4,
            "depth": 2,
            "steps": 12,
            "patience": 6,
            "beta_final": 16.0,
        },
    },
}


def _score_predict(
    pred: np.ndarray,
    proba: np.ndarray,
    split: dict[str, np.ndarray],
    *,
    train_acc: float,
    val_acc: float,
    extra: dict[str, Any],
) -> dict[str, Any]:
    row = {
        "accuracy": _accuracy(pred, split["yte"]),
        "auc": _auc(proba, split["yte"]),
        "train_acc": float(train_acc),
        "val_acc": float(val_acc),
        "train_val_gap": float(train_acc - val_acc),
    }
    row.update(extra)
    return row


def _fit_h_adam(
    split: dict[str, np.ndarray],
    *,
    seed: int,
    n_hyperplanes: int,
    budget: dict[str, Any],
) -> dict[str, Any]:
    from omnibias.tab.torch.arrangement import fit_arrangement

    result = fit_arrangement(
        split["Xtr"],
        split["ytr"],
        X_val=split["Xva"],
        y_val=split["yva"],
        n_hyperplanes=int(n_hyperplanes),
        l1=0.0,
        restarts=int(budget["restarts"]),
        steps=int(budget["steps"]),
        beta_final=float(budget["beta_final"]),
        seed=seed,
        patience=int(budget["patience"]),
        sparse_warmstart=False,
        optimizer="adam",
    )
    pred = result.model.predict(split["Xte"])
    proba = result.model.predict_proba(split["Xte"])
    return _score_predict(
        pred,
        proba,
        split,
        train_acc=result.train_acc,
        val_acc=result.val_acc,
        extra={
            "val_bce": result.val_bce,
            "steps_run": result.steps_run,
            "best_step": result.best_step,
            "stopped_early": result.stopped_early,
            "at_step_cap": result.at_step_cap,
            "n_hyperplanes": int(n_hyperplanes),
            "optimizer": "adam",
        },
    )


def _fit_h2_newton(
    split: dict[str, np.ndarray],
    *,
    seed: int,
    budget: dict[str, Any],
) -> dict[str, Any]:
    from omnibias.tab.torch.arrangement import fit_arrangement

    result = fit_arrangement(
        split["Xtr"],
        split["ytr"],
        X_val=split["Xva"],
        y_val=split["yva"],
        n_hyperplanes=2,
        l1=0.0,
        restarts=int(budget["restarts"]),
        steps=int(budget["steps"]),
        beta_final=float(budget["beta_final"]),
        seed=seed,
        patience=int(budget["patience"]),
        sparse_warmstart=False,
        optimizer=str(budget.get("optimizer", "trust_region")),
    )
    pred = result.model.predict(split["Xte"])
    proba = result.model.predict_proba(split["Xte"])
    return _score_predict(
        pred,
        proba,
        split,
        train_acc=result.train_acc,
        val_acc=result.val_acc,
        extra={
            "val_bce": result.val_bce,
            "steps_run": result.steps_run,
            "best_step": result.best_step,
            "stopped_early": result.stopped_early,
            "at_step_cap": result.at_step_cap,
            "optimizer": result.optimizer,
        },
    )


def _fit_boost_h2(
    split: dict[str, np.ndarray],
    *,
    seed: int,
    budget: dict[str, Any],
    require_converged: bool,
) -> dict[str, Any]:
    from omnibias.tab.torch.arrangement import fit_arrangement_boosted

    result = fit_arrangement_boosted(
        split["Xtr"],
        split["ytr"],
        X_val=split["Xva"],
        y_val=split["yva"],
        n_hyperplanes=2,
        n_stages_max=int(budget["n_stages_max"]),
        learning_rate=float(budget["learning_rate"]),
        stage_patience=int(budget["stage_patience"]),
        weak_restarts=int(budget["weak_restarts"]),
        weak_steps=int(budget["weak_steps"]),
        weak_patience=int(budget["weak_patience"]),
        beta_final=float(budget["beta_final"]),
        seed=seed,
    )
    if require_converged and bool(result.at_stage_cap):
        raise RuntimeError(
            "INVALID EXPERIMENT: boost_h2 hit the stage cap without early-stopping "
            f"(n_stages={result.n_stages}, budget={budget}). Raise n_stages_max "
            "before reading G3b."
        )
    pred = result.model.predict(split["Xte"])
    proba = result.model.predict_proba(split["Xte"])
    return _score_predict(
        pred,
        proba,
        split,
        train_acc=result.train_acc,
        val_acc=result.val_acc,
        extra={
            "val_bce": result.val_bce,
            "n_stages": result.n_stages,
            "best_stage": result.best_stage,
            "stopped_early": result.stopped_early,
            "at_stage_cap": result.at_stage_cap,
        },
    )


def _fit_tab_boost(
    split: dict[str, np.ndarray],
    *,
    seed: int,
    budget: dict[str, Any],
) -> dict[str, Any]:
    from omnibias.tab._core.config import SoftTreeConfig
    from omnibias.tab.torch.boosting import fit_boosted

    d = int(split["Xtr"].shape[1])
    cfg = SoftTreeConfig(
        n_features=d,
        n_trees=int(budget["n_trees"]),
        depth=int(budget["depth"]),
        task="binary",
        beta_final=float(budget["beta_final"]),
        seed=seed,
    )
    model, result = fit_boosted(
        split["Xtr"],
        split["ytr"],
        cfg,
        n_stages=int(budget["n_stages"]),
        learning_rate=float(budget["learning_rate"]),
        inner_steps=int(budget["inner_steps"]),
        val=(split["Xva"], split["yva"]),
        patience=int(budget["patience"]),
    )
    pred = model.predict(split["Xte"])
    proba = model.predict_proba(split["Xte"])
    train_acc = _accuracy(model.predict(split["Xtr"]), split["ytr"])
    val_acc = _accuracy(model.predict(split["Xva"]), split["yva"])
    return _score_predict(
        pred,
        proba,
        split,
        train_acc=train_acc,
        val_acc=val_acc,
        extra={
            "n_stages": result.n_stages,
            "val_metric": result.val_metric,
        },
    )


def _fit_tab_joint(
    split: dict[str, np.ndarray],
    *,
    seed: int,
    budget: dict[str, Any],
) -> dict[str, Any]:
    from omnibias.tab._core.config import SoftTreeConfig
    from omnibias.tab.torch.model import SoftTreeEnsemble
    from omnibias.tab.torch.train import fit_second_order

    d = int(split["Xtr"].shape[1])
    cfg = SoftTreeConfig(
        n_features=d,
        n_trees=int(budget["n_trees"]),
        depth=int(budget["depth"]),
        task="binary",
        beta_final=float(budget["beta_final"]),
        seed=seed,
    )
    model = SoftTreeEnsemble(cfg)
    fit_second_order(
        model,
        split["Xtr"],
        split["ytr"],
        optimizer="trust_region",
        steps=int(budget["steps"]),
        val=(split["Xva"], split["yva"]),
        patience=int(budget["patience"]),
    )
    pred = model.predict(split["Xte"])
    proba = model.predict_proba(split["Xte"])
    train_acc = _accuracy(model.predict(split["Xtr"]), split["ytr"])
    val_acc = _accuracy(model.predict(split["Xva"]), split["yva"])
    return _score_predict(
        pred,
        proba,
        split,
        train_acc=train_acc,
        val_acc=val_acc,
        extra={"optimizer": "trust_region", "n_trees": int(budget["n_trees"])},
    )


def _run_arms(
    split: dict[str, np.ndarray],
    *,
    seed: int,
    budgets: dict[str, dict[str, Any]],
    arr_budget: dict[str, dict[str, Any]],
    require_converged: bool,
    label: str = "",
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}

    def _mark(arm_name: str, t0: float) -> None:
        acc = out[arm_name]["accuracy"]
        dt = time.perf_counter() - t0
        print(f"    {label}{arm_name} acc={acc:.4f} ({dt:.1f}s)", flush=True)

    t = time.perf_counter()
    h2 = _fit_arrangement_arms(
        split, seed=seed, budgets=arr_budget, require_converged=False
    )
    out["h2_adam"] = {
        "accuracy": h2["accuracy"],
        "auc": h2["auc"],
        "train_acc": h2["train_acc"],
        "val_acc": h2["val_acc"],
        "train_val_gap": h2["train_val_gap"],
        "arm": h2["arm"],
        "stopped_early": h2["stopped_early"],
        "at_step_cap": h2["at_step_cap"],
        "val_bce": h2["val_bce"],
    }
    _mark("h2_adam", t)
    t = time.perf_counter()
    out["h2_newton"] = _fit_h2_newton(split, seed=seed, budget=budgets["h2_newton"])
    _mark("h2_newton", t)
    t = time.perf_counter()
    out["h3_adam"] = _fit_h_adam(
        split, seed=seed, n_hyperplanes=3, budget=budgets["h3_adam"]
    )
    _mark("h3_adam", t)
    t = time.perf_counter()
    out["h4_adam"] = _fit_h_adam(
        split, seed=seed, n_hyperplanes=4, budget=budgets["h4_adam"]
    )
    _mark("h4_adam", t)
    t = time.perf_counter()
    out["boost_h2"] = _fit_boost_h2(
        split,
        seed=seed,
        budget=budgets["boost_h2"],
        require_converged=require_converged,
    )
    _mark("boost_h2", t)
    t = time.perf_counter()
    out["tab_boost"] = _fit_tab_boost(split, seed=seed, budget=budgets["tab_boost"])
    _mark("tab_boost", t)
    t = time.perf_counter()
    out["tab_joint"] = _fit_tab_joint(split, seed=seed, budget=budgets["tab_joint"])
    _mark("tab_joint", t)
    return out


def _run_dataset(
    name: str,
    *,
    full: bool,
    grid: list[tuple[int, float, int, int]],
    budgets: dict[str, dict[str, Any]],
    arr_budget: dict[str, dict[str, Any]],
    seeds: tuple[int, ...],
    max_rows: int | None,
) -> dict[str, Any]:
    from omnibias.tab.bench import load_dataset, train_val_test_split

    t0 = time.perf_counter()
    try:
        ds = load_dataset(name, max_rows=max_rows, seed=0)
    except RuntimeError as exc:
        return {
            "dataset": name,
            "status": "skipped",
            "skip_reason": str(exc),
            "per_seed": [],
            "wall_seconds": round(time.perf_counter() - t0, 3),
        }
    if ds.task != "binary":
        return {
            "dataset": name,
            "status": "skipped",
            "skip_reason": f"expected binary, got {ds.task!r}",
            "per_seed": [],
            "wall_seconds": round(time.perf_counter() - t0, 3),
        }

    per_seed: list[dict[str, Any]] = []
    for seed in seeds:
        split = train_val_test_split(ds, seed=seed)
        print(f"  {name} seed={seed}: fitting LightGBM...", flush=True)
        lgbm = _fit_lightgbm(split, seed=seed, grid=grid, require_interior=full)
        print(
            f"    {name} seed={seed} lightgbm acc={lgbm['accuracy']:.4f}",
            flush=True,
        )
        arms = _run_arms(
            split,
            seed=seed,
            budgets=budgets,
            arr_budget=arr_budget,
            require_converged=full,
            label=f"{name} seed={seed} ",
        )
        row: dict[str, Any] = {
            "seed": int(seed),
            "lightgbm_accuracy": float(lgbm["accuracy"]),
            "lightgbm_auc": float(lgbm["auc"]),
            "lightgbm": {
                "train_acc": lgbm["train_acc"],
                "val_acc": lgbm["val_acc"],
                "train_val_gap": lgbm["train_val_gap"],
                "best_iteration": lgbm["best_iteration"],
                "stopped_early": lgbm["stopped_early"],
            },
            "arms": {},
        }
        bits = []
        for arm_name in ARM_ORDER:
            a = arms[arm_name]
            margin = float(a["accuracy"] - lgbm["accuracy"])
            rec = dict(a)
            rec["margin"] = margin
            rec["winner"] = _winner(a["accuracy"], lgbm["accuracy"])
            row["arms"][arm_name] = rec
            bits.append(f"{arm_name}={a['accuracy']:.3f}")
        per_seed.append(row)
        print(
            f"  {name} seed={seed}: lgbm={lgbm['accuracy']:.4f} " + " ".join(bits)
        )

    arm_summary: dict[str, Any] = {}
    for arm_name in ARM_ORDER:
        accs = [r["arms"][arm_name]["accuracy"] for r in per_seed]
        vals = [r["arms"][arm_name]["val_acc"] for r in per_seed]
        lgbms = [r["lightgbm_accuracy"] for r in per_seed]
        mean_acc = float(np.mean(accs))
        mean_lgbm = float(np.mean(lgbms))
        mean_margin = float(mean_acc - mean_lgbm)
        arm_summary[arm_name] = {
            "mean_accuracy": mean_acc,
            "mean_val_acc": float(np.mean(vals)),
            "mean_lightgbm_accuracy": mean_lgbm,
            "mean_margin": mean_margin,
            "winner": _winner(mean_acc, mean_lgbm),
            "not_worse": _winner(mean_acc, mean_lgbm) in ("arrangement", "tie"),
        }
    return {
        "dataset": name,
        "status": "completed",
        "n_rows": int(ds.X.shape[0]),
        "n_features": int(ds.X.shape[1]),
        "max_rows": max_rows,
        "per_seed": per_seed,
        "arm_summary": arm_summary,
        "wall_seconds": round(time.perf_counter() - t0, 3),
    }


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--full",
        action="store_true",
        help="acceptance tier: all eight datasets + full arm budgets",
    )
    args = parser.parse_args(argv)
    full = bool(args.full)
    tier = "full" if full else "smoke"
    grid = LGBM_GRID_FULL if full else LGBM_GRID_SMOKE
    budgets = CAPACITY_BUDGET[tier]
    arr_budget = ARRANGEMENT_BUDGET[tier]
    seeds = SEEDS_FULL if full else SEEDS_SMOKE
    artifact_name = (
        "tabular_arrangement_capacity.json"
        if full
        else "tabular_arrangement_capacity_smoke.json"
    )

    from omnibias.tab.bench import (
        ARRANGEMENT_PUBLIC_MAX_ROWS,
        ARRANGEMENT_PUBLIC_SUITE,
    )

    names = list(ARRANGEMENT_PUBLIC_SUITE) if full else ["breast_cancer"]
    config = {
        "family": "tabular_arrangement_capacity_vs_lightgbm",
        "suite": list(ARRANGEMENT_PUBLIC_SUITE),
        "datasets_run": names,
        "seeds": list(seeds),
        "split": {"train": 0.6, "val": 0.2, "test": 0.2},
        "tie_band_accuracy_points": TIE_BAND,
        "primary_arm": PRIMARY_ARM,
        "g3b_min_datasets": G3B_MIN_DATASETS,
        "arm_order": list(ARM_ORDER),
        "lightgbm_grid_size": len(grid),
        "arrangement_budget": arr_budget,
        "capacity_budget": budgets,
        "fair_protocol": {
            "train_on": "Xtr",
            "early_stop_on": "Xva",
            "score_on": "Xte",
            "no_train_val_refit": True,
            "arm_selection": "val_never_test",
        },
        "g3_frozen": True,
        "tier": tier,
        "smoke_is_wiring_gate": not full,
        "full": full,
    }
    payload = provenance(schema="tabular-arrangement-capacity-v1", config=config)

    t0 = time.perf_counter()
    blocks: dict[str, Any] = {}
    for name in names:
        print(f"running {name}...")
        max_rows = ARRANGEMENT_PUBLIC_MAX_ROWS.get(name)
        if not full and name == "breast_cancer":
            max_rows = 400
        blocks[name] = _run_dataset(
            name,
            full=full,
            grid=grid,
            budgets=budgets,
            arr_budget=arr_budget,
            seeds=seeds,
            max_rows=max_rows,
        )

    completed = [b for b in blocks.values() if b["status"] == "completed"]
    skipped = [b for b in blocks.values() if b["status"] == "skipped"]
    if full and len(completed) < 8:
        raise RuntimeError(
            "INVALID EXPERIMENT: need >=8 completed public binary datasets "
            f"for G3b; got {len(completed)} completed, {len(skipped)} skipped "
            f"({[s['dataset'] for s in skipped]})"
        )

    win_loss_by_arm: dict[str, Any] = {}
    for arm_name in ARM_ORDER:
        table = []
        n_arr = n_lgbm = n_tie = 0
        n_not_worse = 0
        for b in (blocks[n] for n in names):
            if b["status"] != "completed":
                table.append(
                    {
                        "dataset": b["dataset"],
                        "status": b["status"],
                        "winner": None,
                        "skip_reason": b.get("skip_reason"),
                    }
                )
                continue
            s = b["arm_summary"][arm_name]
            winner = s["winner"]
            if winner == "arrangement":
                n_arr += 1
            elif winner == "lightgbm":
                n_lgbm += 1
            else:
                n_tie += 1
            if s["not_worse"]:
                n_not_worse += 1
            table.append(
                {
                    "dataset": b["dataset"],
                    "status": "completed",
                    "winner": winner,
                    "mean_accuracy": s["mean_accuracy"],
                    "mean_lightgbm_accuracy": s["mean_lightgbm_accuracy"],
                    "mean_margin": s["mean_margin"],
                    "not_worse": s["not_worse"],
                }
            )
        win_loss_by_arm[arm_name] = {
            "win_loss_table": table,
            "counts": {
                "arrangement": n_arr,
                "lightgbm": n_lgbm,
                "tie": n_tie,
                "not_worse": n_not_worse,
                "completed": len(completed),
            },
        }

    primary = win_loss_by_arm[PRIMARY_ARM]["counts"]
    g3b_earned = bool(
        full
        and len(completed) >= 8
        and int(primary["not_worse"]) >= G3B_MIN_DATASETS
    )
    tab_boost_nw = int(win_loss_by_arm["tab_boost"]["counts"]["not_worse"])
    tab_boost_would_earn = bool(
        full and len(completed) >= 8 and tab_boost_nw >= G3B_MIN_DATASETS
    )
    finding = None
    if tab_boost_would_earn and not g3b_earned:
        finding = (
            "tab_boost is not-worse on "
            f"{tab_boost_nw}/{len(completed)} datasets but predeclared "
            f"primary {PRIMARY_ARM} is not; G3b stays unearned (no relicense)"
        )

    if g3b_earned:
        licensed = (
            f"on {len(completed)} public binary datasets Newton-boosted H=2 "
            f"arrangements vs tuned LightGBM not-worse on {primary['not_worse']}/"
            f"{len(completed)} (W/L/T="
            f"{primary['arrangement']}/{primary['lightgbm']}/{primary['tie']}; "
            f"primary arm {PRIMARY_ARM}; per-dataset table in win_loss_by_arm)"
        )
    elif not full:
        licensed = (
            "smoke wiring only: breast_cancer exercise of the capacity suite; "
            "G3b not earned"
        )
    else:
        licensed = (
            f"G3b not earned: primary arm {PRIMARY_ARM} not-worse on "
            f"{primary['not_worse']}/{len(completed)} "
            f"(need >={G3B_MIN_DATASETS}); G3 H=2 table stays frozen"
        )

    payload.update(
        {
            "baseline": {
                "name": "tuned LightGBM (same fair protocol as G3)",
                "note": "one LightGBM fit per dataset×seed; all arms scored against it",
            },
            "seeds": list(seeds),
            "datasets": blocks,
            "win_loss_by_arm": win_loss_by_arm,
            "primary_arm": PRIMARY_ARM,
            "honesty": {
                "claim_rung": 1,
                "g3_frozen": True,
                "primary_arm_predeclared": PRIMARY_ARM,
                "arm_selection_on_val_never_test": True,
                "no_relicense_after_test": True,
                "trees_expected_to_win_most": True,
                "smoke_is_wiring_gate": not full,
                "g3_earned": True,
                "g3b_earned": g3b_earned,
                "tab_boost_would_earn_g3b": tab_boost_would_earn,
                "finding": finding,
                "theorem_prover_verified": False,
                "mathlib_verified": False,
                "licensed_sentence": licensed,
            },
            "g3b_earned": g3b_earned,
            "wall_seconds": round(time.perf_counter() - t0, 3),
        }
    )
    out = write_json(artifact_name, payload)
    print(
        f"wrote {out}  completed={len(completed)} "
        f"primary_not_worse={primary['not_worse']}/{len(completed)} "
        f"g3b={g3b_earned}"
    )
    if full:
        scratch_dir = SCRATCH / "beyond_pde"
        scratch_dir.mkdir(parents=True, exist_ok=True)
        scratch_path = scratch_dir / artifact_name
        scratch_path.write_text(out.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"copied to {scratch_path}")
    return dict(payload)


if __name__ == "__main__":
    main()
