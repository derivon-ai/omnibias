# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Theory 05-02 G3: real-world H=2 arrangement vs tuned LightGBM.

Eight public *binary* datasets (see ``omnibias.tab.bench.ARRANGEMENT_PUBLIC_SUITE``).
Fair protocol (same for both arms): stratified 60/20/20, StandardScaler fit on
train only, five seeds; train on ``Xtr``, early-stop + restore best on ``Xva``,
score ``Xte`` -- **no train+val refit**. Tuned LightGBM early-stops on val
logloss; H=2 ``fit_arrangement`` early-stops on val BCE (dense ``l1=0`` + sparse
``l1>0`` arms; keep better val BCE). ``--full`` rejects LightGBM grid-boundary
configs and arrangement arms that hit the Adam step cap without plateauing.

Honesty: trees are expected to win most rows. G3 is earned by publishing the
full per-dataset win/loss table (no aggregate-only licensed sentence), not by
beating LightGBM. G4 (obliqueness diagnostic predictiveness) is reported with
a *frozen* threshold (``diag > 1.0`` -> arrangement) and is earned only if that
rule hits >=75% of completed datasets -- do not retune the diagnostic here.
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from _common import (  # type: ignore[import-not-found]  # noqa: E402
    provenance,
    write_json,
)

SCRATCH = Path(os.environ.get("OMNIBIAS_SCRATCH", "artifacts"))

SEEDS = (0, 1, 2, 3, 4)
TIE_BAND = 0.005  # 0.5 accuracy points
G4_DIAG_THRESHOLD = 1.0  # frozen; do not retune on these eight datasets
G4_MIN_ACCURACY = 0.75
LGBM_N_ESTIMATORS_MAX = 4000
LGBM_EARLY_STOPPING = 50

LGBM_GRID_FULL: list[tuple[int, float, int, int]] = list(
    itertools.product(
        [31, 127, 255, 511],
        [0.05, 0.1],
        [5, 20],
        [-1, 8],
    )
)
LGBM_GRID_SMOKE: list[tuple[int, float, int, int]] = list(
    itertools.product([31, 127], [0.05, 0.1], [20], [-1])
)

ARRANGEMENT_PATIENCE = 50
ARRANGEMENT_BUDGET = {
    "full": {
        "dense": {
            "restarts": 6,
            "steps": 5000,
            "l1": 0.0,
            "beta_final": 64.0,
            "patience": ARRANGEMENT_PATIENCE,
            "min_delta": 1e-4,
        },
        "sparse": {
            "restarts": 4,
            "steps": 5000,
            "l1": 0.02,
            "beta_final": 128.0,
            "patience": ARRANGEMENT_PATIENCE,
            "min_delta": 1e-4,
        },
    },
    "smoke": {
        "dense": {
            "restarts": 2,
            "steps": 200,
            "l1": 0.0,
            "beta_final": 32.0,
            "patience": 20,
            "min_delta": 1e-4,
        },
        "sparse": {
            "restarts": 1,
            "steps": 150,
            "l1": 0.02,
            "beta_final": 64.0,
            "patience": 20,
            "min_delta": 1e-4,
        },
    },
}


def _accuracy(pred: np.ndarray, y: np.ndarray) -> float:
    return float((np.asarray(pred).reshape(-1) == np.asarray(y).reshape(-1)).mean())


def _auc(scores: np.ndarray, y: np.ndarray) -> float:
    """Mann-Whitney AUC for binary labels in {0,1}, averaging ranks on ties."""
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    yy = np.asarray(y, dtype=np.float64).reshape(-1)
    pos = s[yy > 0.5]
    neg = s[yy <= 0.5]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(s.size, dtype=np.float64)
    sorted_s = s[order]
    i = 0
    while i < s.size:
        j = i + 1
        while j < s.size and sorted_s[j] == sorted_s[i]:
            j += 1
        avg = 0.5 * ((i + 1) + j)
        ranks[order[i:j]] = avg
        i = j
    sum_pos = float(ranks[yy > 0.5].sum())
    n_pos = float(pos.size)
    n_neg = float(neg.size)
    return (sum_pos - n_pos * (n_pos + 1.0) / 2.0) / (n_pos * n_neg)


def _grid_boundary(
    cfg: dict[str, Any],
    *,
    leaf_choices: list[int],
    n_estimators_used: int,
    n_estimators_max: int,
) -> bool:
    leaves = int(cfg["num_leaves"])
    at_leaf_max = leaves == max(leaf_choices)
    at_est_cap = int(n_estimators_used) >= int(n_estimators_max)
    return bool(at_leaf_max or at_est_cap)


def _fit_lightgbm(
    split: dict[str, np.ndarray],
    *,
    seed: int,
    grid: list[tuple[int, float, int, int]],
    require_interior: bool,
) -> dict[str, Any]:
    import lightgbm as lgb

    leaf_choices = sorted({g[0] for g in grid})
    best_cfg: tuple[int, float, int, int] | None = None
    best_val = -1.0
    best_n_est = 0
    best_model = None
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="X does not have valid feature names")
        warnings.filterwarnings("ignore", category=UserWarning)
        for leaves, lr, min_child, max_depth in grid:
            model = lgb.LGBMClassifier(
                num_leaves=leaves,
                learning_rate=lr,
                n_estimators=LGBM_N_ESTIMATORS_MAX,
                min_child_samples=min_child,
                max_depth=max_depth,
                random_state=seed,
                verbose=-1,
                n_jobs=1,
            )
            model.fit(
                split["Xtr"],
                split["ytr"].astype(np.int64),
                eval_X=split["Xva"],
                eval_y=split["yva"].astype(np.int64),
                callbacks=[
                    lgb.early_stopping(LGBM_EARLY_STOPPING, verbose=False),
                    lgb.log_evaluation(period=0),
                ],
            )
            n_est = int(getattr(model, "best_iteration_", None) or model.n_estimators_)
            pred_va = model.predict(split["Xva"]).astype(np.float64)
            acc = _accuracy(pred_va, split["yva"])
            if acc > best_val:
                best_val = acc
                best_cfg = (leaves, lr, min_child, max_depth)
                best_n_est = n_est
                best_model = model
        assert best_cfg is not None and best_model is not None
        leaves, lr, min_child, max_depth = best_cfg
        # Fair protocol: keep the early-stopped Xtr model (no train+val refit).
        pred = best_model.predict(split["Xte"]).astype(np.float64)
        proba = best_model.predict_proba(split["Xte"])[:, 1]
        train_acc = _accuracy(
            best_model.predict(split["Xtr"]).astype(np.float64), split["ytr"]
        )
        val_acc = float(best_val)
    cfg = {
        "num_leaves": leaves,
        "learning_rate": lr,
        "n_estimators": int(best_n_est),
        "min_child_samples": min_child,
        "max_depth": max_depth,
        "n_estimators_max": LGBM_N_ESTIMATORS_MAX,
        "early_stopping_rounds": LGBM_EARLY_STOPPING,
    }
    at_boundary = _grid_boundary(
        cfg,
        leaf_choices=leaf_choices,
        n_estimators_used=best_n_est,
        n_estimators_max=LGBM_N_ESTIMATORS_MAX,
    )
    if require_interior and at_boundary:
        raise RuntimeError(
            "INVALID EXPERIMENT: LightGBM config sits on the grid boundary "
            f"(config={cfg}, leaf_choices={leaf_choices}). Extend the grid "
            "or raise n_estimators_max before reading G3."
        )
    stopped_early = int(best_n_est) < int(LGBM_N_ESTIMATORS_MAX)
    return {
        "accuracy": _accuracy(pred, split["yte"]),
        "auc": _auc(proba, split["yte"]),
        "config": cfg,
        "train_acc": float(train_acc),
        "val_acc": val_acc,
        "train_val_gap": float(train_acc - val_acc),
        "best_iteration": int(best_n_est),
        "stopped_early": bool(stopped_early),
        "at_grid_boundary": at_boundary,
    }


def _fit_arrangement_arms(
    split: dict[str, np.ndarray],
    *,
    seed: int,
    budgets: dict[str, dict[str, Any]],
    require_converged: bool,
) -> dict[str, Any]:
    """Run dense + sparse arms; keep the better validation BCE."""
    from omnibias.tab.torch.arrangement import fit_arrangement

    arms: list[dict[str, Any]] = []
    for arm_name, budget in budgets.items():
        sparse = arm_name == "sparse" or float(budget["l1"]) > 0.0
        result = fit_arrangement(
            split["Xtr"],
            split["ytr"],
            X_val=split["Xva"],
            y_val=split["yva"],
            n_hyperplanes=2,
            l1=float(budget["l1"]),
            restarts=int(budget["restarts"]),
            steps=int(budget["steps"]),
            beta_final=float(budget["beta_final"]),
            seed=seed,
            patience=int(budget.get("patience", ARRANGEMENT_PATIENCE)),
            min_delta=float(budget.get("min_delta", 1e-4)),
            sparse_warmstart=sparse,
        )
        pred = result.model.predict(split["Xte"])
        proba = result.model.predict_proba(split["Xte"])
        arms.append(
            {
                "arm": arm_name,
                "accuracy": _accuracy(pred, split["yte"]),
                "auc": _auc(proba, split["yte"]),
                "train_acc": result.train_acc,
                "val_acc": result.val_acc,
                "train_bce": result.train_bce,
                "val_bce": result.val_bce,
                "train_val_gap": result.train_val_gap,
                "steps_run": result.steps_run,
                "best_step": result.best_step,
                "stopped_early": result.stopped_early,
                "at_step_cap": result.at_step_cap,
                "beta_final": result.beta_final,
                "l1": result.l1,
                "sparse_warmstart": result.sparse_warmstart,
                "n_restarts": result.n_restarts,
                "budget": dict(budget),
            }
        )
    best = min(
        arms,
        key=lambda a: (float(a["val_bce"]), -float(a["val_acc"])),
    )
    if require_converged and bool(best["at_step_cap"]):
        raise RuntimeError(
            "INVALID EXPERIMENT: selected arrangement arm hit the Adam step "
            f"cap without early-stopping (arm={best['arm']}, "
            f"steps_run={best['steps_run']}, budget={best['budget']}). "
            "Raise steps or patience before reading G3."
        )
    return {**best, "arms": arms}


def _winner(arr_acc: float, lgbm_acc: float) -> str:
    margin = float(arr_acc) - float(lgbm_acc)
    if abs(margin) < TIE_BAND:
        return "tie"
    return "arrangement" if margin > 0.0 else "lightgbm"


def _predict_from_diag(diag: float) -> str:
    """Frozen G4 rule: dense-over-axis ratio above 1.0 -> arrangement."""
    return "arrangement" if float(diag) > G4_DIAG_THRESHOLD else "lightgbm"


def _pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2:
        return float("nan")
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    if np.std(x) < 1e-15 or np.std(y) < 1e-15:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _run_dataset(
    name: str,
    *,
    full: bool,
    grid: list[tuple[int, float, int, int]],
    budgets: dict[str, dict[str, Any]],
    max_rows: int | None,
) -> dict[str, Any]:
    from omnibias.tab.arrangement import obliqueness_diagnostic
    from omnibias.tab.bench import load_dataset, train_val_test_split

    t0 = time.perf_counter()
    try:
        ds = load_dataset(name, max_rows=max_rows, seed=0)
    except RuntimeError as exc:
        return {
            "dataset": name,
            "status": "skipped",
            "skip_reason": str(exc),
            "winner": None,
            "per_seed": [],
            "wall_seconds": round(time.perf_counter() - t0, 3),
        }
    if ds.task != "binary":
        return {
            "dataset": name,
            "status": "skipped",
            "skip_reason": f"expected binary, got {ds.task!r}",
            "winner": None,
            "per_seed": [],
            "wall_seconds": round(time.perf_counter() - t0, 3),
        }

    per_seed: list[dict[str, Any]] = []
    for seed in SEEDS:
        split = train_val_test_split(ds, seed=seed)
        Xprobe = np.vstack([split["Xtr"], split["Xva"]])
        yprobe = np.concatenate([split["ytr"], split["yva"]])
        diag = obliqueness_diagnostic(Xprobe, yprobe)
        lgbm = _fit_lightgbm(
            split, seed=seed, grid=grid, require_interior=full
        )
        arr = _fit_arrangement_arms(
            split, seed=seed, budgets=budgets, require_converged=full
        )
        margin = float(arr["accuracy"] - lgbm["accuracy"])
        majority = float(max(split["yte"].mean(), 1.0 - split["yte"].mean()))
        row = {
            "seed": int(seed),
            "arrangement_accuracy": float(arr["accuracy"]),
            "lightgbm_accuracy": float(lgbm["accuracy"]),
            "margin": margin,
            "abs_margin": abs(margin),
            "arrangement_auc": float(arr["auc"]),
            "lightgbm_auc": float(lgbm["auc"]),
            "majority_class_rate": majority,
            "obliqueness_diagnostic": float(diag),
            "predicted_winner": _predict_from_diag(diag),
            "lightgbm_config": lgbm["config"],
            "lightgbm_config_at_grid_boundary": bool(lgbm["at_grid_boundary"]),
            "lightgbm": {
                "train_acc": lgbm["train_acc"],
                "val_acc": lgbm["val_acc"],
                "train_val_gap": lgbm["train_val_gap"],
                "best_iteration": lgbm["best_iteration"],
                "stopped_early": lgbm["stopped_early"],
            },
            "arrangement": {
                "arm": arr["arm"],
                "train_acc": arr["train_acc"],
                "val_acc": arr["val_acc"],
                "train_bce": arr["train_bce"],
                "val_bce": arr["val_bce"],
                "train_val_gap": arr["train_val_gap"],
                "steps_run": arr["steps_run"],
                "best_step": arr["best_step"],
                "stopped_early": arr["stopped_early"],
                "at_step_cap": arr["at_step_cap"],
                "beta_final": arr["beta_final"],
                "l1": arr["l1"],
                "sparse_warmstart": arr["sparse_warmstart"],
                "n_restarts": arr["n_restarts"],
                "budget": arr["budget"],
                "arms": [
                    {
                        "arm": a["arm"],
                        "val_acc": a["val_acc"],
                        "val_bce": a["val_bce"],
                        "accuracy": a["accuracy"],
                        "l1": a["l1"],
                        "stopped_early": a["stopped_early"],
                        "at_step_cap": a["at_step_cap"],
                        "steps_run": a["steps_run"],
                        "best_step": a["best_step"],
                    }
                    for a in arr["arms"]
                ],
            },
        }
        per_seed.append(row)
        print(
            f"  {name} seed={seed}: arr={arr['accuracy']:.4f} "
            f"lgbm={lgbm['accuracy']:.4f} margin={margin:+.4f} "
            f"diag={diag:.3f} arm={arr['arm']} "
            f"arr_es={arr['stopped_early']} lgbm_es={lgbm['stopped_early']}"
        )

    mean_arr = float(np.mean([r["arrangement_accuracy"] for r in per_seed]))
    mean_lgbm = float(np.mean([r["lightgbm_accuracy"] for r in per_seed]))
    mean_margin = float(np.mean([r["margin"] for r in per_seed]))
    mean_diag = float(np.mean([r["obliqueness_diagnostic"] for r in per_seed]))
    winner = _winner(mean_arr, mean_lgbm)
    predicted = _predict_from_diag(mean_diag)
    return {
        "dataset": name,
        "status": "completed",
        "n_rows": int(ds.X.shape[0]),
        "n_features": int(ds.X.shape[1]),
        "max_rows": max_rows,
        "per_seed": per_seed,
        "mean_arrangement_accuracy": mean_arr,
        "mean_lightgbm_accuracy": mean_lgbm,
        "mean_margin": mean_margin,
        "mean_obliqueness_diagnostic": mean_diag,
        "winner": winner,
        "predicted_winner": predicted,
        "diagnostic_correct": predicted == winner,
        "wall_seconds": round(time.perf_counter() - t0, 3),
    }


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--full",
        action="store_true",
        help="acceptance tier: all eight datasets + extended LightGBM grid",
    )
    args = parser.parse_args(argv)
    full = bool(args.full)
    tier = "full" if full else "smoke"
    grid = LGBM_GRID_FULL if full else LGBM_GRID_SMOKE
    budgets = ARRANGEMENT_BUDGET[tier]
    artifact_name = (
        "tabular_arrangement_public.json"
        if full
        else "tabular_arrangement_public_smoke.json"
    )

    from omnibias.tab.bench import (
        ARRANGEMENT_PUBLIC_MAX_ROWS,
        ARRANGEMENT_PUBLIC_SUITE,
    )

    names = list(ARRANGEMENT_PUBLIC_SUITE) if full else ["breast_cancer"]
    config = {
        "family": "tabular_arrangement_public_vs_lightgbm",
        "suite": list(ARRANGEMENT_PUBLIC_SUITE),
        "datasets_run": names,
        "seeds": list(SEEDS),
        "split": {"train": 0.6, "val": 0.2, "test": 0.2},
        "tie_band_accuracy_points": TIE_BAND,
        "n_hyperplanes": 2,
        "lightgbm_grid_size": len(grid),
        "lightgbm_n_estimators_max": LGBM_N_ESTIMATORS_MAX,
        "lightgbm_early_stopping_rounds": LGBM_EARLY_STOPPING,
        "arrangement_patience": ARRANGEMENT_PATIENCE,
        "arrangement_budget": budgets,
        "fair_protocol": {
            "train_on": "Xtr",
            "early_stop_on": "Xva",
            "score_on": "Xte",
            "no_train_val_refit": True,
            "arrangement_stop_metric": "val_bce",
            "lightgbm_stop_metric": "binary_logloss",
        },
        "g4_diag_threshold": G4_DIAG_THRESHOLD,
        "g4_min_accuracy": G4_MIN_ACCURACY,
        "tier": tier,
        "smoke_is_wiring_gate": not full,
        "full": full,
    }
    payload = provenance(schema="tabular-arrangement-public-v1", config=config)

    t0 = time.perf_counter()
    blocks: dict[str, Any] = {}
    for name in names:
        print(f"running {name}...")
        max_rows = ARRANGEMENT_PUBLIC_MAX_ROWS.get(name)
        if not full and name == "breast_cancer":
            max_rows = 400  # keep smoke cheap
        blocks[name] = _run_dataset(
            name,
            full=full,
            grid=grid,
            budgets=budgets,
            max_rows=max_rows,
        )

    completed = [b for b in blocks.values() if b["status"] == "completed"]
    skipped = [b for b in blocks.values() if b["status"] == "skipped"]
    if full and len(completed) < 8:
        raise RuntimeError(
            "INVALID EXPERIMENT: need >=8 completed public binary datasets "
            f"for G3; got {len(completed)} completed, {len(skipped)} skipped "
            f"({[s['dataset'] for s in skipped]})"
        )

    win_loss_table = [
        {
            "dataset": b["dataset"],
            "status": b["status"],
            "winner": b.get("winner"),
            "mean_arrangement_accuracy": b.get("mean_arrangement_accuracy"),
            "mean_lightgbm_accuracy": b.get("mean_lightgbm_accuracy"),
            "mean_margin": b.get("mean_margin"),
            "mean_obliqueness_diagnostic": b.get("mean_obliqueness_diagnostic"),
            "predicted_winner": b.get("predicted_winner"),
            "diagnostic_correct": b.get("diagnostic_correct"),
            "skip_reason": b.get("skip_reason"),
        }
        for b in (blocks[n] for n in names)
    ]
    n_arr = sum(1 for b in completed if b["winner"] == "arrangement")
    n_lgbm = sum(1 for b in completed if b["winner"] == "lightgbm")
    n_tie = sum(1 for b in completed if b["winner"] == "tie")

    diags = [float(b["mean_obliqueness_diagnostic"]) for b in completed]
    margins = [float(b["mean_margin"]) for b in completed]
    diag_correct = [bool(b["diagnostic_correct"]) for b in completed]
    g4_acc = float(np.mean(diag_correct)) if diag_correct else float("nan")
    corr = _pearson(diags, margins)
    g3_earned = full and len(completed) >= 8
    g4_earned = bool(
        full and len(completed) >= 8 and g4_acc >= G4_MIN_ACCURACY
    )

    licensed = (
        f"on {len(completed)} public binary datasets the H=2 soft hyperplane "
        f"arrangement vs tuned LightGBM win/loss/tie is "
        f"{n_arr}/{n_lgbm}/{n_tie} "
        f"(per-dataset table in win_loss_table; trees expected to win most)"
        if g3_earned
        else (
            "smoke wiring only: breast_cancer exercise of the public "
            "arrangement vs LightGBM path; G3 not earned"
            if not full
            else "G3 not earned: fewer than eight completed public datasets"
        )
    )

    payload.update(
        {
            "baseline": {
                "name": "tuned LightGBM (val-selected grid + early stopping)",
                "note": (
                    "num_leaves x learning_rate x min_child_samples x max_depth; "
                    "n_estimators via early stopping up to "
                    f"{LGBM_N_ESTIMATORS_MAX}; train on Xtr only (no train+val "
                    "refit); arrangement early-stops on val BCE with the same "
                    "patience and no train+val refit"
                ),
            },
            "seeds": list(SEEDS),
            "datasets": blocks,
            "win_loss_table": win_loss_table,
            "win_loss_counts": {
                "arrangement": n_arr,
                "lightgbm": n_lgbm,
                "tie": n_tie,
                "completed": len(completed),
                "skipped": len(skipped),
            },
            "g4": {
                "diag_threshold": G4_DIAG_THRESHOLD,
                "rule": "predict arrangement if mean_obliqueness_diagnostic > threshold else lightgbm",
                "predictiveness": g4_acc,
                "corr_diag_margin": corr,
                "n_scored": len(completed),
                "retuned": False,
            },
            "honesty": {
                "claim_rung": 1,
                "trees_expected_to_win_most": True,
                "no_aggregate_only_headline": True,
                "temperature_collapse": True,
                "bias_collapse": False,
                "smoke_is_wiring_gate": not full,
                "obliqueness_diagnostic_retuned": False,
                "g3_earned": g3_earned,
                "g4_earned": g4_earned,
                "theorem_prover_verified": False,
                "mathlib_verified": False,
                "licensed_sentence": licensed,
            },
            "g3_earned": g3_earned,
            "g4_earned": g4_earned,
            "wall_seconds": round(time.perf_counter() - t0, 3),
        }
    )

    out = write_json(artifact_name, payload)
    print(
        f"wrote {out}  completed={len(completed)} "
        f"W/L/T={n_arr}/{n_lgbm}/{n_tie} "
        f"g3={g3_earned} g4={g4_earned} "
        f"g4_acc={g4_acc:.3f} corr={corr}"
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
