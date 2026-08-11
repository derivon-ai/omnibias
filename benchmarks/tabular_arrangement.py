# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Wave-0 falsifier A4: tabular arrangement vs tuned LightGBM (05-02 G1/G2).

Constructed datasets (D=10, n=10_000, binary labels):

* Oblique XOR (G1): ``y = 1[w1.x > 0] XOR 1[w2.x > 0]``. Constructed to favour
  the arrangement; failure kills the tabular sub-application.
* Axis AND (G2): ``y = 1[x_3 > 0.5] AND 1[x_7 < 0.2]``. Arrangement must reach
  parity (within 2 accuracy points) of tuned LightGBM.

Gates (five seeds, worst-seed via ``require_all_seeds``):

* G1: ``arrangement_acc - lightgbm_acc >= 0.10`` on every seed.
* G2: ``|arrangement_acc - lightgbm_acc| <= 0.02`` on every seed.

G3/G4 (public benchmarks / diagnostic predictiveness) remain unearned.
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
from _gates import (  # type: ignore[import-not-found]  # noqa: E402
    gates_block,
    require_all_seeds,
)

SCRATCH = Path(os.environ.get("OMNIBIAS_SCRATCH", "artifacts"))

SEEDS = (0, 1, 2, 3, 4)
N_SAMPLES = 10_000
N_FEATURES = 10
N_TRAIN = 6_000
N_VAL = 2_000
# N_TEST = 2_000
G1_MIN_MARGIN = 0.10
G2_MAX_ABS_MARGIN = 0.02

LGBM_GRID_FULL = list(
    itertools.product([31, 127, 511], [0.05, 0.1], [200, 600])
)
LGBM_GRID_SMOKE = list(itertools.product([31, 127], [0.05, 0.1], [200]))


def _split(X: np.ndarray, y: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "Xtr": X[:N_TRAIN],
        "ytr": y[:N_TRAIN],
        "Xva": X[N_TRAIN : N_TRAIN + N_VAL],
        "yva": y[N_TRAIN : N_TRAIN + N_VAL],
        "Xte": X[N_TRAIN + N_VAL :],
        "yte": y[N_TRAIN + N_VAL :],
    }


def _accuracy(pred: np.ndarray, y: np.ndarray) -> float:
    return float((np.asarray(pred).reshape(-1) == np.asarray(y).reshape(-1)).mean())


def _auc(scores: np.ndarray, y: np.ndarray) -> float:
    """Mann-Whitney AUC for binary labels in {0,1}."""
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    yy = np.asarray(y, dtype=np.float64).reshape(-1)
    pos = s[yy > 0.5]
    neg = s[yy <= 0.5]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    # P(score_pos > score_neg) + 0.5 P(tie)
    # vectorised via ranks
    order = np.argsort(s)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, s.size + 1, dtype=np.float64)
    # average ties
    # simple version without tie correction is fine for our constructed sets
    sum_pos = float(ranks[yy > 0.5].sum())
    n_pos = float(pos.size)
    n_neg = float(neg.size)
    return (sum_pos - n_pos * (n_pos + 1.0) / 2.0) / (n_pos * n_neg)


def _fit_lightgbm(
    split: dict[str, np.ndarray],
    *,
    seed: int,
    grid: list[tuple[int, float, int]],
) -> dict[str, Any]:
    import lightgbm as lgb

    best_cfg: tuple[int, float, int] | None = None
    best_val = -1.0
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="X does not have valid feature names")
        warnings.filterwarnings("ignore", category=UserWarning)
        for leaves, lr, n_est in grid:
            model = lgb.LGBMClassifier(
                num_leaves=leaves,
                learning_rate=lr,
                n_estimators=n_est,
                random_state=seed,
                verbose=-1,
                n_jobs=1,
            )
            model.fit(split["Xtr"], split["ytr"].astype(np.int64))
            pred = model.predict(split["Xva"]).astype(np.float64)
            acc = _accuracy(pred, split["yva"])
            if acc > best_val:
                best_val = acc
                best_cfg = (leaves, lr, n_est)
        assert best_cfg is not None
        leaves, lr, n_est = best_cfg
        model = lgb.LGBMClassifier(
            num_leaves=leaves,
            learning_rate=lr,
            n_estimators=n_est,
            random_state=seed,
            verbose=-1,
            n_jobs=1,
        )
        Xfit = np.vstack([split["Xtr"], split["Xva"]])
        yfit = np.concatenate([split["ytr"], split["yva"]])
        model.fit(Xfit, yfit.astype(np.int64))
        pred = model.predict(split["Xte"]).astype(np.float64)
        proba = model.predict_proba(split["Xte"])[:, 1]
    return {
        "accuracy": _accuracy(pred, split["yte"]),
        "auc": _auc(proba, split["yte"]),
        "config": {
            "num_leaves": leaves,
            "learning_rate": lr,
            "n_estimators": n_est,
        },
        "val_accuracy": best_val,
    }


def _fit_arrangement(
    split: dict[str, np.ndarray],
    *,
    seed: int,
    family: str,
    full: bool,
) -> dict[str, Any]:
    from omnibias.tab.torch.arrangement import fit_arrangement

    Xfit = np.vstack([split["Xtr"], split["Xva"]])
    yfit = np.concatenate([split["ytr"], split["yva"]])
    if family == "oblique_xor":
        result = fit_arrangement(
            Xfit,
            yfit,
            n_hyperplanes=2,
            l1=0.0,
            restarts=8 if full else 4,
            steps=600 if full else 350,
            beta_final=64.0,
            seed=seed,
            val_fraction=0.2,
            sparse_warmstart=False,
        )
    else:
        result = fit_arrangement(
            Xfit,
            yfit,
            n_hyperplanes=2,
            l1=0.02,
            restarts=4 if full else 2,
            steps=400 if full else 250,
            beta_final=128.0,
            seed=seed,
            val_fraction=0.2,
            sparse_warmstart=True,
        )
    pred = result.model.predict(split["Xte"])
    proba = result.model.predict_proba(split["Xte"])
    state = result.model.numpy_state()
    return {
        "accuracy": _accuracy(pred, split["yte"]),
        "auc": _auc(proba, split["yte"]),
        "train_acc": result.train_acc,
        "val_acc": result.val_acc,
        "beta_final": result.beta_final,
        "l1": result.l1,
        "sparse_warmstart": result.sparse_warmstart,
        "n_restarts": result.n_restarts,
        "W_l1": float(np.abs(state["W"]).sum()),
        "W_nnz_features": int(np.sum(np.max(np.abs(state["W"]), axis=0) > 1e-2)),
    }


def _run_family(
    *,
    family: str,
    full: bool,
    grid: list[tuple[int, float, int]],
) -> dict[str, Any]:
    from omnibias.tab.arrangement import (
        make_axis_rule,
        make_oblique_xor,
        obliqueness_diagnostic,
    )

    maker = make_oblique_xor if family == "oblique_xor" else make_axis_rule
    per_seed: list[dict[str, Any]] = []
    for seed in SEEDS:
        X, y, meta = maker(n_samples=N_SAMPLES, n_features=N_FEATURES, seed=seed)
        split = _split(X, y)
        diag = obliqueness_diagnostic(
            np.vstack([split["Xtr"], split["Xva"]]),
            np.concatenate([split["ytr"], split["yva"]]),
        )
        lgbm = _fit_lightgbm(split, seed=seed, grid=grid)
        arr = _fit_arrangement(split, seed=seed, family=family, full=full)
        margin = float(arr["accuracy"] - lgbm["accuracy"])
        majority = float(
            max(split["yte"].mean(), 1.0 - split["yte"].mean())
        )
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
            "lightgbm_config": lgbm["config"],
            "arrangement": {
                k: arr[k]
                for k in (
                    "train_acc",
                    "val_acc",
                    "beta_final",
                    "l1",
                    "sparse_warmstart",
                    "n_restarts",
                    "W_l1",
                    "W_nnz_features",
                )
            },
            "meta": {
                k: meta[k]
                for k in meta
                if k not in ("w1", "w2")  # keep artifact small
            },
        }
        per_seed.append(row)
        print(
            f"  {family} seed={seed}: arr={arr['accuracy']:.4f} "
            f"lgbm={lgbm['accuracy']:.4f} margin={margin:+.4f} "
            f"diag={diag:.3f}"
        )
    return {
        "family": family,
        "per_seed": per_seed,
        "constructed_to_favour_arrangement": family == "oblique_xor",
    }


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--full",
        action="store_true",
        help="full LightGBM grid + more arrangement restarts/steps",
    )
    args = parser.parse_args(argv)
    full = bool(args.full)
    grid = LGBM_GRID_FULL if full else LGBM_GRID_SMOKE
    artifact_name = (
        "tabular_arrangement.json" if full else "tabular_arrangement_smoke.json"
    )

    config = {
        "family": "tabular_arrangement_vs_lightgbm",
        "n_samples": N_SAMPLES,
        "n_features": N_FEATURES,
        "n_train": N_TRAIN,
        "n_val": N_VAL,
        "n_test": N_SAMPLES - N_TRAIN - N_VAL,
        "seeds": list(SEEDS),
        "g1_min_margin": G1_MIN_MARGIN,
        "g2_max_abs_margin": G2_MAX_ABS_MARGIN,
        "lightgbm_grid_size": len(grid),
        "full": full,
        "g3_earned": False,
        "g4_earned": False,
    }
    payload = provenance(schema="tabular-arrangement-v1", config=config)

    t0 = time.perf_counter()
    print("running oblique_xor (G1)...")
    oblique = _run_family(family="oblique_xor", full=full, grid=grid)
    print("running axis_and (G2)...")
    axis = _run_family(family="axis_and", full=full, grid=grid)

    entries = [
        require_all_seeds(
            oblique["per_seed"],
            key="margin",
            expected=G1_MIN_MARGIN,
            tol=0.0,
            direction="min",
            name="g1_oblique_margin",
        ),
        require_all_seeds(
            axis["per_seed"],
            key="abs_margin",
            expected=0.0,
            tol=G2_MAX_ABS_MARGIN,
            name="g2_axis_abs_margin",
        ),
    ]
    gates = dict(gates_block(entries))

    payload.update(
        {
            "baseline": {
                "name": "tuned LightGBM (val-selected grid)",
                "note": (
                    "num_leaves x learning_rate x n_estimators grid; "
                    "refit on train+val with the chosen config"
                ),
            },
            "seeds": list(SEEDS),
            "oblique_xor": oblique,
            "axis_and": axis,
            "gates": gates,
            "honesty": {
                "claim_rung": 1,
                "constructed_to_favour_arrangement": True,
                "temperature_collapse": True,
                "bias_collapse": False,
                "g1_earned": bool(entries[0]["passed"]),
                "g2_earned": bool(entries[1]["passed"]),
                "g3_earned": False,
                "g4_earned": False,
                "g5_earned": False,
                "g6_earned": False,
                "g7_earned": False,
                "tabular_subapplication_alive": bool(entries[0]["passed"]),
                "theorem_prover_verified": False,
                "mathlib_verified": False,
                "licensed_sentence": (
                    "on the constructed oblique XOR dataset the H=2 soft "
                    "hyperplane arrangement beats tuned LightGBM by at least "
                    "10 accuracy points over five seeds; on the constructed "
                    "axis AND dataset it stays within 2 accuracy points"
                    if gates["all_passed"]
                    else (
                        "G1 failed: the tabular arrangement sub-application "
                        "is dead on the constructed-to-win oblique case"
                        if not entries[0]["passed"]
                        else (
                            "G1 passed but G2 failed: optimization / L1 path "
                            "needs work; tabular sub-application stays alive"
                        )
                    )
                ),
            },
            "wall_seconds": round(time.perf_counter() - t0, 3),
        }
    )

    out = write_json(artifact_name, payload)
    print(
        f"wrote {out}  all_passed={gates['all_passed']}  "
        f"g1={entries[0]['passed']} g2={entries[1]['passed']}"
    )
    if full:
        scratch_dir = SCRATCH / "beyond_pde"
        scratch_dir.mkdir(parents=True, exist_ok=True)
        scratch_path = scratch_dir / artifact_name
        scratch_path.write_text(out.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"copied to {scratch_path}")
    if not gates["all_passed"]:
        raise SystemExit(1)
    return dict(payload)


if __name__ == "__main__":
    main()
