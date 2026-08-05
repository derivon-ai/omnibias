# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Sweep register x arm x width x seed x label-noise, and aggregate the runs.

:func:`run_sweep` trains every requested combination (sharing one frozen MNIST-1D
feature set; only the label-noise mask and the model init vary with ``seed``) and
writes a per-run JSON to the scratch directory. :func:`collect_runs` reads those
JSONs back into flat summary rows and :func:`write_summary` reduces them to a
per-run CSV plus a ``(register, arm, width, noise)`` aggregate (mean +/- std test
error, mean curvature) -- the tables the plots and the double-descent / optimizer
-axis analysis consume.
"""

from __future__ import annotations

import csv
import json
import math
import statistics
from dataclasses import asdict
from pathlib import Path

from examples.mnist1d_double_descent.arms import OptimizerArm, arms_for_register, get_arm
from examples.mnist1d_double_descent.data import (
    DataBundle,
    Mnist1DConfig,
    load_mnist1d,
    synthetic_mnist1d,
)
from examples.mnist1d_double_descent.models import MLP1D, REGISTERS, Register, count_parameters
from examples.mnist1d_double_descent.train import RunConfig, RunResult, save_run, train_run

#: The default double-descent width grid (dense near the ~50-unit threshold).
DEFAULT_WIDTHS: tuple[int, ...] = (
    1, 2, 3, 5, 8, 13, 20, 30, 40, 50, 60, 75, 100, 150, 250, 500, 1000,
)


def _effective_lr(arm: OptimizerArm, lr_override: float | None) -> float:
    return arm.lr if lr_override is None else lr_override


def _approx_n_params(width: int, *, in_dim: int, num_classes: int, depth: int) -> int:
    model = MLP1D(in_dim=in_dim, hidden=width, num_classes=num_classes, depth=depth)
    return count_parameters(model)


def make_bundle(
    *,
    label_noise: float,
    seed: int,
    synthetic: bool,
    cfg: Mnist1DConfig | None,
    scratch_dir: str | Path | None,
    allow_pip: bool,
) -> DataBundle:
    """Build one (noise, seed) MNIST-1D bundle (features frozen; noise mask per seed)."""
    if synthetic:
        return synthetic_mnist1d(label_noise=label_noise, seed=seed)
    return load_mnist1d(
        cfg, label_noise=label_noise, noise_seed=seed,
        scratch_dir=scratch_dir, allow_pip=allow_pip,
    )


def run_sweep(
    *,
    registers: tuple[Register, ...] = REGISTERS,
    arm_names: tuple[str, ...],
    widths: tuple[int, ...] = DEFAULT_WIDTHS,
    seeds: tuple[int, ...] = (0, 1, 2),
    noise_levels: tuple[float, ...] = (0.0, 0.15),
    steps: int = 400,
    depth: int = 1,
    lr_override: float | None = None,
    batch_size: int | None = None,
    log_every: int = 1,
    device: str = "cpu",
    scratch_dir: str | Path = "runs",
    dense_max_params: int = 1500,
    curvature: bool = True,
    curvature_every: int = 0,
    curv_batch: int = 0,
    curv_power_iters: int = 40,
    curv_hutch: int = 8,
    synthetic: bool = False,
    cfg: Mnist1DConfig | None = None,
    allow_pip: bool = True,
    log: bool = False,
) -> list[RunResult]:
    """Train all requested runs, writing each to ``scratch_dir``; returns the results."""
    scratch = Path(scratch_dir)
    results: list[RunResult] = []
    for register in registers:
        reg_arms = arms_for_register(register, arm_names)
        for noise in noise_levels:
            for seed in seeds:
                bundle = make_bundle(
                    label_noise=noise, seed=seed, synthetic=synthetic,
                    cfg=cfg, scratch_dir=scratch, allow_pip=allow_pip,
                )
                for width in widths:
                    n_params = _approx_n_params(
                        width, in_dim=bundle.in_dim, num_classes=bundle.num_classes, depth=depth
                    )
                    for arm_name in reg_arms:
                        arm = get_arm(arm_name)
                        if arm.dense_only and n_params > dense_max_params:
                            continue
                        run_cfg = RunConfig(
                            register=register, arm=arm_name, width=width, depth=depth,
                            seed=seed, label_noise=noise, steps=steps,
                            lr=_effective_lr(arm, lr_override), batch_size=batch_size,
                            log_every=log_every,
                            curvature=curvature, curvature_every=curvature_every,
                            curv_batch=curv_batch, curv_power_iters=curv_power_iters,
                            curv_hutch=curv_hutch, dense_max_params=dense_max_params,
                            device=device, source=bundle.source,
                        )
                        result = _safe_train(bundle, arm, run_cfg, log=log)
                        save_run(result, scratch)
                        results.append(result)
    return results


def _safe_train(bundle: DataBundle, arm: OptimizerArm, cfg: RunConfig, *, log: bool) -> RunResult:
    try:
        return train_run(bundle, arm, cfg, log=log)
    except Exception as exc:  # noqa: BLE001 -- one failed config must not sink the sweep
        return RunResult(
            config={**asdict(cfg), "data_source": bundle.source},
            n_params=0,
            final_train_loss=math.nan,
            final_train_err=math.nan,
            final_test_err=math.nan,
            best_test_err=math.nan,
            interpolation_step=-1,
            err_at_interpolation=-1.0,
            wall_time=0.0,
            status="error",
            error=f"{type(exc).__name__}: {exc}",
        )


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

_ROW_FIELDS = [
    "register", "arm", "width", "depth", "seed", "label_noise", "n_params",
    "data_source", "status", "final_test_err", "best_test_err", "final_train_err",
    "final_train_loss", "interpolation_step", "err_at_interpolation",
    "lambda_max_final", "lambda_min_final", "cond_final", "trace_final",
    "lambda_max_interp", "cond_interp", "wall_time", "opt_state_bytes", "update_time_s",
]


def _curv(d: dict[str, object] | None, key: str) -> float:
    if not isinstance(d, dict) or key not in d:
        return math.nan
    return float(d[key])  # type: ignore[arg-type]


def summary_row(result: dict[str, object]) -> dict[str, object]:
    """Flatten one parsed run dict into a single summary row."""
    cfg = result["config"]
    assert isinstance(cfg, dict)
    cf = result.get("curvature_final")
    ci = result.get("curvature_at_interpolation")
    cf = cf if isinstance(cf, dict) else None
    ci = ci if isinstance(ci, dict) else None
    return {
        "register": cfg["register"],
        "arm": cfg["arm"],
        "width": cfg["width"],
        "depth": cfg["depth"],
        "seed": cfg["seed"],
        "label_noise": cfg["label_noise"],
        "n_params": result["n_params"],
        "data_source": cfg.get("data_source", ""),
        "status": result.get("status", "ok"),
        "final_test_err": result["final_test_err"],
        "best_test_err": result["best_test_err"],
        "final_train_err": result["final_train_err"],
        "final_train_loss": result["final_train_loss"],
        "interpolation_step": result["interpolation_step"],
        "err_at_interpolation": result["err_at_interpolation"],
        "lambda_max_final": _curv(cf, "lambda_max"),
        "lambda_min_final": _curv(cf, "lambda_min"),
        "cond_final": _curv(cf, "condition_number"),
        "trace_final": _curv(cf, "trace"),
        "lambda_max_interp": _curv(ci, "lambda_max"),
        "cond_interp": _curv(ci, "condition_number"),
        "wall_time": result["wall_time"],
        "opt_state_bytes": result.get("opt_state_bytes", 0),
        "update_time_s": result.get("update_time_s", 0.0),
    }


def collect_runs(scratch_dir: str | Path) -> list[dict[str, object]]:
    """Read every ``*.json`` run under ``scratch_dir`` (recursively) into flat summary rows.

    Recursion lets one call aggregate both a flat local sweep and the cluster layout, where
    :mod:`~examples.mnist1d_double_descent.sweep.gen_jobs` writes each job into its own
    ``<scratch-base>/<tag>/`` subdirectory. Non-run JSONs (those lacking a ``config`` key)
    are skipped, so a stray summary or data-cache file in the tree is ignored.
    """
    rows: list[dict[str, object]] = []
    for path in sorted(Path(scratch_dir).rglob("*.json")):
        with path.open(encoding="utf-8") as fh:
            try:
                data = json.load(fh)
            except json.JSONDecodeError:
                continue
        if isinstance(data, dict) and "config" in data:
            rows.append(summary_row(data))
    return rows


def _mean_std(values: list[float]) -> tuple[float, float]:
    finite = [v for v in values if math.isfinite(v)]
    if not finite:
        return math.nan, math.nan
    mean = statistics.fmean(finite)
    std = statistics.pstdev(finite) if len(finite) > 1 else 0.0
    return mean, std


def aggregate_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Aggregate per-run rows to ``(register, arm, width, label_noise)`` mean +/- std."""
    grouped: dict[tuple[object, object, object, object], list[dict[str, object]]] = {}
    for r in rows:
        if r.get("status") != "ok":
            continue
        key = (r["register"], r["arm"], r["width"], r["label_noise"])
        grouped.setdefault(key, []).append(r)
    out: list[dict[str, object]] = []
    for (register, arm, width, noise), group in sorted(
        grouped.items(), key=lambda kv: (str(kv[0][0]), str(kv[0][1]), float(kv[0][2]), float(kv[0][3]))
    ):
        test_mean, test_std = _mean_std([float(g["final_test_err"]) for g in group])
        best_mean, best_std = _mean_std([float(g["best_test_err"]) for g in group])
        lam_mean, _ = _mean_std([float(g["lambda_max_final"]) for g in group])
        cond_mean, _ = _mean_std([float(g["cond_final"]) for g in group])
        interp_mean, _ = _mean_std([float(g["interpolation_step"]) for g in group])
        upd_mean, _ = _mean_std([float(g.get("update_time_s", 0.0)) for g in group])
        state_bytes = int(group[0].get("opt_state_bytes", 0))  # constant across seeds
        out.append({
            "register": register, "arm": arm, "width": width, "label_noise": noise,
            "n_params": group[0]["n_params"], "n_runs": len(group),
            "test_err_mean": test_mean, "test_err_std": test_std,
            "best_err_mean": best_mean, "best_err_std": best_std,
            "lambda_max_mean": lam_mean, "cond_mean": cond_mean,
            "interpolation_step_mean": interp_mean,
            "opt_state_bytes": state_bytes, "update_time_s_mean": upd_mean,
        })
    return out


def write_summary(scratch_dir: str | Path, out_dir: str | Path) -> tuple[Path, Path, Path]:
    """Read runs from ``scratch_dir`` and write ``runs.csv`` + ``summary.csv`` + ``summary.json``."""
    rows = collect_runs(scratch_dir)
    agg = aggregate_rows(rows)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    runs_csv = out / "runs.csv"
    summary_csv = out / "summary.csv"
    summary_json = out / "summary.json"
    with runs_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=_ROW_FIELDS)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in _ROW_FIELDS})
    agg_fields = list(agg[0].keys()) if agg else ["register", "arm", "width", "label_noise"]
    with summary_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=agg_fields)
        writer.writeheader()
        for r in agg:
            writer.writerow(r)
    with summary_json.open("w", encoding="utf-8") as fh:
        json.dump(agg, fh, indent=2)
    return runs_csv, summary_csv, summary_json


__all__ = [
    "DEFAULT_WIDTHS",
    "aggregate_rows",
    "collect_runs",
    "make_bundle",
    "run_sweep",
    "summary_row",
    "write_summary",
]
