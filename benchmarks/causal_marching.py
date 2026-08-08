# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Causal marching vs whole-interval / causal-only / marching-only on 1-D heat.

Manufactured solution ``u = sin(pi x) exp(-pi^2 t)`` for ``u_t = u_xx`` on
``[0,1] x [0,1]``. Arms share an equal step budget.

Modes
-----
* ``--smoke`` (default): 1 seed, tiny nets / steps — CI wiring gate.
* ``--full``: >=5 seeds, larger budget — acceptance artifact.

Decision rule (smoke): every arm finishes with finite reference MSE and the
combined arm reports a causality index. Full mode additionally records
distributions across seeds (reference / seam / trivial rates / timing).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(__file__))
from _common import provenance, write_json  # noqa: E402

from omnibias.pinn._core.coords import CoordinateSpec
from omnibias.pinn._core.marching import TimeWindowSchedule
from omnibias.pinn.train.torch import march_solve

SCRATCH = Path(os.environ.get("OMNIBIAS_SCRATCH", "artifacts"))


class _MLP(nn.Module):
    def __init__(self, width: int = 32) -> None:
        super().__init__()
        dt = torch.float64
        self.net = nn.Sequential(
            nn.Linear(2, width, dtype=dt),
            nn.Tanh(),
            nn.Linear(width, width, dtype=dt),
            nn.Tanh(),
            nn.Linear(width, 1, dtype=dt),
        )

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        return self.net(coords)[:, 0]


def _exact(coords: torch.Tensor) -> torch.Tensor:
    return torch.sin(np.pi * coords[:, 0]) * torch.exp(-(np.pi**2) * coords[:, 1])


def _heat_residual(fld: nn.Module, coords: torch.Tensor) -> torch.Tensor:
    """Autograd heat residual u_t - u_xx (manufactured source is zero)."""
    coords = coords.detach().requires_grad_(True)
    u = fld(coords)
    ones = torch.ones_like(u)
    grads = torch.autograd.grad(u, coords, grad_outputs=ones, create_graph=True)[0]
    u_x = grads[:, 0]
    u_t = grads[:, 1]
    u_xx = torch.autograd.grad(u_x, coords, grad_outputs=ones, create_graph=True)[0][
        :, 0
    ]
    return u_t - u_xx


@dataclass(frozen=True)
class ArmConfig:
    name: str
    n_windows: int
    epsilon: float
    steps_per_window: int
    n_time_bins: int


def _eval_reference_mse(field: nn.Module, n: int = 64) -> float:
    xs = torch.linspace(0.0, 1.0, n, dtype=torch.float64)
    ts = torch.linspace(0.0, 1.0, n, dtype=torch.float64)
    xx, tt = torch.meshgrid(xs, ts, indexing="ij")
    coords = torch.stack([xx.reshape(-1), tt.reshape(-1)], dim=-1)
    with torch.no_grad():
        err = (field(coords) - _exact(coords)) ** 2
    return float(err.mean())


def _run_arm(
    arm: ArmConfig,
    *,
    seed: int,
    width: int,
    n_slice: int,
    per_bin: int,
    device: torch.device,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    field = _MLP(width=width).to(device)
    cs = CoordinateSpec(
        ("x", "t"), domain=((0.0, 1.0), (0.0, 1.0)), time_axis="t"
    )
    schedule = TimeWindowSchedule(
        0.0,
        1.0,
        n_windows=arm.n_windows,
        n_time_bins=arm.n_time_bins,
        epsilon=arm.epsilon,
        tolerance=1e-3,
    )
    xs = np.linspace(0.0, 1.0, n_slice, endpoint=False)
    ic = np.sin(np.pi * xs)

    def residual_fn(fld: nn.Module, coords: torch.Tensor) -> torch.Tensor:
        return _heat_residual(fld, coords.to(device))

    def value_fn(fld: nn.Module, coords: torch.Tensor) -> torch.Tensor:
        return fld(coords.to(device))

    t0 = time.perf_counter()
    result = march_solve(
        field,
        residual_fn,
        cs,
        schedule,
        steps_per_window=arm.steps_per_window,
        max_steps_per_window=arm.steps_per_window,
        lr=1e-2,
        per_bin=per_bin,
        n_slice=n_slice,
        ic_values=ic,
        value_fn=value_fn,
        seed=seed,
        dtype=torch.float64,
        check_trivial=True,
        trivial_mode="variance",
        advance_policy="force",
    )
    elapsed = time.perf_counter() - t0
    ref_mse = _eval_reference_mse(field.cpu())
    seam = [
        float(w.seam_mse) if w.seam_mse is not None else float("nan")
        for w in result.windows
    ]
    return {
        "arm": arm.name,
        "seed": seed,
        "n_windows": len(result.windows),
        "all_converged": result.all_converged,
        "reference_mse": ref_mse,
        "mean_seam_mse": float(np.nanmean(seam)) if seam else float("nan"),
        "last_causality_index": result.windows[-1].causality.causality_index,
        "unlocked_fraction": result.windows[-1].causality.unlocked_fraction,
        "trivial": bool(result.trivial.is_trivial) if result.trivial else None,
        "elapsed_seconds": elapsed,
        "total_steps": sum(w.steps_run for w in result.windows),
    }


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_arm: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_arm.setdefault(str(row["arm"]), []).append(row)
    out: dict[str, Any] = {}
    for name, group in by_arm.items():
        ref = np.asarray([g["reference_mse"] for g in group], dtype=float)
        seam = np.asarray([g["mean_seam_mse"] for g in group], dtype=float)
        times = np.asarray([g["elapsed_seconds"] for g in group], dtype=float)
        trivial = np.asarray([bool(g["trivial"]) for g in group], dtype=float)
        out[name] = {
            "n_seeds": len(group),
            "reference_mse_median": float(np.median(ref)),
            "reference_mse_mean": float(np.mean(ref)),
            "seam_mse_median": float(np.median(seam)),
            "trivial_rate": float(np.mean(trivial)),
            "elapsed_seconds_median": float(np.median(times)),
            "converged_rate": float(
                np.mean([bool(g["all_converged"]) for g in group])
            ),
        }
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--full",
        action="store_true",
        help="multi-seed acceptance run (default is --smoke)",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="torch device (cpu or cuda)",
    )
    args = parser.parse_args(argv)
    smoke = not args.full
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")

    # Equal total step budget across arms.
    if smoke:
        seeds = [0]
        width, n_slice, per_bin, bins = 16, 16, 4, 4
        budget = 40
    else:
        seeds = list(range(5))
        width, n_slice, per_bin, bins = 32, 32, 8, 8
        budget = 200

    arms = [
        ArmConfig("whole_interval", 1, 0.0, budget, bins),
        ArmConfig("causal_only", 1, 1.0, budget, bins),
        ArmConfig("marching_only", 4, 0.0, budget // 4, bins),
        ArmConfig("causal_marching", 4, 1.0, budget // 4, bins),
    ]

    rows: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    for arm in arms:
        for seed in seeds:
            rows.append(
                _run_arm(
                    arm,
                    seed=seed,
                    width=width,
                    n_slice=n_slice,
                    per_bin=per_bin,
                    device=device,
                )
            )

    summary = _summarize(rows)
    payload = provenance(
        schema="causal_marching/v2",
        config={
            "mode": "smoke" if smoke else "full",
            "seeds": seeds,
            "width": width,
            "n_slice": n_slice,
            "per_bin": per_bin,
            "n_time_bins": bins,
            "equal_step_budget": budget,
            "device": str(device),
            "pde": "u_t = u_xx",
            "solution": "sin(pi x) exp(-pi^2 t)",
        },
    )
    payload.update(
        {
            "runs": rows,
            "summary": summary,
            "elapsed_seconds": time.perf_counter() - t0,
        }
    )

    for row in rows:
        assert np.isfinite(row["reference_mse"]), row
        assert np.isfinite(row["elapsed_seconds"]), row
    assert "causal_marching" in summary

    SCRATCH.mkdir(parents=True, exist_ok=True)
    scratch_path = SCRATCH / "causal_marching_full.json"
    if not smoke:
        scratch_path.write_text(
            __import__("json").dumps(payload, indent=2) + "\n", encoding="utf-8"
        )

    write_json("causal_marching.json", payload)
    print("wrote docs/benchmarks/causal_marching.json")
    if not smoke:
        print(f"wrote {scratch_path}")


if __name__ == "__main__":
    main()
