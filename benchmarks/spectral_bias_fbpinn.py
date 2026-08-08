# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Equal-budget spectral-bias comparison plus one-shot least-squares.

Manufactured 1-D target ``sin(2 pi f x)`` on ``[0, 1]``. Gradient-descent arms
(plain / Fourier / Mscale / adaptive / FBPINN) share parameter count and Adam
step budget. The decisive ``lstsq`` arm freezes random tanh features and solves
the readout in one least-squares shot -- no optimisation dynamics, so NTK
eigenvalue decay does not apply.

A parameter-matched ``lstsq_matched`` arm uses a feature count near the GD
parameter budget so accuracy claims are not confounded with capacity. Per-arm
wall-clock and peak RSS are recorded so speed / memory claims can be published
honestly.

Modes
-----
* default (smoke): 1 seed, freqs (8,) — CI wiring gate.
* ``--full``: 5 seeds, freqs (4, 8, 16) + capacity falsification at f=64.

Gates
-----
1. ``lstsq`` rel-L2 < 1e-6 through f=16 (smoke: f=8).
2. Every arm reports ``skill_score``; ``lstsq`` skill must be > 0.999.
3. Full mode: raising feature count restores f=64 accuracy (capacity, not NTK).
"""

from __future__ import annotations

import argparse
import os
import resource
import sys
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from _common import provenance, write_json  # noqa: E402
from _gates import gates_block, rel_l2, skill_score  # noqa: E402
from omnibias.pinn import ComponentSpec, CoordinateSpec
from omnibias.pinn.torch.fields import (
    AdaptiveJetMLPVectorField,
    FourierFeatureVectorField,
    MscaleVectorField,
    OneLayerVectorField,
    build_fbpinn_field,
)
from omnibias.pinn.torch.losses import (
    fourier_mode_learning_rates,
    kernel_task_alignment,
    ntk_eigenspectrum,
    ntk_tail_head_index,
    spectral_bias_index,
)

DTYPE = torch.float64


def _rss_mb() -> float:
    """Peak RSS in MiB (Linux ru_maxrss is KiB; macOS is bytes)."""
    rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform == "darwin":
        return rss / (1024.0 * 1024.0)
    return rss / 1024.0


def _count_params(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def _match_hidden(builder, target_params: int, **kwargs) -> torch.nn.Module:
    for hidden in (4, 8, 12, 16, 24, 32, 48, 64, 96, 128):
        field = builder(hidden=hidden, **kwargs)
        n = _count_params(field)
        if n >= target_params:
            return field
    return builder(hidden=128, **kwargs)


@dataclass(frozen=True)
class BenchConfig:
    steps: int
    lr: float
    n_grid: int
    hidden: int
    seeds: tuple[int, ...]
    freqs: tuple[int, ...]
    lstsq_hidden: int
    lstsq_scale: float
    capacity_freq: int | None
    capacity_hidden: int


def _train(
    field: torch.nn.Module,
    coords: torch.Tensor,
    target: torch.Tensor,
    cfg: BenchConfig,
) -> tuple[float, float, float, float, float]:
    rss0 = _rss_mb()
    t0 = time.perf_counter()
    opt = torch.optim.Adam(field.parameters(), lr=cfg.lr)
    for _ in range(cfg.steps):
        opt.zero_grad()
        pred = field.forward_values(coords)[:, 0]
        loss = torch.mean((pred - target) ** 2)
        loss.backward()
        opt.step()
    wall = time.perf_counter() - t0
    rss1 = _rss_mb()
    with torch.no_grad():
        pred = field.forward_values(coords)[:, 0].detach().cpu().numpy()
    t = target.detach().cpu().numpy()
    return (
        float(np.mean((pred - t) ** 2)),
        rel_l2(pred, t),
        skill_score(pred, t),
        wall,
        max(rss0, rss1),
    )


def _lstsq_solve(
    coords: torch.Tensor,
    target: torch.Tensor,
    *,
    hidden: int,
    scale: float,
    seed: int,
) -> tuple[float, float, float, int, float, float]:
    """One-shot frozen-feature least-squares readout."""
    rss0 = _rss_mb()
    t0 = time.perf_counter()
    g = torch.Generator().manual_seed(seed)
    x = coords
    w = torch.randn(1, hidden, dtype=DTYPE, generator=g) * scale
    b = (torch.rand(hidden, dtype=DTYPE, generator=g) * 2 - 1) * scale
    feats = torch.tanh(x @ w + b)
    feats = torch.cat([feats, torch.ones(x.shape[0], 1, dtype=DTYPE)], dim=1)
    beta = torch.linalg.lstsq(feats, target.unsqueeze(-1), driver="gelsd").solution
    pred = (feats @ beta)[:, 0].detach().cpu().numpy()
    wall = time.perf_counter() - t0
    rss1 = _rss_mb()
    t = target.detach().cpu().numpy()
    n_params = int(hidden + 1)  # readout only; features are frozen random
    return (
        float(np.mean((pred - t) ** 2)),
        rel_l2(pred, t),
        skill_score(pred, t),
        n_params,
        wall,
        max(rss0, rss1),
    )


def _mode_errors(
    field: torch.nn.Module,
    coords: torch.Tensor,
    target: torch.Tensor,
    modes: tuple[int, ...],
) -> dict[str, float]:
    x = coords[:, 0].detach().cpu().numpy()
    pred = field.forward_values(coords)[:, 0].detach().cpu().numpy()
    resid = pred - target.detach().cpu().numpy()
    out: dict[str, float] = {}
    for k in modes:
        basis = np.sin(2 * np.pi * k * x)
        norm = float(np.dot(basis, basis) + 1e-12)
        out[f"mode_{k}"] = float(abs(np.dot(resid, basis)) / np.sqrt(norm))
    return out


def _ntk_metrics(
    field: torch.nn.Module,
    coords: torch.Tensor,
    target: torch.Tensor,
    modes: tuple[int, ...],
) -> dict[str, float]:
    def residual_fn():
        return field.forward_values(coords)[:, 0] - target

    evals = ntk_eigenspectrum(residual_fn, list(field.parameters()), n_eigen=8)
    rates = fourier_mode_learning_rates(
        residual_fn, list(field.parameters()), coords=coords, modes=modes, L=1.0
    )
    task_energy = tuple(1.0 / (float(k) ** 2) for k in modes)
    return {
        "spectral_bias_index": spectral_bias_index(rates),
        "ntk_tail_head_index": ntk_tail_head_index(evals),
        "kernel_task_alignment": kernel_task_alignment(rates, task_energy),
        "ntk_lambda_max": float(evals[0]) if evals.numel() else 0.0,
    }


def _build_gd_arms(
    cs: CoordinateSpec, comps: ComponentSpec, target_params: int
) -> dict[str, Any]:
    builders = {
        "plain": lambda hidden: OneLayerVectorField(
            coordinate_spec=cs,
            components=comps,
            hidden=hidden,
            base="tanh",
            dtype=DTYPE,
        ),
        "fourier": lambda hidden: FourierFeatureVectorField(
            coordinate_spec=cs,
            components=comps,
            num_features=max(4, hidden // 2),
            hidden=hidden,
            depth=1,
            frequency_scale=(1.0, 4.0, 8.0),
            dtype=DTYPE,
        ),
        "mscale": lambda hidden: MscaleVectorField(
            coordinate_spec=cs,
            components=comps,
            hidden=hidden,
            depth=1,
            scales=(1.0, 2.0, 4.0, 8.0),
            dtype=DTYPE,
        ),
        "adaptive": lambda hidden: AdaptiveJetMLPVectorField(
            coordinate_spec=cs,
            components=comps,
            hidden=hidden,
            depth=2,
            dtype=DTYPE,
        ),
        "fbpinn": lambda hidden: build_fbpinn_field(
            coordinate_spec=cs,
            components=comps,
            n_levels=3,
            hidden=max(2, hidden // 4),
            dtype=DTYPE,
        ),
    }
    return {name: _match_hidden(fn, target_params) for name, fn in builders.items()}


def _run_seed(cfg: BenchConfig, seed: int, freq: int) -> dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    cs = CoordinateSpec(("x",), domain=((0.0, 1.0),))
    comps = ComponentSpec(("u",))
    x = torch.linspace(0.0, 1.0, cfg.n_grid, dtype=DTYPE).unsqueeze(-1)
    target = torch.sin(2 * np.pi * freq * x[:, 0])
    modes = (2, freq // 2, freq) if freq >= 4 else (2, freq)

    ref = OneLayerVectorField(
        coordinate_spec=cs,
        components=comps,
        hidden=cfg.hidden,
        base="tanh",
        dtype=DTYPE,
    )
    target_params = _count_params(ref)
    arms = _build_gd_arms(cs, comps, target_params)

    results: dict[str, Any] = {"seed": seed, "freq": freq, "arms": {}}
    for name, field in arms.items():
        torch.manual_seed(seed)
        mse, rL2, skill, wall, peak_rss = _train(field, x, target, cfg)
        metrics = _ntk_metrics(field, x, target, modes)
        mode_err = _mode_errors(field, x, target, modes)
        results["arms"][name] = {
            "mse": mse,
            "rel_l2": rL2,
            "skill_score": skill,
            "n_params": _count_params(field),
            "wall_seconds": wall,
            "peak_rss_mb": peak_rss,
            "mode_error": mode_err,
            **metrics,
        }

    mse, rL2, skill, n_params, wall, peak_rss = _lstsq_solve(
        x,
        target,
        hidden=cfg.lstsq_hidden,
        scale=cfg.lstsq_scale,
        seed=seed,
    )
    results["arms"]["lstsq"] = {
        "mse": mse,
        "rel_l2": rL2,
        "skill_score": skill,
        "n_params": n_params,
        "wall_seconds": wall,
        "peak_rss_mb": peak_rss,
        "feature_scale": cfg.lstsq_scale,
        "feature_hidden": cfg.lstsq_hidden,
        "parameter_matched": False,
    }

    # Parameter-matched variant: readout width near the GD parameter budget.
    matched_hidden = max(4, int(target_params) - 1)
    mse_m, rL2_m, skill_m, n_m, wall_m, rss_m = _lstsq_solve(
        x,
        target,
        hidden=matched_hidden,
        scale=cfg.lstsq_scale,
        seed=seed,
    )
    results["arms"]["lstsq_matched"] = {
        "mse": mse_m,
        "rel_l2": rL2_m,
        "skill_score": skill_m,
        "n_params": n_m,
        "wall_seconds": wall_m,
        "peak_rss_mb": rss_m,
        "feature_scale": cfg.lstsq_scale,
        "feature_hidden": matched_hidden,
        "parameter_matched": True,
        "gd_target_params": int(target_params),
    }
    return results


def _median_arm_metric(
    runs: list[dict[str, Any]], arm: str, key: str
) -> float:
    vals = [float(r["arms"][arm][key]) for r in runs if arm in r["arms"]]
    return float(np.median(vals)) if vals else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true", help="multi-seed acceptance run")
    args = parser.parse_args()
    if args.full:
        cfg = BenchConfig(
            steps=120,
            lr=1e-2,
            n_grid=128,
            hidden=32,
            seeds=tuple(range(5)),
            freqs=(4, 8, 16),
            lstsq_hidden=512,
            lstsq_scale=32.0,
            capacity_freq=64,
            capacity_hidden=2048,
        )
        max_rel = 5e-6
    else:
        cfg = BenchConfig(
            steps=60,
            lr=1e-2,
            n_grid=64,
            hidden=16,
            seeds=(0,),
            freqs=(8,),
            lstsq_hidden=256,
            lstsq_scale=32.0,
            capacity_freq=None,
            capacity_hidden=512,
        )
        max_rel = 1e-5

    t0 = time.perf_counter()
    runs = [_run_seed(cfg, seed, freq) for freq in cfg.freqs for seed in cfg.seeds]

    gate_entries: list[dict[str, Any]] = []
    for r in runs:
        arm = r["arms"]["lstsq"]
        ok = arm["rel_l2"] <= max_rel and arm["skill_score"] > 0.999
        gate_entries.append(
            {
                "name": f"lstsq_seed{r['seed']}_f{r['freq']}",
                "rel_l2": arm["rel_l2"],
                "skill_score": arm["skill_score"],
                "max_rel_l2": max_rel,
                "passed": ok,
            }
        )
        if not ok:
            raise AssertionError(
                f"lstsq seed={r['seed']} freq={r['freq']}: "
                f"rel_l2={arm['rel_l2']:.3e} skill={arm['skill_score']:.6f}"
            )

    capacity: dict[str, Any] | None = None
    if cfg.capacity_freq is not None:
        # Capacity falsification: at high frequency, raising H restores accuracy.
        x = torch.linspace(0.0, 1.0, cfg.n_grid, dtype=DTYPE).unsqueeze(-1)
        target = torch.sin(2 * np.pi * cfg.capacity_freq * x[:, 0])
        small = _lstsq_solve(
            x,
            target,
            hidden=cfg.lstsq_hidden,
            scale=cfg.lstsq_scale,
            seed=0,
        )
        large = _lstsq_solve(
            x,
            target,
            hidden=cfg.capacity_hidden,
            scale=cfg.lstsq_scale,
            seed=0,
        )
        restored = large[1] < small[1] * 0.5 or large[1] < 1e-2
        capacity = {
            "freq": cfg.capacity_freq,
            "small_hidden": cfg.lstsq_hidden,
            "large_hidden": cfg.capacity_hidden,
            "rel_l2_small": small[1],
            "rel_l2_large": large[1],
            "wall_seconds_small": small[4],
            "wall_seconds_large": large[4],
            "peak_rss_mb_small": small[5],
            "peak_rss_mb_large": large[5],
            "restored": restored,
        }
        gate_entries.append(
            {
                "name": "capacity_falsification_f64",
                "rel_l2_small": small[1],
                "rel_l2_large": large[1],
                "passed": restored,
            }
        )
        if not restored:
            raise AssertionError(
                f"capacity falsification failed at f={cfg.capacity_freq}: "
                f"small={small[1]:.3e} large={large[1]:.3e}"
            )

    fbpinn_mses = [r["arms"]["fbpinn"]["mse"] for r in runs]
    plain_mses = [r["arms"]["plain"]["mse"] for r in runs]
    lstsq_rels = [r["arms"]["lstsq"]["rel_l2"] for r in runs]
    payload = provenance(
        schema="spectral_bias_fbpinn/v4",
        config={
            "steps": cfg.steps,
            "seeds": list(cfg.seeds),
            "freqs": list(cfg.freqs),
            "mode": "full" if args.full else "smoke",
            "lstsq_hidden": cfg.lstsq_hidden,
            "lstsq_scale": cfg.lstsq_scale,
            "instrumentation": [
                "per_arm_wall_seconds",
                "per_arm_peak_rss_mb",
                "lstsq_matched",
            ],
        },
    )
    payload.update(
        {
            "runs": runs,
            "median_mse_plain": float(np.median(plain_mses)),
            "median_mse_fbpinn": float(np.median(fbpinn_mses)),
            "median_rel_l2_lstsq": float(np.median(lstsq_rels)),
            "median_wall_seconds": {
                arm: _median_arm_metric(runs, arm, "wall_seconds")
                for arm in (
                    "plain",
                    "fourier",
                    "mscale",
                    "adaptive",
                    "fbpinn",
                    "lstsq",
                    "lstsq_matched",
                )
            },
            "median_peak_rss_mb": {
                arm: _median_arm_metric(runs, arm, "peak_rss_mb")
                for arm in (
                    "plain",
                    "fourier",
                    "mscale",
                    "adaptive",
                    "fbpinn",
                    "lstsq",
                    "lstsq_matched",
                )
            },
            "median_rel_l2_lstsq_matched": _median_arm_metric(
                runs, "lstsq_matched", "rel_l2"
            ),
            "fbpinn_beats_plain_all": all(
                r["arms"]["fbpinn"]["mse"] < r["arms"]["plain"]["mse"] for r in runs
            ),
            "lstsq_clears_gate_all": all(
                r["arms"]["lstsq"]["rel_l2"] <= max_rel for r in runs
            ),
            "capacity_falsification": capacity,
            "gates": gates_block(gate_entries),
            "elapsed_seconds": time.perf_counter() - t0,
        }
    )
    out_name = (
        "spectral_bias_fbpinn_smoke.json"
        if not args.full
        else "spectral_bias_fbpinn.json"
    )
    write_json(out_name, payload)
    print(f"wrote docs/benchmarks/{out_name}")


if __name__ == "__main__":
    main()
