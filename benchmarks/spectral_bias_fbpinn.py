# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Equal-budget spectral-bias comparison: plain / Fourier / Mscale / adaptive / FBPINN.

Manufactured 1-D target ``sin(2 pi f x)`` on ``[0, 1]``. Arms share parameter
count and Adam step budget. Records mode-wise error and NTK task alignment as
mechanism evidence.

Modes
-----
* ``--smoke`` (default): 1 seed, tiny nets — CI wiring gate.
* ``--full``: multiple seeds / frequencies — acceptance artifact.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from _common import provenance, write_json  # noqa: E402

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


def _count_params(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def _match_hidden(builder, target_params: int, *, cs, comps, **kwargs) -> torch.nn.Module:
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


def _train(field: torch.nn.Module, coords: torch.Tensor, target: torch.Tensor, cfg: BenchConfig) -> float:
    opt = torch.optim.Adam(field.parameters(), lr=cfg.lr)
    for _ in range(cfg.steps):
        opt.zero_grad()
        pred = field.forward_values(coords)[:, 0]
        loss = torch.mean((pred - target) ** 2)
        loss.backward()
        opt.step()
    with torch.no_grad():
        return float(torch.mean((field.forward_values(coords)[:, 0] - target) ** 2))


def _mode_errors(
    field: torch.nn.Module, coords: torch.Tensor, target: torch.Tensor, modes: tuple[int, ...]
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


def _build_arms(cs: CoordinateSpec, comps: ComponentSpec, target_params: int) -> dict[str, Any]:
    builders = {
        "plain": lambda hidden: OneLayerVectorField(
            coordinate_spec=cs, components=comps, hidden=hidden, base="tanh", dtype=DTYPE
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
    return {
        name: _match_hidden(fn, target_params, cs=cs, comps=comps)
        for name, fn in builders.items()
    }


def _run_seed(cfg: BenchConfig, seed: int, freq: int) -> dict[str, Any]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    cs = CoordinateSpec(("x",), domain=((0.0, 1.0),))
    comps = ComponentSpec(("u",))
    x = torch.linspace(0.0, 1.0, cfg.n_grid, dtype=DTYPE).unsqueeze(-1)
    target = torch.sin(2 * np.pi * freq * x[:, 0])
    modes = (2, freq // 2, freq) if freq >= 4 else (2, freq)

    ref = OneLayerVectorField(
        coordinate_spec=cs, components=comps, hidden=cfg.hidden, base="tanh", dtype=DTYPE
    )
    target_params = _count_params(ref)
    arms = _build_arms(cs, comps, target_params)

    results: dict[str, Any] = {"seed": seed, "freq": freq, "arms": {}}
    for name, field in arms.items():
        torch.manual_seed(seed)
        mse = _train(field, x, target, cfg)
        metrics = _ntk_metrics(field, x, target, modes)
        mode_err = _mode_errors(field, x, target, modes)
        results["arms"][name] = {
            "mse": mse,
            "n_params": _count_params(field),
            "mode_error": mode_err,
            **metrics,
        }
    return results


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
        )
    else:
        cfg = BenchConfig(
            steps=60,
            lr=1e-2,
            n_grid=64,
            hidden=16,
            seeds=(0,),
            freqs=(8,),
        )

    t0 = time.perf_counter()
    runs = [_run_seed(cfg, seed, freq) for freq in cfg.freqs for seed in cfg.seeds]
    fbpinn_mses = [r["arms"]["fbpinn"]["mse"] for r in runs]
    plain_mses = [r["arms"]["plain"]["mse"] for r in runs]
    payload = provenance(
        schema="spectral_bias_fbpinn/v2",
        config={
            "steps": cfg.steps,
            "seeds": list(cfg.seeds),
            "freqs": list(cfg.freqs),
            "mode": "full" if args.full else "smoke",
        },
    )
    payload.update(
        {
            "runs": runs,
            "median_mse_plain": float(np.median(plain_mses)),
            "median_mse_fbpinn": float(np.median(fbpinn_mses)),
            "fbpinn_beats_plain_all": all(
                r["arms"]["fbpinn"]["mse"] < r["arms"]["plain"]["mse"] for r in runs
            ),
            "elapsed_seconds": time.perf_counter() - t0,
        }
    )
    if not args.full:
        assert payload["fbpinn_beats_plain_all"]
    write_json("spectral_bias_fbpinn.json", payload)
    print("wrote docs/benchmarks/spectral_bias_fbpinn.json")


if __name__ == "__main__":
    main()
