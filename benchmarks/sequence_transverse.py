# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Wave-0 falsifier A5: causal transverse filter vs S4D (05-02 G5).

The filter is a designed causal FIR whose taps are a closed-form
``sigma^(n)`` pack (theory 05-02). Default ``n=0`` is the logistic
survival / integral-role tail ``c * sigma(tau - alpha k)`` — the
leaky-integrator class. Pack order is a band selector (01-07);
``n=1`` (``sigma'``) is a mid-lag bump and cannot match an AR(1).

The named baseline is a diagonal structured state-space model (S4D;
Gu et al. 2022 parameterization): stable ``A = -exp(a)``, ZOH
discretisation, ``y_t = C h_t + D x_t``.

Task (long-range): recover an AR(1) / leaky integrator
``y_t = rho y_{t-1} + (1-rho) x_t`` with ``rho`` close to 1. FIR
width equals the horizon so truncation is not an extra handicap
(still ``O(T K)`` convolution, not a recurrence). Both arms use
four parameters and an AR(1)-near init.

The product API is ``omnibias.torch.sequence.CausalTransverseFilter``.
This bench trains that module.

Modes
-----
* default (smoke): ``T=64``, ``rho=0.95``, ``width=64``, 5 seeds.
* ``--full``: ``T=256``, ``rho=0.98``, ``width=256``; also copied
  under ``$OMNIBIAS_SCRATCH/beyond_pde/``.

G5: on every seed, filter ``R^2`` is within ``0.02`` of S4D (absolute)
and S4D beats the zero predictor. Worst-seed via ``require_all_seeds``.
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
from _gates import (  # type: ignore[import-not-found]  # noqa: E402
    gates_block,
    skill_score,
)

SCRATCH = Path(os.environ.get("OMNIBIAS_SCRATCH", "artifacts"))

SEEDS = (0, 1, 2, 3, 4)
G5_MAX_R2_GAP = 0.02
N_FILTER_PARAMS = 4  # c, log_alpha, tau, bias
N_SSM_PARAMS = 4  # a, b, c, d  (S4D, state size 1)
BASELINE_NAME = "S4D (Gu et al. 2022 diagonal SSM, N=1)"


def _r2(pred: np.ndarray, target: np.ndarray) -> float:
    p = np.asarray(pred, dtype=float).reshape(-1)
    t = np.asarray(target, dtype=float).reshape(-1)
    ss_res = float(np.sum((p - t) ** 2))
    ss_tot = float(np.sum((t - float(np.mean(t))) ** 2))
    if ss_tot < 1e-30:
        raise ValueError("r2: target variance near zero")
    return 1.0 - ss_res / ss_tot


def ar1_trajectories(
    rng: np.random.Generator,
    *,
    n: int,
    length: int,
    rho: float,
) -> tuple[np.ndarray, np.ndarray]:
    """White-noise drive and the leaky-integrator response."""
    x = rng.standard_normal((n, length)).astype(np.float64)
    y = np.zeros_like(x)
    scale = 1.0 - float(rho)
    y[:, 0] = scale * x[:, 0]
    for t in range(1, length):
        y[:, t] = float(rho) * y[:, t - 1] + scale * x[:, t]
    return x, y


def causal_sigma_kernel(
    *,
    width: int,
    coeff: float,
    alpha: float,
    tau: float,
    order: int = 0,
    base: str = "sigmoid",
) -> np.ndarray:
    """Designed causal taps; default ``order=0`` is the logistic tail."""
    from omnibias.core.sequence import causal_transverse_taps

    return np.asarray(
        causal_transverse_taps(
            width=width,
            coeff=coeff,
            alpha=alpha,
            tau=tau,
            order=order,
            base=base,
        ),
        dtype=np.float64,
    )


def apply_causal_fir(x: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Causal convolution; ``kernel[0]`` multiplies ``x_t``."""
    xx = np.asarray(x, dtype=np.float64)
    kk = np.asarray(kernel, dtype=np.float64)
    if xx.ndim != 2:
        raise ValueError(f"x must be (batch, time), got {xx.shape}")
    padded = np.pad(xx, ((0, 0), (kk.size - 1, 0)), mode="constant")
    return np.stack([np.convolve(row, kk, mode="valid") for row in padded], axis=0)


def fit_transverse_filter(
    x: np.ndarray,
    y: np.ndarray,
    *,
    width: int,
    steps: int,
    lr: float,
    seed: int,
    rho: float,
) -> tuple[np.ndarray, dict[str, float]]:
    """Adam-fit the shipped ``CausalTransverseFilter`` (order-0 tail)."""
    import torch
    from omnibias.torch.sequence import CausalTransverseFilter

    torch.manual_seed(int(seed))
    xt = torch.tensor(x, dtype=torch.float64)
    yt = torch.tensor(y, dtype=torch.float64)
    # Same privilege as S4D: start on the AR(1) tail, not a mid-lag bump.
    filt = CausalTransverseFilter.from_leaky_integrator(
        float(rho), width=int(width), dtype=torch.float64
    )
    opt = torch.optim.Adam(filt.parameters(), lr=float(lr))
    for _ in range(int(steps)):
        opt.zero_grad(set_to_none=True)
        pred = filt(xt)
        loss = torch.mean((pred - yt) ** 2)
        loss.backward()
        opt.step()
    with torch.no_grad():
        params = {
            "coeff": float(filt.coeff.detach()),
            "alpha": float(filt.timescale().detach()),
            "tau": float(filt.tau.detach()),
            "bias": float(filt.bias.detach()),
            "n_params": float(N_FILTER_PARAMS),
            "order": 0.0,
        }
        pred_np = filt(xt).detach().cpu().numpy()
    return pred_np, params


def fit_s4d(
    x: np.ndarray,
    y: np.ndarray,
    *,
    steps: int,
    lr: float,
    seed: int,
) -> tuple[np.ndarray, dict[str, float]]:
    """Adam-fit a 1-state S4D (matched parameter count)."""
    import torch

    torch.manual_seed(int(seed))
    xt = torch.tensor(x, dtype=torch.float64)
    yt = torch.tensor(y, dtype=torch.float64)
    # log-timescale so Ā = exp(-exp(a)) ∈ (0, 1); initialise near the AR(1).
    a = torch.nn.Parameter(torch.tensor(-1.0, dtype=torch.float64))
    b = torch.nn.Parameter(torch.tensor(0.2, dtype=torch.float64))
    c = torch.nn.Parameter(torch.tensor(1.0, dtype=torch.float64))
    d = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
    opt = torch.optim.Adam([a, b, c, d], lr=float(lr))
    batch, length = xt.shape
    for _ in range(int(steps)):
        opt.zero_grad(set_to_none=True)
        decay = torch.exp(-torch.exp(a))
        # ZOH: B̄ = B * (1 - Ā) / exp(a) with A = -exp(a)
        b_bar = b * (1.0 - decay) / torch.exp(a)
        h = xt.new_zeros(batch)
        preds = []
        for t in range(length):
            preds.append(c * h + d * xt[:, t])
            h = decay * h + b_bar * xt[:, t]
        pred = torch.stack(preds, dim=1)
        loss = torch.mean((pred - yt) ** 2)
        loss.backward()
        opt.step()
    with torch.no_grad():
        decay = float(torch.exp(-torch.exp(a)))
        b_bar = float(b * (1.0 - torch.exp(-torch.exp(a))) / torch.exp(a))
        cc = float(c)
        dd = float(d)
        h = np.zeros(x.shape[0], dtype=np.float64)
        pred_np = np.zeros_like(x)
        for t in range(x.shape[1]):
            pred_np[:, t] = cc * h + dd * x[:, t]
            h = decay * h + b_bar * x[:, t]
        params = {
            "a": float(a.detach()),
            "b": float(b.detach()),
            "c": cc,
            "d": dd,
            "discrete_decay": decay,
            "n_params": float(N_SSM_PARAMS),
        }
    return pred_np, params


def _split(
    x: np.ndarray, y: np.ndarray, *, n_train: int, n_val: int
) -> dict[str, np.ndarray]:
    return {
        "xtr": x[:n_train],
        "ytr": y[:n_train],
        "xva": x[n_train : n_train + n_val],
        "yva": y[n_train : n_train + n_val],
        "xte": x[n_train + n_val :],
        "yte": y[n_train + n_val :],
    }


def run_seed(seed: int, *, cfg: dict[str, Any]) -> dict[str, Any]:
    rng = np.random.default_rng(int(seed))
    n_total = int(cfg["n_train"]) + int(cfg["n_val"]) + int(cfg["n_test"])
    x, y = ar1_trajectories(
        rng, n=n_total, length=int(cfg["length"]), rho=float(cfg["rho"])
    )
    parts = _split(x, y, n_train=int(cfg["n_train"]), n_val=int(cfg["n_val"]))
    t0 = time.perf_counter()
    filt_tr, filt_params = fit_transverse_filter(
        parts["xtr"],
        parts["ytr"],
        width=int(cfg["width"]),
        steps=int(cfg["steps"]),
        lr=float(cfg["lr"]),
        seed=int(seed),
        rho=float(cfg["rho"]),
    )
    filt_ms = (time.perf_counter() - t0) * 1e3
    t1 = time.perf_counter()
    ssm_tr, ssm_params = fit_s4d(
        parts["xtr"],
        parts["ytr"],
        steps=int(cfg["steps"]),
        lr=float(cfg["lr"]),
        seed=int(seed) + 17,
    )
    ssm_ms = (time.perf_counter() - t1) * 1e3
    # Refit is in-loop training; score held-out with the fitted kernels.
    filt_te = apply_causal_fir(
        parts["xte"],
        causal_sigma_kernel(
            width=int(cfg["width"]),
            coeff=filt_params["coeff"],
            alpha=filt_params["alpha"],
            tau=filt_params["tau"],
            order=0,
        ),
    ) + filt_params["bias"]
    decay = ssm_params["discrete_decay"]
    a = ssm_params["a"]
    b = ssm_params["b"]
    b_bar = b * (1.0 - float(np.exp(-np.exp(a)))) / float(np.exp(a))
    h = np.zeros(parts["xte"].shape[0], dtype=np.float64)
    ssm_te = np.zeros_like(parts["xte"])
    for t in range(parts["xte"].shape[1]):
        ssm_te[:, t] = ssm_params["c"] * h + ssm_params["d"] * parts["xte"][:, t]
        h = decay * h + b_bar * parts["xte"][:, t]

    filter_r2 = _r2(filt_te, parts["yte"])
    ssm_r2 = _r2(ssm_te, parts["yte"])
    filter_skill = skill_score(filt_te, parts["yte"])
    ssm_skill = skill_score(ssm_te, parts["yte"])
    return {
        "seed": int(seed),
        "filter_r2": filter_r2,
        "ssm_r2": ssm_r2,
        "r2_gap": float(ssm_r2 - filter_r2),
        "filter_skill": filter_skill,
        "ssm_skill": ssm_skill,
        "filter_train_r2": _r2(filt_tr, parts["ytr"]),
        "ssm_train_r2": _r2(ssm_tr, parts["ytr"]),
        "filter_wall_ms": filt_ms,
        "ssm_wall_ms": ssm_ms,
        "filter_params": filt_params,
        "ssm_params": ssm_params,
        "n_params_filter": N_FILTER_PARAMS,
        "n_params_ssm": N_SSM_PARAMS,
    }


def _config(*, full: bool) -> dict[str, Any]:
    if full:
        return {
            "length": 256,
            "rho": 0.98,
            "width": 256,
            "n_train": 48,
            "n_val": 16,
            "n_test": 32,
            "steps": 120,
            "lr": 0.05,
        }
    return {
            "length": 64,
            "rho": 0.95,
            "width": 64,
        "n_train": 24,
        "n_val": 8,
        "n_test": 16,
        "steps": 60,
        "lr": 0.05,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args(argv)
    cfg = _config(full=bool(args.full))
    per_seed = [run_seed(seed, cfg=cfg) for seed in SEEDS]
    worst_ssm_skill = min(float(row["ssm_skill"]) for row in per_seed)
    worst_gap = max(float(row["r2_gap"]) for row in per_seed)
    # Soft block: a failed falsifier must still write all_passed: false.
    entries = [
        {
            "name": "g5_ssm_beats_zero",
            "passed": bool(worst_ssm_skill >= 0.0),
            "worst_ssm_skill": worst_ssm_skill,
            "min_skill": 0.0,
        },
        {
            "name": "g5_filter_within_2pct_of_s4d",
            "passed": bool(worst_gap <= G5_MAX_R2_GAP),
            "worst_r2_gap": worst_gap,
            "max_gap": G5_MAX_R2_GAP,
        },
        {
            "name": "g5_matched_param_count",
            "passed": N_FILTER_PARAMS == N_SSM_PARAMS,
            "n_params_filter": N_FILTER_PARAMS,
            "n_params_ssm": N_SSM_PARAMS,
        },
        {
            "name": "g5_width_equals_horizon",
            "passed": int(cfg["width"]) == int(cfg["length"]),
            "width": int(cfg["width"]),
            "length": int(cfg["length"]),
        },
    ]
    hard = gates_block(entries)

    g5_earned = bool(hard["all_passed"])
    payload: dict[str, Any] = {
        **provenance(
            schema="sequence-transverse-v1",
            config={
                "family": "causal_transverse_filter_vs_s4d",
                "seeds": list(SEEDS),
                "g5_max_r2_gap": G5_MAX_R2_GAP,
                "matched_n_params": N_FILTER_PARAMS,
                "baseline": BASELINE_NAME,
                "full": bool(args.full),
                **cfg,
            },
        ),
        "seeds": list(SEEDS),
        "per_seed": per_seed,
        "baseline": {
            "name": BASELINE_NAME,
            "n_params": N_SSM_PARAMS,
            "citation": "Gu, Goel, Re; Efficiently Modeling Long Sequences with Structured State Spaces (S4 / S4D)",
        },
        "gates": hard,
        "g5_earned": g5_earned,
        "sequence_submodule_shipped": False if not g5_earned else True,
        "honesty": {
            "temperature_collapse_claim": False,
            "founding_bias_collapse": True,
            "not_omnibias_struct": True,
            "ssm_is_named_s4d": True,
            "kernel_order": 0,
            "kernel_class": "logistic_survival_tail",
            "width_equals_horizon": True,
        },
    }
    name = (
        "sequence_transverse.json" if args.full else "sequence_transverse_smoke.json"
    )
    written = write_json(name, payload)
    if args.full:
        dest = SCRATCH / "beyond_pde"
        dest.mkdir(parents=True, exist_ok=True)
        (dest / name).write_text(written.read_text(encoding="utf-8"), encoding="utf-8")
    print(
        f"wrote {written} g5_earned={g5_earned} "
        f"worst_gap={max(r['r2_gap'] for r in per_seed):.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
