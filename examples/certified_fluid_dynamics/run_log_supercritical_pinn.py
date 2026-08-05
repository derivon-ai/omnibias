# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Learnable-beta PINN at Tao's logarithmically supercritical edge.

Track C's fractional order is a *power* ``alpha``; the genuine research frontier
is one logarithm weaker than critical hyperdissipation. Tao (2009) proved global
regularity for the dissipation ``|k|^{5/2} / g(|k|)^2`` with
``g(r) = (log(e + r^2))^{beta}`` **iff** ``\int^\infty dr/(r g^4) = \infty``,
i.e. ``4 beta <= 1`` -- the borderline is ``beta_c = 1/4``.

This driver makes ``beta`` a *learnable* parameter and recovers it from data, the
same inverse-problem PINN as :mod:`run_fractional_pinn` but with the
log-supercritical rate ``nu |k|^{5/2}/(log(e+k^2))^{2 beta}`` in the residual. A
spectral neural field ``U(y,t) = sum_k g_k(t) sin(k y)`` and a differentiable
:class:`~omnibias.fractional.torch.order.LearnableOrder` (bounding ``beta``) are
trained jointly; recovering ``beta`` also recovers *which side of Tao's threshold*
the dynamics lives on.

Honest scope: system identification of a linear dissipation model. Tao's theorem
is **external** and only cited; ``unproven_claim`` never appears.

Usage::

    python -m examples.certified_fluid_dynamics.run_log_supercritical_pinn --smoke
    python -m examples.certified_fluid_dynamics.run_log_supercritical_pinn \
        --out-dir "artifacts/omnibias_runs/log_supercritical"
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch  # noqa: E402
from omnibias.fractional.torch.order import LearnableOrder  # noqa: E402

from examples.certified_fluid_dynamics.fractional_ns_theory import (  # noqa: E402
    TWO_PI,
    classify_log_supercritical,
    tao_dissipation_symbol_torch,
)
from examples.certified_fluid_dynamics.run_fractional_pinn import ModeAmplitudeField  # noqa: E402


def _json_default(o: object) -> object:
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")


def _emit(event: str, **payload: object) -> None:
    print(json.dumps({"event": event, **payload}, sort_keys=True, default=_json_default), flush=True)


def _rate_multiplier(k: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    r"""Log-supercritical rate multiplier ``|k|^{5/2}/(log(e+k^2))^{2 beta}`` per mode."""
    return tao_dissipation_symbol_torch(k * k, beta)


def _true_field(t: torch.Tensor, y: torch.Tensor, k: torch.Tensor, *, nu: float, beta_true: float) -> torch.Tensor:
    c0 = 1.0 / k
    rate = nu * _rate_multiplier(k, torch.tensor(beta_true, dtype=torch.float64))
    g = c0[None, :] * torch.exp(-rate[None, :] * t)
    return g @ torch.sin(torch.outer(k, y))


def train_recover(args: argparse.Namespace) -> dict:
    torch.manual_seed(args.seed)
    dtype = torch.float64
    field = ModeAmplitudeField(args.modes, args.hidden, args.depth).to(dtype)
    beta_param = LearnableOrder(init=args.beta_init, lo=0.05, hi=1.5).to(dtype)
    k = field.k

    y = torch.linspace(0.0, TWO_PI, args.ny + 1, dtype=dtype)[:-1]
    t_data = torch.linspace(0.0, args.T, args.n_snapshots, dtype=dtype).unsqueeze(-1)
    u_data = _true_field(t_data, y, k, nu=args.nu, beta_true=args.beta_true)

    opt = torch.optim.Adam(
        [
            {"params": list(field.parameters()), "lr": args.lr},
            {"params": list(beta_param.parameters()), "lr": args.lr * args.beta_lr_mult},
        ]
    )
    history: list[dict] = []

    for step in range(args.steps):
        opt.zero_grad(set_to_none=True)

        u_pred = field(t_data, y)
        loss_data = ((u_pred - u_data) ** 2).mean()

        t_col = (torch.rand(args.collocation, 1, dtype=dtype) * args.T).requires_grad_(True)
        g = field.amplitudes(t_col)
        dg = torch.stack(
            [torch.autograd.grad(g[:, j].sum(), t_col, create_graph=True)[0][:, 0] for j in range(args.modes)],
            dim=-1,
        )
        beta = beta_param()
        rate = args.nu * _rate_multiplier(k, beta)  # (K,), differentiable in beta
        modes_res = dg + rate[None, :] * g
        loss_phys = (modes_res**2).mean()

        loss = loss_data + args.phys_weight * loss_phys
        loss.backward()
        opt.step()

        if step % args.log_every == 0 or step == args.steps - 1:
            rec = {
                "step": step,
                "beta": float(beta.detach()),
                "loss_data": float(loss_data.detach()),
                "loss_phys": float(loss_phys.detach()),
            }
            history.append(rec)
            if step % (args.log_every * 10) == 0 or step == args.steps - 1:
                _emit("train", **rec)

    with torch.no_grad():
        t_val = torch.linspace(0.0, args.T, 41, dtype=dtype).unsqueeze(-1)
        u_val = field(t_val, y)
        u_ref = _true_field(t_val, y, k, nu=args.nu, beta_true=args.beta_true)
        rel_l2 = float((u_val - u_ref).norm() / u_ref.norm())
        beta_final = float(beta_param().item())

    true_applies = classify_log_supercritical(args.beta_true)["divergence_condition_met"]
    rec_applies = classify_log_supercritical(beta_final)["divergence_condition_met"]
    return {
        "beta_true": args.beta_true,
        "beta_recovered": beta_final,
        "beta_abs_err": abs(beta_final - args.beta_true),
        "field_rel_l2": rel_l2,
        "tao_applies_true": true_applies,
        "tao_applies_recovered": rec_applies,
        "regularity_side_recovered": bool(true_applies == rec_applies),
        "final_loss_data": history[-1]["loss_data"],
        "final_loss_phys": history[-1]["loss_phys"],
        "history": history,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Learnable-beta PINN at Tao's log-supercritical edge")
    p.add_argument("--beta-true", type=float, default=0.25)
    p.add_argument("--beta-init", type=float, default=0.4)
    p.add_argument("--modes", type=int, default=6)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--ny", type=int, default=32)
    p.add_argument("--n-snapshots", type=int, default=8)
    p.add_argument("--collocation", type=int, default=256)
    p.add_argument("--nu", type=float, default=0.05)
    p.add_argument("--T", type=float, default=1.0)
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--beta-lr-mult", type=float, default=20.0, help="beta lr = lr * this")
    p.add_argument("--phys-weight", type=float, default=1.0)
    p.add_argument("--threads", type=int, default=1, help="torch CPU threads (1 avoids oversubscription)")
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument(
        "--beta-sweep", type=float, nargs="*", default=[0.15, 0.25, 0.4, 0.6],
        help="recover each of these beta_true values, straddling the 0.25 edge",
    )
    p.add_argument(
        "--out-dir", type=str,
        default=os.path.join(
            os.environ.get("OMNIBIAS_SCRATCH", "artifacts"),
            "omnibias_runs", "log_supercritical",
        ),
    )
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


def apply_smoke(args: argparse.Namespace) -> None:
    args.steps = 400
    args.hidden = 32
    args.collocation = 128
    args.beta_sweep = [0.25]


def main() -> None:
    args = parse_args()
    if args.smoke:
        apply_smoke(args)
    torch.set_num_threads(max(1, int(args.threads)))
    os.makedirs(args.out_dir, exist_ok=True)
    t_start = time.time()
    _emit("start", config=vars(args), borderline_beta=0.25)

    beta_values = args.beta_sweep if args.beta_sweep else [args.beta_true]
    runs = []
    for b in beta_values:
        args.beta_true = b
        result = train_recover(args)
        runs.append(result)
        _emit(
            "recovered",
            beta_true=result["beta_true"],
            beta_recovered=result["beta_recovered"],
            beta_abs_err=result["beta_abs_err"],
            regularity_side_recovered=result["regularity_side_recovered"],
        )

    summary = {
        "unproven_claim": False,
        "note": (
            "Learnable-beta PINN at Tao's log-supercritical edge; system-ID of a linear "
            "dissipation model. Tao's theorem is external, not omnibias-verified."
        ),
        "borderline_beta": 0.25,
        "config": {kk: vv for kk, vv in vars(args).items() if kk != "beta_true"},
        "max_beta_abs_err": float(max(r["beta_abs_err"] for r in runs)),
        "max_field_rel_l2": float(max(r["field_rel_l2"] for r in runs)),
        "all_regularity_sides_recovered": bool(all(r["regularity_side_recovered"] for r in runs)),
        "runs": runs,
        "elapsed_s": round(time.time() - t_start, 2),
    }
    summary_path = os.path.join(args.out_dir, "log_supercritical_summary.json")
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True, default=_json_default)
    _emit("saved", summary_json=summary_path, elapsed_s=summary["elapsed_s"])


if __name__ == "__main__":
    main()
