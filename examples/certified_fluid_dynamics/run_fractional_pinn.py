# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Learnable-alpha PINN: recover the fractional dissipation order jointly with the field.

An honest inverse-problem PINN that realises the original intuition -- *"the
derivative order is a learnable, possibly fractional, parameter"* -- as a
parameter-identification task.

The unforced fractional shear ``U(y,t)`` (the x-velocity of a divergence-free
shear) obeys the linear fractional heat equation

.. math::  \partial_t U + \nu (-\Delta)^{\alpha} U = 0 ,

so each Fourier mode decays at its own rate ``nu k^{2 alpha}``. We generate data
from a *known* ``alpha_true`` at a few time snapshots, then train a spectral
neural field ``U_theta(y,t) = sum_k g_k(t) sin(k y)`` (mode amplitudes from a
small MLP in ``t``) together with a differentiable
:class:`~omnibias.fractional.torch.order.LearnableOrder` exponent, minimising a
data loss plus the PDE residual evaluated *with the learnable order*. The joint
optimum recovers both the field and ``alpha_true``.

This is system identification of a fractional model; it makes no claim about
Navier-Stokes regularity (``unproven_claim`` never appears).

Usage::

    python -m examples.certified_fluid_dynamics.run_fractional_pinn --smoke
    python -m examples.certified_fluid_dynamics.run_fractional_pinn \
        --alpha-true 1.0 --out-dir "artifacts/omnibias_runs/fractional_pinn"
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
from torch import nn  # noqa: E402

from examples.certified_fluid_dynamics.fractional_ns_theory import TWO_PI  # noqa: E402


def _json_default(o: object) -> object:
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")


def _emit(event: str, **payload: object) -> None:
    print(json.dumps({"event": event, **payload}, sort_keys=True, default=_json_default), flush=True)


class ModeAmplitudeField(nn.Module):
    """Spectral neural field ``U(y,t) = sum_k g_k(t) sin(k y)`` with an MLP time head."""

    def __init__(self, n_modes: int, hidden: int, depth: int) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(1, hidden), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers += [nn.Linear(hidden, n_modes)]
        self.net = nn.Sequential(*layers)
        self.register_buffer("k", torch.arange(1, n_modes + 1, dtype=torch.float64))

    def amplitudes(self, t: torch.Tensor) -> torch.Tensor:
        return self.net(t)

    def forward(self, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        g = self.amplitudes(t)  # (Nt, K)
        basis = torch.sin(torch.outer(self.k, y))  # (K, Ny)
        return g @ basis  # (Nt, Ny)


def _true_amplitudes(t: torch.Tensor, k: torch.Tensor, *, nu: float, alpha_true: float) -> torch.Tensor:
    c0 = 1.0 / k  # multi-scale initial amplitudes
    rate = nu * k ** (2.0 * alpha_true)
    return c0[None, :] * torch.exp(-rate[None, :] * t)


def _true_field(t: torch.Tensor, y: torch.Tensor, k: torch.Tensor, *, nu: float, alpha_true: float) -> torch.Tensor:
    g = _true_amplitudes(t, k, nu=nu, alpha_true=alpha_true)
    return g @ torch.sin(torch.outer(k, y))


def train_recover(args: argparse.Namespace) -> dict:
    torch.manual_seed(args.seed)
    dtype = torch.float64
    field = ModeAmplitudeField(args.modes, args.hidden, args.depth).to(dtype)
    order = LearnableOrder(init=args.alpha_init, lo=0.1, hi=2.0).to(dtype)
    k = field.k

    y = torch.linspace(0.0, TWO_PI, args.ny + 1, dtype=dtype)[:-1]
    t_data = torch.linspace(0.0, args.T, args.n_snapshots, dtype=dtype).unsqueeze(-1)
    u_data = _true_field(t_data, y, k, nu=args.nu, alpha_true=args.alpha_true)

    # The scalar order can move much faster than the field weights.
    opt = torch.optim.Adam(
        [
            {"params": list(field.parameters()), "lr": args.lr},
            {"params": list(order.parameters()), "lr": args.lr * args.alpha_lr_mult},
        ]
    )
    history: list[dict] = []

    for step in range(args.steps):
        opt.zero_grad(set_to_none=True)

        # Data loss at the observed snapshots.
        u_pred = field(t_data, y)
        loss_data = ((u_pred - u_data) ** 2).mean()

        # Physics loss: PDE residual with the LEARNABLE order over t-collocation.
        t_col = (torch.rand(args.collocation, 1, dtype=dtype) * args.T).requires_grad_(True)
        g = field.amplitudes(t_col)  # (Nc, K)
        dg = torch.stack(
            [
                torch.autograd.grad(g[:, j].sum(), t_col, create_graph=True)[0][:, 0]
                for j in range(args.modes)
            ],
            dim=-1,
        )  # (Nc, K)
        alpha = order()
        kpow = torch.exp(2.0 * alpha * torch.log(k))  # k^{2 alpha}, differentiable
        modes_res = dg + args.nu * kpow[None, :] * g  # (Nc, K)
        loss_phys = (modes_res**2).mean()

        loss = loss_data + args.phys_weight * loss_phys
        loss.backward()
        opt.step()

        if step % args.log_every == 0 or step == args.steps - 1:
            rec = {
                "step": step,
                "alpha": float(alpha.detach()),
                "loss_data": float(loss_data.detach()),
                "loss_phys": float(loss_phys.detach()),
            }
            history.append(rec)
            if step % (args.log_every * 10) == 0 or step == args.steps - 1:
                _emit("train", **rec)

    # Validation on a dense grid.
    with torch.no_grad():
        t_val = torch.linspace(0.0, args.T, 41, dtype=dtype).unsqueeze(-1)
        u_val = field(t_val, y)
        u_ref = _true_field(t_val, y, k, nu=args.nu, alpha_true=args.alpha_true)
        rel_l2 = float((u_val - u_ref).norm() / u_ref.norm())
        alpha_final = float(order().item())

    return {
        "alpha_true": args.alpha_true,
        "alpha_recovered": alpha_final,
        "alpha_abs_err": abs(alpha_final - args.alpha_true),
        "field_rel_l2": rel_l2,
        "final_loss_data": history[-1]["loss_data"],
        "final_loss_phys": history[-1]["loss_phys"],
        "history": history,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Learnable-alpha PINN: recover fractional order + field")
    p.add_argument("--alpha-true", type=float, default=1.0)
    p.add_argument("--alpha-init", type=float, default=0.6)
    p.add_argument("--modes", type=int, default=4)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--ny", type=int, default=32)
    p.add_argument("--n-snapshots", type=int, default=6)
    p.add_argument("--collocation", type=int, default=256)
    p.add_argument("--nu", type=float, default=0.05)
    p.add_argument("--T", type=float, default=1.0)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--alpha-lr-mult", type=float, default=20.0, help="order lr = lr * this")
    p.add_argument("--phys-weight", type=float, default=1.0)
    p.add_argument("--threads", type=int, default=1, help="torch CPU threads (1 avoids oversubscription)")
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument(
        "--alpha-sweep", type=float, nargs="*", default=[0.75, 1.0, 1.25, 1.5],
        help="recover each of these alpha_true values (set empty to use --alpha-true only)",
    )
    p.add_argument(
        "--out-dir", type=str,
        default=os.path.join(
            os.environ.get("OMNIBIAS_SCRATCH", "artifacts"),
            "omnibias_runs", "fractional_pinn",
        ),
    )
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


def apply_smoke(args: argparse.Namespace) -> None:
    args.steps = 200
    args.hidden = 32
    args.collocation = 128
    args.alpha_sweep = [1.0]


def main() -> None:
    args = parse_args()
    if args.smoke:
        apply_smoke(args)
    torch.set_num_threads(max(1, int(args.threads)))
    os.makedirs(args.out_dir, exist_ok=True)
    t_start = time.time()
    _emit("start", config=vars(args))

    alpha_values = args.alpha_sweep if args.alpha_sweep else [args.alpha_true]
    runs = []
    for a in alpha_values:
        args.alpha_true = a
        result = train_recover(args)
        runs.append(result)
        _emit(
            "recovered",
            alpha_true=result["alpha_true"],
            alpha_recovered=result["alpha_recovered"],
            alpha_abs_err=result["alpha_abs_err"],
            field_rel_l2=result["field_rel_l2"],
        )

    summary = {
        "unproven_claim": False,
        "note": "Learnable-alpha PINN inverse problem; system-ID of a fractional model, not an NS regularity claim.",
        "config": {k: v for k, v in vars(args).items() if k != "alpha_true"},
        "max_alpha_abs_err": float(max(r["alpha_abs_err"] for r in runs)),
        "max_field_rel_l2": float(max(r["field_rel_l2"] for r in runs)),
        "runs": runs,
        "elapsed_s": round(time.time() - t_start, 2),
    }
    summary_path = os.path.join(args.out_dir, "fractional_pinn_summary.json")
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True, default=_json_default)
    _emit("saved", summary_json=summary_path, elapsed_s=summary["elapsed_s"])


if __name__ == "__main__":
    main()
