# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Track A: 3D incompressible Navier-Stokes PINN, validated against exact ABC flow.

Trains a neural velocity field ``u = curl(A)`` -- divergence-free *by
construction* via the :class:`VectorPotentialField` cage -- with omnibias
closed-form derivatives, to satisfy the incompressible 3D Navier-Stokes momentum
residual on a periodic box. It is scored against the known exact **decaying
Arnold-Beltrami-Childress (ABC)** solution

    U(x)   = (sin z + cos y,  sin x + cos z,  sin y + cos x)
    u(x,t) = exp(-nu t) U(x),   p(x,t) = -0.5 exp(-2 nu t) |U|^2,   f = 0

which is an exact unsteady solution of incompressible NS (Beltrami:
``curl U = U``, ``Delta U = -U``, so advection is balanced by pressure and the
viscous term gives pure exponential decay).

This is a **validated numerical solution of one specific 3D NS instance**, not a
global-regularity statement. No ``unproven_claim`` is made anywhere.

Usage
-----
Local smoke test (CPU, seconds)::

    python -m examples.certified_fluid_dynamics.run_abc_3d_pinn --smoke

Full run (submit to your GPU cluster)::

    python -m examples.certified_fluid_dynamics.run_abc_3d_pinn \
        --steps 20000 --out-dir "artifacts/omnibias_runs/abc3d"
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time

import torch
from omnibias.pinn._core.components import ComponentSpec
from omnibias.pinn._core.coords import CoordinateSpec
from omnibias.pinn.torch.cage.incompressible import (
    VectorPotentialField,
    coulomb_gauge_loss,
)
from omnibias.pinn.torch.equations import NavierStokes
from omnibias.pinn.torch.fields.spectral import SpectralVectorField

TWO_PI = 2.0 * math.pi
VEL = ("u", "v", "w")


# --------------------------------------------------------------------------- #
# Exact decaying ABC solution (ground truth / initial condition).             #
# --------------------------------------------------------------------------- #
def abc_velocity(coords: torch.Tensor, nu: float) -> torch.Tensor:
    """Exact decaying ABC velocity ``u(x,t) = exp(-nu t) U(x)``; returns ``(B, 3)``."""
    x, y, z, t = coords[:, 0], coords[:, 1], coords[:, 2], coords[:, 3]
    decay = torch.exp(-nu * t).unsqueeze(-1)
    big_u = torch.stack(
        [
            torch.sin(z) + torch.cos(y),
            torch.sin(x) + torch.cos(z),
            torch.sin(y) + torch.cos(x),
        ],
        dim=-1,
    )
    return decay * big_u


# --------------------------------------------------------------------------- #
# Model / data.                                                               #
# --------------------------------------------------------------------------- #
def build_field(
    K: int, time_hidden: int, time_depth: int, dtype: torch.dtype, device: str
) -> VectorPotentialField:
    coord = CoordinateSpec(
        axes=("x", "y", "z", "t"),
        periodicity=(True, True, True, False),
        domain=((0.0, TWO_PI), (0.0, TWO_PI), (0.0, TWO_PI), (0.0, 1.0)),
        time_axis="t",
    )
    components = ComponentSpec(names=("A1", "A2", "A3", "p"))
    base = SpectralVectorField(
        coordinate_spec=coord,
        components=components,
        K=K,
        L=TWO_PI,
        time_hidden=time_hidden,
        time_depth=time_depth,
        activation="tanh",
        dtype=dtype,
    )
    field = VectorPotentialField(
        base=base,
        A_components=("A1", "A2", "A3"),
        velocity_names=VEL,
        passthrough_names=("p",),
        spatial_axes=("x", "y", "z"),
    )
    return field.to(device)


def sample_coords(
    n: int, T: float, dtype: torch.dtype, device: str, *, t0: bool = False
) -> torch.Tensor:
    xyz = torch.rand(n, 3, dtype=dtype, device=device) * TWO_PI
    if t0:
        t = torch.zeros(n, 1, dtype=dtype, device=device)
    else:
        t = torch.rand(n, 1, dtype=dtype, device=device) * T
    return torch.cat([xyz, t], dim=-1)


def velocity_stack(state) -> torch.Tensor:
    return torch.stack([state.ops.value(state, c) for c in VEL], dim=-1)


# --------------------------------------------------------------------------- #
# Train / validate.                                                           #
# --------------------------------------------------------------------------- #
def train(args: argparse.Namespace, device: str):
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    torch.manual_seed(args.seed)
    field = build_field(args.K, args.time_hidden, args.time_depth, dtype, device)
    equation = NavierStokes(
        viscosity=args.nu,
        density=1.0,
        form="primitive_3d",
        velocity=VEL,
        pressure="p",
        incompressibility="hard",
    )
    opt = torch.optim.Adam(field.parameters(), lr=args.lr)
    scheduler = (
        None
        if args.no_lr_decay
        else torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=args.steps, eta_min=args.lr * 0.01
        )
    )
    history: list[dict[str, float]] = []
    t_start = time.time()
    for step in range(args.steps):
        opt.zero_grad(set_to_none=True)

        pde_state = field(sample_coords(args.batch, args.T, dtype, device))
        res = equation(pde_state).residual
        loss_pde = (res * res).mean()

        ic = sample_coords(args.ic_batch, args.T, dtype, device, t0=True)
        ic_state = field(ic)
        loss_ic = ((velocity_stack(ic_state) - abc_velocity(ic, args.nu)) ** 2).mean()
        loss_gauge = coulomb_gauge_loss(
            field, ic, inner_state=ic_state.extra["_cage_inner_state"]
        )

        loss = loss_pde + args.ic_weight * loss_ic + args.gauge_weight * loss_gauge
        loss.backward()
        opt.step()
        if scheduler is not None:
            scheduler.step()

        if step % args.log_every == 0 or step == args.steps - 1:
            rec = {
                "step": step,
                "loss": float(loss.detach()),
                "pde": float(loss_pde.detach()),
                "ic": float(loss_ic.detach()),
                "gauge": float(loss_gauge.detach()),
                "lr": opt.param_groups[0]["lr"],
                "elapsed_s": round(time.time() - t_start, 2),
            }
            history.append(rec)
            print(json.dumps(rec), flush=True)

    return field, equation, history


@torch.no_grad()
def validate(field, equation, args: argparse.Namespace, device: str) -> dict[str, float]:
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    coords = sample_coords(args.val_batch, args.T, dtype, device)
    state = field(coords)
    u_pred = velocity_stack(state)
    u_true = abc_velocity(coords, args.nu)
    rel_l2 = (u_pred - u_true).norm() / u_true.norm()
    max_div = state.ops.divergence(state, VEL).abs().max()
    res = equation(field(coords)).residual
    return {
        "rel_l2_velocity": float(rel_l2),
        "max_abs_divergence": float(max_div),
        "rms_momentum_residual": float(res.pow(2).mean().sqrt()),
    }


# --------------------------------------------------------------------------- #
# Entry point.                                                                #
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Track A: 3D ABC Navier-Stokes PINN")
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--K", type=int, default=6, help="Fourier mode pairs per spatial axis")
    p.add_argument("--time-hidden", type=int, default=128)
    p.add_argument("--time-depth", type=int, default=3)
    p.add_argument("--batch", type=int, default=4096, help="interior collocation batch")
    p.add_argument("--ic-batch", type=int, default=2048)
    p.add_argument("--val-batch", type=int, default=8192)
    p.add_argument("--nu", type=float, default=0.05, help="viscosity (density=1)")
    p.add_argument("--T", type=float, default=1.0, help="time horizon")
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--ic-weight", type=float, default=10.0)
    p.add_argument("--gauge-weight", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--dtype", choices=("float64", "float32"), default="float64")
    p.add_argument("--log-every", type=int, default=500)
    p.add_argument(
        "--no-lr-decay",
        action="store_true",
        help="disable cosine LR decay (default: decay to lr*0.01 over --steps)",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default=os.path.join(
            os.environ.get("OMNIBIAS_SCRATCH", "artifacts"),
            "omnibias_runs",
            "abc3d",
        ),
        help="artifact directory (checkpoint + metrics.json); override with $OMNIBIAS_SCRATCH",
    )
    p.add_argument("--smoke", action="store_true", help="tiny CPU config for a quick check")
    return p.parse_args()


def apply_smoke(args: argparse.Namespace) -> None:
    args.steps = 40
    args.K = 4
    args.time_hidden = 32
    args.time_depth = 2
    args.batch = 256
    args.ic_batch = 256
    args.val_batch = 1024
    args.log_every = 10


def main() -> None:
    args = parse_args()
    if args.smoke:
        apply_smoke(args)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu = torch.cuda.get_device_name(0) if device == "cuda" else "none"
    print(
        json.dumps(
            {"event": "start", "device": device, "gpu": gpu, "config": vars(args)}
        ),
        flush=True,
    )

    field, equation, history = train(args, device)
    metrics = validate(field, equation, args, device)
    print(json.dumps({"event": "validate", **metrics}, sort_keys=True), flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    ckpt = os.path.join(args.out_dir, "abc3d_field.pt")
    torch.save(field.state_dict(), ckpt)
    summary = {
        "config": vars(args),
        "device": device,
        "gpu": gpu,
        "history": history,
        "metrics": metrics,
        "checkpoint": ckpt,
        "unproven_claim": False,
        "note": "validated numerical solution of one 3D NS instance (exact decaying ABC); not a global-regularity statement",
    }
    metrics_path = os.path.join(args.out_dir, "metrics.json")
    with open(metrics_path, "w") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True)
    print(json.dumps({"event": "saved", "metrics_json": metrics_path}), flush=True)


if __name__ == "__main__":
    main()
