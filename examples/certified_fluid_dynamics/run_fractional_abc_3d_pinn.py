# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""3D hyperdissipative (fractional) Navier-Stokes PINN, validated against an exact flow.

The genuinely-3D, ``alpha``-dependent extension of Track A
(:mod:`run_abc_3d_pinn`). The dissipation is the **fractional Laplacian**
``(-Delta)^alpha`` (nonlocal), so the momentum residual is evaluated on a full
periodic ``N^3`` grid by FFT (exact for the band-limited ``u = curl(A)``), not at
scattered collocation points.

Exact solution (alpha-dependent): a **Beltrami field on the wavenumber shell**
``|k| = K``,

    U_K(x) = (sin K z + cos K y,  sin K x + cos K z,  sin K y + cos K x),
    u(x,t) = exp(-nu K^{2 alpha} t) U_K(x),   p = -1/2 exp(-2 nu K^{2 alpha} t)|U_K|^2 .

Because ``curl U_K = K U_K`` and ``(-Delta)^a U_K = K^{2a} U_K`` with advection a
pure pressure gradient, this solves fractional NS with the **alpha-dependent**
decay ``nu K^{2 alpha}`` (``K = 1`` recovers Track A's alpha-independent ABC).

Velocity is ``u = curl(A)`` via :class:`VectorPotentialField`, so ``div u = 0`` by
construction. This is a **validated numerical solution of one 3D fractional-NS
instance**, not a global-regularity statement -- ``unproven_claim`` never appears.

Usage
-----
Local smoke (CPU, seconds)::

    python -m examples.certified_fluid_dynamics.run_fractional_abc_3d_pinn --smoke

Full run (submit to your GPU cluster)::

    python -m examples.certified_fluid_dynamics.run_fractional_abc_3d_pinn \
        --steps 4000 --grid 32 --K 6 --alpha 1.0 --shell-wavenumber 2 \
        --out-dir "artifacts/omnibias_runs/fractional_abc3d"
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch  # noqa: E402
from omnibias.fractional.torch.order import LearnableOrder  # noqa: E402
from omnibias.pinn._core.components import ComponentSpec  # noqa: E402
from omnibias.pinn._core.coords import CoordinateSpec  # noqa: E402
from omnibias.pinn.torch.cage.incompressible import (  # noqa: E402
    VectorPotentialField,
    coulomb_gauge_loss,
)
from omnibias.pinn.torch.fields.spectral import SpectralVectorField  # noqa: E402

from examples.certified_fluid_dynamics.fractional_ns_theory import (  # noqa: E402
    fractional_ns_residual_torch,
)

TWO_PI = 2.0 * math.pi
VEL = ("u", "v", "w")


# --------------------------------------------------------------------------- #
# Exact Beltrami-shell solution (ground truth / initial condition).           #
# --------------------------------------------------------------------------- #
def beltrami_shell_velocity(coords: torch.Tensor, wavenumber: int, nu: float, alpha: float) -> torch.Tensor:
    """Exact velocity ``u(x,t) = exp(-nu K^{2a} t) U_K(x)``; returns ``(B, 3)``."""
    x, y, z, t = coords[:, 0], coords[:, 1], coords[:, 2], coords[:, 3]
    k = wavenumber
    rate = nu * (k ** (2.0 * alpha))
    decay = torch.exp(-rate * t).unsqueeze(-1)
    big_u = torch.stack(
        [
            torch.sin(k * z) + torch.cos(k * y),
            torch.sin(k * x) + torch.cos(k * z),
            torch.sin(k * y) + torch.cos(k * x),
        ],
        dim=-1,
    )
    return decay * big_u


# --------------------------------------------------------------------------- #
# Model / sampling.                                                           #
# --------------------------------------------------------------------------- #
def build_field(K: int, time_hidden: int, time_depth: int, dtype: torch.dtype, device: str) -> VectorPotentialField:
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


def grid_coords(n: int, t: float, dtype: torch.dtype, device: str) -> torch.Tensor:
    """Uniform periodic ``N^3`` grid at fixed time ``t``; returns ``(N^3, 4)``."""
    ax = torch.arange(n, dtype=dtype, device=device) * (TWO_PI / n)
    gx, gy, gz = torch.meshgrid(ax, ax, ax, indexing="ij")
    xyz = torch.stack([gx.reshape(-1), gy.reshape(-1), gz.reshape(-1)], dim=-1)
    tt = torch.full((xyz.shape[0], 1), float(t), dtype=dtype, device=device)
    return torch.cat([xyz, tt], dim=-1)


def batch_coords(n: int, T: float, dtype: torch.dtype, device: str, *, t0: bool = False) -> torch.Tensor:
    xyz = torch.rand(n, 3, dtype=dtype, device=device) * TWO_PI
    t = torch.zeros(n, 1, dtype=dtype, device=device) if t0 else torch.rand(n, 1, dtype=dtype, device=device) * T
    return torch.cat([xyz, t], dim=-1)


def velocity_stack(state) -> torch.Tensor:
    return torch.stack([state.ops.value(state, c) for c in VEL], dim=-1)


def field_grid_fields(field, n: int, t: float, dtype: torch.dtype, device: str):
    """Sample ``u = curl(A)``, ``p`` and the closed-form ``u_t`` on the ``N^3`` grid."""
    coords = grid_coords(n, t, dtype, device)
    state = field(coords)
    u = velocity_stack(state).reshape(n, n, n, 3).permute(3, 0, 1, 2)  # (3, N, N, N)
    p = state.ops.value(state, "p").reshape(n, n, n)
    u_t = torch.stack(
        [state.ops.derivative(state, c, axis="t", order=1).reshape(n, n, n) for c in VEL]
    )  # (3, N, N, N), closed-form time derivative
    return u, p, u_t


# --------------------------------------------------------------------------- #
# Train / validate.                                                           #
# --------------------------------------------------------------------------- #
def train(args: argparse.Namespace, device: str):
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    torch.manual_seed(args.seed)
    field = build_field(args.K, args.time_hidden, args.time_depth, dtype, device)

    params = [{"params": list(field.parameters()), "lr": args.lr}]
    order = None
    if args.learn_alpha:
        order = LearnableOrder(init=args.alpha_init, lo=0.25, hi=2.0).to(dtype).to(device)
        params.append({"params": list(order.parameters()), "lr": args.lr * args.alpha_lr_mult})
    # Interior-trajectory supervision from the exact solution. Required to make a
    # single-shell alpha identifiable, and it also conditions the fixed-alpha
    # forward solve (IC + PDE alone stalls in a poor minimum at wavenumber K > 1).
    use_data = args.learn_alpha or args.supervise
    opt = torch.optim.Adam(params)
    scheduler = (
        None
        if args.no_lr_decay
        else torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps, eta_min=args.lr * 0.01)
    )

    history: list[dict[str, float]] = []
    t_start = time.time()
    for step in range(args.steps):
        opt.zero_grad(set_to_none=True)
        alpha = order() if order is not None else torch.tensor(args.alpha, dtype=dtype, device=device)

        # Fractional-NS residual on a full periodic grid at a random time.
        t_rand = float(torch.rand(1).item() * args.T)
        u, p, u_t = field_grid_fields(field, args.grid, t_rand, dtype, device)
        res, div = fractional_ns_residual_torch(u, p, u_t, alpha=alpha, nu=args.nu)
        loss_pde = (res * res).mean()

        # Initial condition (random batch at t=0) + Coulomb gauge.
        ic = batch_coords(args.ic_batch, args.T, dtype, device, t0=True)
        ic_state = field(ic)
        u_ic = velocity_stack(ic_state)
        u_ic_true = beltrami_shell_velocity(ic, args.shell_wavenumber, args.nu, args.alpha)
        loss_ic = ((u_ic - u_ic_true) ** 2).mean()
        loss_gauge = coulomb_gauge_loss(field, ic, inner_state=ic_state.extra["_cage_inner_state"])

        loss = loss_pde + args.ic_weight * loss_ic + args.gauge_weight * loss_gauge

        # Interior trajectory supervision from the exact solution.
        loss_data = torch.zeros((), dtype=dtype, device=device)
        if use_data:
            data = batch_coords(args.data_batch, args.T, dtype, device)
            u_data = velocity_stack(field(data))
            u_data_true = beltrami_shell_velocity(data, args.shell_wavenumber, args.nu, args.alpha)
            loss_data = ((u_data - u_data_true) ** 2).mean()
            loss = loss + args.data_weight * loss_data

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
                "max_abs_div": float(div.abs().max().detach()),
                "lr": opt.param_groups[0]["lr"],
                "elapsed_s": round(time.time() - t_start, 2),
            }
            if use_data:
                rec["data"] = float(loss_data.detach())
            if order is not None:
                rec["alpha"] = float(alpha.detach())
            history.append(rec)
            print(json.dumps(rec), flush=True)

    recovered_alpha = float(order().item()) if order is not None else args.alpha
    return field, history, recovered_alpha


@torch.no_grad()
def validate(field, args: argparse.Namespace, device: str, alpha_eval: float) -> dict[str, float]:
    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    metrics: dict[str, float] = {}
    rel_l2s, divs = [], []
    for t in (0.0, 0.25 * args.T, 0.5 * args.T, args.T):
        u, p, u_t = field_grid_fields(field, args.grid, float(t), dtype, device)
        coords = grid_coords(args.grid, float(t), dtype, device)
        u_true = beltrami_shell_velocity(coords, args.shell_wavenumber, args.nu, args.alpha)
        u_true = u_true.reshape(args.grid, args.grid, args.grid, 3).permute(3, 0, 1, 2)
        rel_l2s.append(float((u - u_true).norm() / u_true.norm()))
        res, div = fractional_ns_residual_torch(u, p, u_t, alpha=alpha_eval, nu=args.nu)
        divs.append(float(div.abs().max()))
        if t == 0.0:
            metrics["rms_residual_t0"] = float(res.pow(2).mean().sqrt())
    metrics["rel_l2_velocity_mean"] = float(sum(rel_l2s) / len(rel_l2s))
    metrics["rel_l2_velocity_max"] = float(max(rel_l2s))
    metrics["max_abs_divergence"] = float(max(divs))
    return metrics


# --------------------------------------------------------------------------- #
# Entry point.                                                                #
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="3D fractional (hyperdissipative) Navier-Stokes PINN")
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--grid", type=int, default=32, help="N: periodic grid resolution per axis")
    p.add_argument("--K", type=int, default=6, help="Fourier mode pairs per spatial axis")
    p.add_argument("--time-hidden", type=int, default=128)
    p.add_argument("--time-depth", type=int, default=3)
    p.add_argument("--alpha", type=float, default=1.0, help="fractional dissipation order (true / fixed)")
    p.add_argument("--shell-wavenumber", type=int, default=2, help="K: Beltrami shell (rate ~ K^{2a})")
    p.add_argument("--nu", type=float, default=0.05)
    p.add_argument("--T", type=float, default=1.0)
    p.add_argument("--ic-batch", type=int, default=4096)
    p.add_argument("--data-batch", type=int, default=4096)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--ic-weight", type=float, default=25.0)
    p.add_argument("--gauge-weight", type=float, default=1e-3)
    p.add_argument("--data-weight", type=float, default=25.0)
    p.add_argument(
        "--supervise", action="store_true",
        help="add exact-solution interior supervision (validated solve; conditions K>1 shells)",
    )
    p.add_argument("--learn-alpha", action="store_true", help="recover alpha jointly (adds interior data)")
    p.add_argument("--alpha-init", type=float, default=0.75)
    p.add_argument("--alpha-lr-mult", type=float, default=10.0)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--dtype", choices=("float64", "float32"), default="float64")
    p.add_argument("--log-every", type=int, default=200)
    p.add_argument("--no-lr-decay", action="store_true")
    p.add_argument("--threads", type=int, default=0, help="torch CPU threads (0 leaves default; use 1 on CPU)")
    p.add_argument(
        "--out-dir", type=str,
        default=os.path.join(
            os.environ.get("OMNIBIAS_SCRATCH", "artifacts"),
            "omnibias_runs", "fractional_abc3d",
        ),
    )
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


def apply_smoke(args: argparse.Namespace) -> None:
    args.steps = 30
    args.grid = 8
    args.K = 3
    args.time_hidden = 32
    args.time_depth = 2
    args.ic_batch = 256
    args.data_batch = 256
    args.log_every = 10
    args.threads = 1


def main() -> None:
    args = parse_args()
    if args.smoke:
        apply_smoke(args)
    if args.threads > 0:
        torch.set_num_threads(args.threads)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu = torch.cuda.get_device_name(0) if device == "cuda" else "none"
    print(json.dumps({"event": "start", "device": device, "gpu": gpu, "config": vars(args)}), flush=True)

    field, history, recovered_alpha = train(args, device)
    metrics = validate(field, args, device, recovered_alpha)
    if args.learn_alpha:
        metrics["alpha_true"] = args.alpha
        metrics["alpha_recovered"] = recovered_alpha
        metrics["alpha_abs_err"] = abs(recovered_alpha - args.alpha)
    print(json.dumps({"event": "validate", **metrics}, sort_keys=True), flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    ckpt = os.path.join(args.out_dir, "fractional_abc3d_field.pt")
    torch.save(field.state_dict(), ckpt)
    summary = {
        "config": vars(args),
        "device": device,
        "gpu": gpu,
        "history": history,
        "metrics": metrics,
        "checkpoint": ckpt,
        "unproven_claim": False,
        "note": (
            "validated numerical solution of one 3D fractional-NS instance "
            "(exact decaying Beltrami shell); not a global-regularity statement"
        ),
    }
    metrics_path = os.path.join(args.out_dir, "metrics.json")
    with open(metrics_path, "w") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True)
    print(json.dumps({"event": "saved", "metrics_json": metrics_path}), flush=True)


if __name__ == "__main__":
    main()
