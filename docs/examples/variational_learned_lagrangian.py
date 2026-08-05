# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Learn a Lagrangian from data -- a Lagrangian Neural Network (LNN).

Run:

    pip install omnibias-variational[torch]
    python docs/examples/variational_learned_lagrangian.py

The forward map ``acceleration(L; q, qdot, t) = M^{-1} F`` (solve Euler-Lagrange
for the acceleration, ``M = d2L/dqdot^2``) turns a Lagrangian into its equations
of motion. Because it is differentiable in the parameters of ``L``, we can *learn*
a Lagrangian directly from observed accelerations -- the LNN objective
``|| acceleration(L_theta; q, qdot) - qddot* ||^2`` (``lagrangian_dynamics_loss``).

We fit a structured ``L_theta = 1/2 a qdot^2 - V_theta(q)`` (learnable scalar mass
``a`` and a small potential network ``V_theta``) to the ideal pendulum
``qddot* = -sin(q)`` and compare it to a black-box network trained to predict the
acceleration directly. Both reproduce the accelerations on held-out data, but only
the Lagrangian model has a conserved energy *by construction*, so on a long
rollout the true energy stays bounded for the LNN and drifts for the black box.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from omnibias.variational import Lagrangian
from omnibias.variational.torch import ops as var

DT = torch.float64
SEED = 0


# ----------------------------------------------------------------------------
# Ground truth: the ideal pendulum  qddot = -sin(q),  E = 1/2 qdot^2 + (1-cos q)
# ----------------------------------------------------------------------------
def true_acceleration(q: torch.Tensor, qdot: torch.Tensor) -> torch.Tensor:
    return -torch.sin(q)


def energy(q: torch.Tensor, qdot: torch.Tensor) -> torch.Tensor:
    return 0.5 * qdot**2 + (1.0 - torch.cos(q))


# ----------------------------------------------------------------------------
# A tiny functional MLP (params are plain tensors, so torch.func can trace it)
# ----------------------------------------------------------------------------
def init_mlp(sizes: list[int], seed: int) -> list[torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    params: list[torch.Tensor] = []
    for a, b in zip(sizes[:-1], sizes[1:], strict=True):
        w = torch.randn(a, b, generator=g, dtype=DT) * math.sqrt(2.0 / a)
        params.append(w.requires_grad_())
        params.append(torch.zeros(b, dtype=DT, requires_grad=True))
    return params


def mlp_forward(params: list[torch.Tensor], x: torch.Tensor) -> torch.Tensor:
    n = len(params) // 2
    h = x
    for i in range(n):
        h = h @ params[2 * i] + params[2 * i + 1]
        if i < n - 1:
            h = torch.tanh(h)
    return h


# ----------------------------------------------------------------------------
# Symplectic (semi-implicit) Euler rollout of  qddot = acc(q, v)  -- one
# acceleration evaluation per step, structure-preserving for a mechanical system.
# ----------------------------------------------------------------------------
def symplectic_rollout(acc, q0, v0, dt, steps):  # type: ignore[no-untyped-def]
    q, v = q0, v0
    qs, vs = [q], [v]
    for _ in range(steps):
        v = v + dt * acc(q, v)
        q = q + dt * v
        qs.append(q)
        vs.append(v)
    return torch.cat(qs), torch.cat(vs)


def main() -> None:
    torch.set_default_dtype(DT)
    torch.manual_seed(SEED)

    # ---- data: random states with their true accelerations -----------------
    n = 64
    q = (torch.rand(n, 1) * 2 - 1) * 2.5
    qdot = (torch.rand(n, 1) * 2 - 1) * 2.5
    t = torch.zeros(n, 1)
    qddot_star = true_acceleration(q, qdot)

    # ---- model 1: structured Lagrangian  L = 1/2 a qdot^2 - V_theta(q) ------
    mass_raw = torch.zeros((), dtype=DT, requires_grad=True)
    v_params = init_mlp([1, 16, 16, 1], seed=1)

    def lagrangian_fn(qq: torch.Tensor, vv: torch.Tensor, tt: torch.Tensor) -> torch.Tensor:
        a = F.softplus(mass_raw)
        potential = mlp_forward(v_params, qq).squeeze(-1)
        return 0.5 * a * (vv**2).sum(-1) - potential

    lag = Lagrangian(lagrangian_fn, dof=("theta",))
    lnn_params = [mass_raw, *v_params]
    opt = torch.optim.Adam(lnn_params, lr=2e-2)
    print("training the Lagrangian network (lagrangian_dynamics_loss)...", flush=True)
    for step in range(300):
        opt.zero_grad()
        loss = var.lagrangian_dynamics_loss(lag, q, qdot, qddot_star, t)
        loss.backward()
        opt.step()
        if step % 50 == 0:
            print(f"  step {step:4d}   loss = {loss.item():.3e}", flush=True)

    # ---- model 2: black-box acceleration net  qddot = g_phi(q, qdot) --------
    g_params = init_mlp([2, 16, 16, 1], seed=2)
    opt2 = torch.optim.Adam(g_params, lr=2e-2)
    print("training the black-box acceleration network...", flush=True)
    for _ in range(1500):
        opt2.zero_grad()
        pred = mlp_forward(g_params, torch.cat([q, qdot], dim=-1))
        loss2 = ((pred - qddot_star) ** 2).mean()
        loss2.backward()
        opt2.step()

    # ---- held-out accuracy of the two forward accelerations -----------------
    with torch.no_grad():
        qt = (torch.rand(128, 1) * 2 - 1) * 2.5
        vt = (torch.rand(128, 1) * 2 - 1) * 2.5
        tgt = true_acceleration(qt, vt)
        rmse_lnn = (var.acceleration(lag, qt, vt, torch.zeros_like(qt)) - tgt).pow(2).mean().sqrt()
        rmse_bb = (mlp_forward(g_params, torch.cat([qt, vt], -1)) - tgt).pow(2).mean().sqrt()
    print(f"\nheld-out acceleration RMSE:  LNN = {rmse_lnn.item():.3e}   black-box = {rmse_bb.item():.3e}")

    # ---- long rollout: is the *true* energy conserved? ----------------------
    dt, steps = 0.02, 800
    q0 = torch.tensor([[2.0]])
    v0 = torch.tensor([[0.0]])
    with torch.no_grad():
        def acc_lnn(qq, vv):  # type: ignore[no-untyped-def]
            return var.acceleration(lag, qq, vv, torch.zeros_like(qq))

        def acc_bb(qq, vv):  # type: ignore[no-untyped-def]
            return mlp_forward(g_params, torch.cat([qq, vv], -1))

        def acc_true(qq, vv):  # type: ignore[no-untyped-def]
            return true_acceleration(qq, vv)

        q_lnn, v_lnn = symplectic_rollout(acc_lnn, q0, v0, dt, steps)
        q_bb, v_bb = symplectic_rollout(acc_bb, q0, v0, dt, steps)
        q_ref, v_ref = symplectic_rollout(acc_true, q0, v0, dt, steps)

        e0 = energy(q0, v0)
        drift_lnn = (energy(q_lnn, v_lnn) - e0).abs().max().item()
        drift_bb = (energy(q_bb, v_bb) - e0).abs().max().item()
        drift_ref = (energy(q_ref, v_ref) - e0).abs().max().item()
        traj_err_lnn = (q_lnn - q_ref).abs().max().item()

    print(f"\nrollout over {steps * dt:.0f} time units (E0 = {e0.item():.4f}):")
    print(f"  true-energy drift   reference (exact accel) = {drift_ref:.2e}")
    print(f"                      LNN (learned Lagrangian) = {drift_lnn:.2e}")
    print(f"                      black-box acceleration   = {drift_bb:.2e}")
    print(f"  LNN trajectory max error vs reference        = {traj_err_lnn:.2e}")

    assert rmse_lnn.item() < 0.05, "LNN should fit the accelerations"
    assert drift_lnn < 0.1, "the learned Lagrangian should nearly conserve energy"
    assert drift_lnn < drift_bb, "the Lagrangian model should conserve energy better"
    print("\nOK: the learned Lagrangian reproduces the dynamics and conserves energy;")
    print("    the black-box acceleration model does not.")


if __name__ == "__main__":
    main()
