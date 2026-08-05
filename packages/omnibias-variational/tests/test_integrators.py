# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Variational / symplectic integrators.

- Stormer-Verlet keeps the energy error *bounded* over a long horizon (it is
  symplectic), whereas explicit Euler drifts without bound.
- The midpoint discrete Euler-Lagrange residual is a consistent (2nd-order)
  discretisation: on the exact solution it shrinks like ``dt^3`` as ``dt`` halves.
All float64, torch/jax parity.
"""

from __future__ import annotations

import numpy as np
import torch
from _traj import to_np
from omnibias.variational import Lagrangian
from omnibias.variational.jax import ops as jv
from omnibias.variational.torch import ops as tv

OMEGA = 2.0


def _sho_lagrangian():  # type: ignore[no-untyped-def]
    return Lagrangian(
        lambda q, qd, t: 0.5 * (qd**2).sum(-1) - 0.5 * OMEGA**2 * (q**2).sum(-1),
        dof=("q",),
    )


def _grad_v(q):  # type: ignore[no-untyped-def]
    return OMEGA**2 * q


def test_stormer_verlet_energy_is_bounded() -> None:
    dt, n_steps = 0.05, 4000
    q = torch.ones(1, 1, dtype=torch.float64)
    v = torch.zeros(1, 1, dtype=torch.float64)
    e0 = 0.5 * OMEGA**2
    max_err = 0.0
    for _ in range(n_steps):
        q, v = tv.stormer_verlet_step(q, v, grad_potential=_grad_v, dt=dt)
        e = 0.5 * float(v[0, 0]) ** 2 + 0.5 * OMEGA**2 * float(q[0, 0]) ** 2
        max_err = max(max_err, abs(e - e0) / e0)
    # Symplectic: bounded, O((omega dt)^2) energy oscillation.
    assert max_err < 1e-2


def test_explicit_euler_energy_drifts() -> None:
    dt, n_steps = 0.05, 4000
    q, v = 1.0, 0.0
    e0 = 0.5 * OMEGA**2
    for _ in range(n_steps):
        q, v = q + dt * v, v - dt * OMEGA**2 * q
        e = 0.5 * v**2 + 0.5 * OMEGA**2 * q**2
    # Non-symplectic: energy grows without bound (huge relative error).
    assert (abs(e - e0) / e0) > 1.0


def test_discrete_el_is_second_order_consistent() -> None:
    centers = np.array([0.2, 0.4, 0.6, 0.8, 1.0], dtype=np.float64)
    lag = _sho_lagrangian()

    def max_residual(h):  # type: ignore[no-untyped-def]
        qp = torch.as_tensor(np.cos(OMEGA * (centers - h))[:, None], dtype=torch.float64)
        qc = torch.as_tensor(np.cos(OMEGA * centers)[:, None], dtype=torch.float64)
        qn = torch.as_tensor(np.cos(OMEGA * (centers + h))[:, None], dtype=torch.float64)
        r = tv.discrete_euler_lagrange_residual(qp, qc, qn, lagrangian=lag, dt=h)
        return float(to_np(r).__abs__().max())

    r_coarse = max_residual(0.1)
    r_fine = max_residual(0.05)
    # 2nd-order accurate integrator -> local residual ~ dt^3, ratio ~ 8.
    assert r_fine < r_coarse / 4.0


def test_integrators_cross_backend() -> None:
    import jax.numpy as jnp

    lag = _sho_lagrangian()
    centers = np.array([0.2, 0.5, 0.9], dtype=np.float64)
    h = 0.1
    qp = np.cos(OMEGA * (centers - h))[:, None]
    qc = np.cos(OMEGA * centers)[:, None]
    qn = np.cos(OMEGA * (centers + h))[:, None]
    t = tv.discrete_euler_lagrange_residual(
        torch.as_tensor(qp), torch.as_tensor(qc), torch.as_tensor(qn), lagrangian=lag, dt=h,
    )
    j = jv.discrete_euler_lagrange_residual(
        jnp.asarray(qp), jnp.asarray(qc), jnp.asarray(qn), lagrangian=lag, dt=h,
    )
    assert np.allclose(to_np(t), to_np(j), rtol=1e-12, atol=1e-12)

    q0 = jnp.ones((1, 1), dtype=jnp.float64)
    v0 = jnp.zeros((1, 1), dtype=jnp.float64)
    qn_j, vn_j = jv.stormer_verlet_step(q0, v0, grad_potential=lambda q: OMEGA**2 * q, dt=0.05)
    qn_t, vn_t = tv.stormer_verlet_step(
        torch.ones(1, 1, dtype=torch.float64), torch.zeros(1, 1, dtype=torch.float64),
        grad_potential=_grad_v, dt=0.05,
    )
    assert np.allclose(to_np(qn_j), to_np(qn_t), rtol=1e-12, atol=1e-12)
    assert np.allclose(to_np(vn_j), to_np(vn_t), rtol=1e-12, atol=1e-12)
