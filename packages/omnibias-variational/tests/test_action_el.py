# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Action integral and Euler-Lagrange residual, validated on the harmonic oscillator.

For ``L = 1/2 qdot^2 - 1/2 w^2 q^2`` the harmonic path ``q(t) = cos(w t)`` solves
the equations of motion, so its Euler-Lagrange residual vanishes. Off a solution
(``q = t``) the residual has the closed form ``qddot + w^2 q = w^2 t``. The action
of the free particle over ``[0, 1]`` is exactly ``1/2``. All float64, torch/jax
parity checked.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from _traj import jax_state, sho_specs, to_np, torch_state
from omnibias.fields._core.quadrature import gauss_legendre
from omnibias.fields.torch.ops.integral import quadrature_nodes
from omnibias.variational import Lagrangian
from omnibias.variational.jax import ops as jv
from omnibias.variational.torch import ops as tv

W = 1.3
T = np.array([-0.7, -0.2, 0.4, 1.1, 1.9], dtype=np.float64)


def _sho(dof):  # type: ignore[no-untyped-def]
    return Lagrangian(
        lambda q, qd, t: 0.5 * (qd**2).sum(-1) - 0.5 * W**2 * (q**2).sum(-1),
        dof=dof,
    )


def _free(dof):  # type: ignore[no-untyped-def]
    return Lagrangian(lambda q, qd, t: 0.5 * (qd**2).sum(-1), dof=dof)


def test_el_zero_on_harmonic_solution() -> None:
    state = torch_state(sho_specs, W, T)
    res = to_np(tv.euler_lagrange_residual(state, _sho(("cos",))))
    assert np.allclose(res, 0.0, atol=1e-10)


def test_el_offsolution_matches_closed_form() -> None:
    # q = t is not a solution: EL = qddot + w^2 q = w^2 t.
    state = torch_state(sho_specs, W, T)
    res = to_np(tv.euler_lagrange_residual(state, _sho(("lin",))))
    assert np.allclose(res[:, 0], W**2 * T, atol=1e-10)


def test_el_multi_dof_matches_closed_form() -> None:
    # Uncoupled 2-DOF SHO on (cos, lin): EL_cos = 0, EL_lin = w^2 t.
    state = torch_state(sho_specs, W, T)
    res = to_np(tv.euler_lagrange_residual(state, _sho(("cos", "lin"))))
    assert np.allclose(res[:, 0], 0.0, atol=1e-10)
    assert np.allclose(res[:, 1], W**2 * T, atol=1e-10)


def test_el_free_particle_is_acceleration() -> None:
    # Free particle: EL = qddot. On cos(w t) that is -w^2 cos(w t).
    state = torch_state(sho_specs, W, T)
    res = to_np(tv.euler_lagrange_residual(state, _free(("cos",))))
    assert np.allclose(res[:, 0], -(W**2) * np.cos(W * T), atol=1e-10)


def test_action_free_particle_over_unit_interval() -> None:
    # q = t, L = 1/2 qdot^2 = 1/2, so S = integral_0^1 1/2 dt = 1/2.
    rule = gauss_legendre([(0.0, 1.0)], 8)
    nodes = quadrature_nodes(rule, like=torch.zeros(1, dtype=torch.float64))
    state = torch_state(sho_specs, W, nodes[:, 0].numpy())
    S = float(to_np(tv.action(state, _free(("lin",)), rule=rule)))
    assert abs(S - 0.5) < 1e-12


def test_action_matches_manual_quadrature() -> None:
    rule = gauss_legendre([(0.0, 2.0)], 24)
    nodes = quadrature_nodes(rule, like=torch.zeros(1, dtype=torch.float64))
    state = torch_state(sho_specs, W, nodes[:, 0].numpy())
    lag = _sho(("cos",))
    S = to_np(tv.action(state, lag, rule=rule))
    manual = to_np(tv.integrate_values(tv.lagrangian_values(state, lag), rule=rule))
    assert np.allclose(S, manual, rtol=1e-14, atol=1e-14)


def test_integrate_values_of_raw_tensor() -> None:
    # integral_0^1 t^2 dt = 1/3.
    rule = gauss_legendre([(0.0, 1.0)], 8)
    nodes = quadrature_nodes(rule, like=torch.zeros(1, dtype=torch.float64))
    vals = nodes[:, 0] ** 2
    got = float(to_np(tv.integrate_values(vals, rule=rule)))
    assert abs(got - 1.0 / 3.0) < 1e-12


@pytest.mark.parametrize("dof", [("cos",), ("cos", "lin")])
def test_el_cross_backend(dof) -> None:  # type: ignore[no-untyped-def]
    # A coupled Lagrangian to exercise the mixed-Hessian einsum paths.
    def fn(q, qd, t):  # type: ignore[no-untyped-def]
        base = 0.5 * (qd**2).sum(-1) - 0.5 * W**2 * (q**2).sum(-1)
        if q.shape[-1] == 2:
            base = base + 0.5 * q[..., 0] * q[..., 1]
        return base

    lag = Lagrangian(fn, dof=dof)
    ts = torch_state(sho_specs, W, T)
    js = jax_state(sho_specs, W, T)
    t = to_np(tv.euler_lagrange_residual(ts, lag))
    j = to_np(jv.euler_lagrange_residual(js, lag))
    assert np.allclose(t, j, rtol=1e-12, atol=1e-12)


def test_action_cross_backend() -> None:
    rule = gauss_legendre([(0.0, 2.0)], 16)
    nodes_t = quadrature_nodes(rule, like=torch.zeros(1, dtype=torch.float64))[:, 0].numpy()
    lag = _sho(("cos",))
    ts = torch_state(sho_specs, W, nodes_t)
    js = jax_state(sho_specs, W, nodes_t)
    t = to_np(tv.action(ts, lag, rule=rule))
    j = to_np(jv.action(js, lag, rule=rule))
    assert np.allclose(t, j, rtol=1e-12, atol=1e-12)
