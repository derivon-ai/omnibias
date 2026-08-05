# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Forward Lagrangian dynamics: the equations-of-motion solve.

- Harmonic oscillator ``L = 1/2 qdot^2 - 1/2 w^2 q^2``: ``acceleration == -w^2 q``,
  ``mass_matrix == I``.
- Anisotropic constant mass ``L = 1/2 qdot^T A qdot - 1/2 q^T K q``:
  ``acceleration == A^{-1}(-K q)``, ``mass_matrix == A``.
- Position-dependent scalar mass ``L = 1/2 (1 + q^2) qdot^2 - 1/2 q^2``: the
  acceleration matches the hand-derived Euler-Lagrange equation.
- ``inverse_dynamics`` inverts ``acceleration`` (round trip ~ 0) and, on a
  trajectory, equals the existing ``euler_lagrange_residual`` (the two are duals).
All float64, torch/jax parity checked to rtol=1e-12.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from _traj import sho_specs, to_np, torch_state
from omnibias.variational import Lagrangian
from omnibias.variational.jax import ops as jv
from omnibias.variational.torch import ops as tv

DT = torch.float64
W = 1.3
T = np.array([-0.7, -0.2, 0.4, 1.1, 1.9], dtype=np.float64)

_Q = np.array([[0.3], [-1.0], [2.0]], dtype=np.float64)
_QD = np.array([[1.0], [0.5], [-0.2]], dtype=np.float64)
_T0 = np.zeros((3, 1), dtype=np.float64)


def _sho(dof=("q",)):  # type: ignore[no-untyped-def]
    return Lagrangian(
        lambda q, qd, t: 0.5 * (qd**2).sum(-1) - 0.5 * W**2 * (q**2).sum(-1), dof=dof,
    )


def _posmass():  # type: ignore[no-untyped-def]
    # L = 1/2 (1 + q^2) qdot^2 - 1/2 q^2  (backend-generic: only ** and *).
    return Lagrangian(
        lambda q, qd, t: 0.5 * ((1.0 + q**2) * qd**2).sum(-1) - 0.5 * (q**2).sum(-1),
        dof=("q",),
    )


def _tt(a):  # type: ignore[no-untyped-def]
    return torch.tensor(a, dtype=DT)


def test_harmonic_acceleration_mass_force() -> None:
    lag = _sho()
    q, qd, t = _tt(_Q), _tt(_QD), _tt(_T0)
    acc = tv.acceleration(lag, q, qd, t)
    assert np.allclose(to_np(acc), -W**2 * _Q, atol=1e-10)
    m = tv.mass_matrix(lag, q, qd, t)
    assert m.shape == (3, 1, 1)
    assert np.allclose(to_np(m), 1.0, atol=1e-12)
    force = tv.generalized_force(lag, q, qd, t)
    assert np.allclose(to_np(force), -W**2 * _Q, atol=1e-10)


def test_anisotropic_constant_mass() -> None:
    a_mat = torch.tensor([[2.0, 0.5], [0.5, 1.0]], dtype=DT)
    k_mat = torch.tensor([[3.0, 0.2], [0.2, 1.5]], dtype=DT)

    def fn(q, qd, t):  # type: ignore[no-untyped-def]
        return 0.5 * (qd * (qd @ a_mat)).sum(-1) - 0.5 * (q * (q @ k_mat)).sum(-1)

    lag = Lagrangian(fn, dof=("x", "y"))
    q = torch.tensor([[0.3, -0.4], [1.0, 0.7]], dtype=DT)
    qd = torch.tensor([[1.0, 0.2], [-0.5, 0.9]], dtype=DT)
    t = torch.zeros(2, 1, dtype=DT)

    m = tv.mass_matrix(lag, q, qd, t)
    assert np.allclose(to_np(m), to_np(a_mat)[None], atol=1e-10)
    acc = tv.acceleration(lag, q, qd, t)
    expected = torch.linalg.solve(a_mat, (-(q @ k_mat)).T).T
    assert np.allclose(to_np(acc), to_np(expected), atol=1e-10)


def test_position_dependent_mass_matches_manual() -> None:
    lag = _posmass()
    q, qd, t = _tt(_Q), _tt(_QD), _tt(_T0)
    acc = tv.acceleration(lag, q, qd, t)
    # L = 1/2 m(q) qd^2 - V,  m = 1 + q^2, V = 1/2 q^2
    #   => qddot = -(1/2 m'(q) qd^2 + V'(q)) / m(q) = -q (qd^2 + 1) / (1 + q^2).
    manual = -_Q * (_QD**2 + 1.0) / (1.0 + _Q**2)
    assert np.allclose(to_np(acc), manual, atol=1e-12)


def test_inverse_dynamics_round_trip_is_zero() -> None:
    lag = _posmass()
    q, qd, t = _tt(_Q), _tt(_QD), _tt(_T0)
    acc = tv.acceleration(lag, q, qd, t)
    tau = tv.inverse_dynamics(lag, q, qd, acc, t)
    assert np.allclose(to_np(tau), 0.0, atol=1e-12)


def test_dynamics_rhs_pairs_velocity_and_acceleration() -> None:
    lag = _posmass()
    q, qd, t = _tt(_Q), _tt(_QD), _tt(_T0)
    v, acc = tv.dynamics_rhs(lag, q, qd, t)
    assert np.allclose(to_np(v), _QD, atol=1e-12)
    assert np.allclose(to_np(acc), to_np(tv.acceleration(lag, q, qd, t)), atol=1e-12)


def test_duality_residual_equals_inverse_dynamics() -> None:
    # q = t (off-solution): euler_lagrange_residual must equal inverse_dynamics
    # evaluated with the trajectory's own closed-form qddot.
    lag = _sho(("lin",))
    state = torch_state(sho_specs, W, T)
    q, qd, qddot, t = tv.trajectory(state, lag)
    el = tv.euler_lagrange_residual(state, lag)
    inv = tv.inverse_dynamics(lag, q, qd, qddot, t)
    assert np.allclose(to_np(el), to_np(inv), atol=1e-12)
    assert np.allclose(to_np(el), W**2 * T[:, None], atol=1e-10)


def test_predicted_acceleration_matches_solution() -> None:
    # On q* = cos(w t) the equations of motion hold, so the Lagrangian's
    # predicted acceleration equals the trajectory's own closed-form qddot.
    lag = _sho(("cos",))
    state = torch_state(sho_specs, W, T)
    _q, _qd, qddot, _t = tv.trajectory(state, lag)
    pred = tv.predicted_acceleration(state, lag)
    assert np.allclose(to_np(pred), to_np(qddot), atol=1e-10)


def test_higher_order_not_implemented() -> None:
    lag = Lagrangian(
        lambda q, q1, q2, t: 0.5 * (q1**2).sum(-1) - 0.5 * (q**2).sum(-1),
        dof=("q",), order=2,
    )
    q, qd, t = _tt(_Q), _tt(_QD), _tt(_T0)
    with pytest.raises(NotImplementedError):
        tv.acceleration(lag, q, qd, t)
    with pytest.raises(NotImplementedError):
        tv.mass_matrix(lag, q, qd, t)


def test_dynamics_cross_backend() -> None:
    import jax.numpy as jnp

    qj, qdj, tj = jnp.asarray(_Q), jnp.asarray(_QD), jnp.asarray(_T0)
    qt, qdt, tt = _tt(_Q), _tt(_QD), _tt(_T0)
    for lag in (_sho(), _posmass()):
        assert np.allclose(
            to_np(tv.acceleration(lag, qt, qdt, tt)),
            to_np(jv.acceleration(lag, qj, qdj, tj)),
            rtol=1e-12, atol=1e-12,
        )
        assert np.allclose(
            to_np(tv.mass_matrix(lag, qt, qdt, tt)),
            to_np(jv.mass_matrix(lag, qj, qdj, tj)),
            rtol=1e-12, atol=1e-12,
        )
        assert np.allclose(
            to_np(tv.generalized_force(lag, qt, qdt, tt)),
            to_np(jv.generalized_force(lag, qj, qdj, tj)),
            rtol=1e-12, atol=1e-12,
        )
        acc_t = tv.acceleration(lag, qt, qdt, tt)
        acc_j = jv.acceleration(lag, qj, qdj, tj)
        assert np.allclose(
            to_np(tv.inverse_dynamics(lag, qt, qdt, acc_t, tt)),
            to_np(jv.inverse_dynamics(lag, qj, qdj, acc_j, tj)),
            rtol=1e-12, atol=1e-11,
        )
