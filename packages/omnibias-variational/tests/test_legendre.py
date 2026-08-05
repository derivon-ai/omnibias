# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""The Legendre transform / Hamiltonian bridge.

- ``velocity_from_momentum`` inverts ``p = dL/dqdot`` (round trip qdot -> p -> qdot).
- ``legendre_transform`` gives the true ``H(q, p)`` (``= 1/2 p^2 + 1/2 w^2 q^2``
  for the oscillator) and satisfies the Legendre involution ``L = p.qdot - H``.
- On a trajectory it agrees with the along-trajectory ``hamiltonian`` energy.
- ``canonical_equations`` of ``hamiltonian_from_lagrangian(L)`` reproduce the
  Lagrangian flow: ``qdot = dH/dp`` recovers the velocity and ``pdot = -dH/dq``
  equals ``dL/dq``.
All float64, torch/jax parity checked to rtol=1e-12.
"""

from __future__ import annotations

import numpy as np
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
_P = np.array([[0.6], [-0.3], [1.4]], dtype=np.float64)
_T0 = np.zeros((3, 1), dtype=np.float64)


def _sho(dof=("q",)):  # type: ignore[no-untyped-def]
    return Lagrangian(
        lambda q, qd, t: 0.5 * (qd**2).sum(-1) - 0.5 * W**2 * (q**2).sum(-1), dof=dof,
    )


def _posmass():  # type: ignore[no-untyped-def]
    return Lagrangian(
        lambda q, qd, t: 0.5 * ((1.0 + q**2) * qd**2).sum(-1) - 0.5 * (q**2).sum(-1),
        dof=("q",),
    )


def _tt(a):  # type: ignore[no-untyped-def]
    return torch.tensor(a, dtype=DT)


def test_velocity_momentum_round_trip() -> None:
    q, qd, t = _tt(_Q), _tt(_QD), _tt(_T0)
    for lag in (_sho(), _posmass()):
        p = tv.momentum(lag, q, qd, t)
        qd_back = tv.velocity_from_momentum(lag, q, p, t)
        assert np.allclose(to_np(qd_back), _QD, atol=1e-12)


def test_legendre_transform_harmonic_value() -> None:
    lag = _sho()
    q, p, t = _tt(_Q), _tt(_P), _tt(_T0)
    h = tv.legendre_transform(lag, q, p, t)
    expected = 0.5 * (_P**2).sum(-1) + 0.5 * W**2 * (_Q**2).sum(-1)
    assert np.allclose(to_np(h), expected, atol=1e-12)


def test_legendre_involution() -> None:
    # L(q, qdot) = p . qdot - H(q, p) with p = dL/dqdot.
    lag = _posmass()
    q, qd, t = _tt(_Q), _tt(_QD), _tt(_T0)
    p = tv.momentum(lag, q, qd, t)
    h = tv.legendre_transform(lag, q, p, t)
    lrec = (p * qd).sum(-1) - h
    lval = lag.fn(q, qd, t)
    assert np.allclose(to_np(lrec), to_np(lval), atol=1e-12)


def test_hamiltonian_equals_trajectory_energy() -> None:
    lag = _sho(("cos",))
    state = torch_state(sho_specs, W, T)
    q, qd, _qddot, t = tv.trajectory(state, lag)
    p = tv.momentum(lag, q, qd, t)
    h_phase = tv.legendre_transform(lag, q, p, t)
    h_traj = tv.hamiltonian(state, lag)
    assert np.allclose(to_np(h_phase), to_np(h_traj), atol=1e-10)


def test_canonical_equations_reproduce_forward_dynamics() -> None:
    q, qd, t = _tt(_Q), _tt(_QD), _tt(_T0)
    for lag in (_sho(), _posmass()):
        ham = tv.hamiltonian_from_lagrangian(lag)
        p = tv.momentum(lag, q, qd, t)
        qdot_c, pdot_c = tv.canonical_equations(ham, q, p, t)
        g_q, _g_v = tv.lagrangian_partials(lag, q, qd, t)
        assert np.allclose(to_np(qdot_c), _QD, atol=1e-10)      # dH/dp = qdot
        assert np.allclose(to_np(pdot_c), to_np(g_q), atol=1e-10)  # -dH/dq = dL/dq


def test_canonical_pdot_is_acceleration_for_unit_mass() -> None:
    # For unit-mass L, p = qdot so pdot = qddot = acceleration.
    lag = _sho()
    q, qd, t = _tt(_Q), _tt(_QD), _tt(_T0)
    ham = tv.hamiltonian_from_lagrangian(lag)
    p = tv.momentum(lag, q, qd, t)
    _qdot_c, pdot_c = tv.canonical_equations(ham, q, p, t)
    acc = tv.acceleration(lag, q, qd, t)
    assert np.allclose(to_np(pdot_c), to_np(acc), atol=1e-10)


def test_legendre_cross_backend() -> None:
    import jax.numpy as jnp

    qj, pj, tj = jnp.asarray(_Q), jnp.asarray(_P), jnp.asarray(_T0)
    qt, pt, tt = _tt(_Q), _tt(_P), _tt(_T0)
    for lag in (_sho(), _posmass()):
        assert np.allclose(
            to_np(tv.velocity_from_momentum(lag, qt, pt, tt)),
            to_np(jv.velocity_from_momentum(lag, qj, pj, tj)),
            rtol=1e-12, atol=1e-12,
        )
        assert np.allclose(
            to_np(tv.legendre_transform(lag, qt, pt, tt)),
            to_np(jv.legendre_transform(lag, qj, pj, tj)),
            rtol=1e-12, atol=1e-12,
        )
        ham_t = tv.hamiltonian_from_lagrangian(lag)
        ham_j = jv.hamiltonian_from_lagrangian(lag)
        qc_t, pc_t = tv.canonical_equations(ham_t, qt, pt, tt)
        qc_j, pc_j = jv.canonical_equations(ham_j, qj, pj, tj)
        assert np.allclose(to_np(qc_t), to_np(qc_j), rtol=1e-12, atol=1e-12)
        assert np.allclose(to_np(pc_t), to_np(pc_j), rtol=1e-12, atol=1e-12)
