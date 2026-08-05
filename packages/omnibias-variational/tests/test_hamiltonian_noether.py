# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Hamiltonian / energy and Noether charges, validated by conservation laws.

- Harmonic oscillator ``q = cos(w t)``: energy ``H = 1/2 qdot^2 + 1/2 w^2 q^2``
  is the constant ``1/2 w^2``; the energy residual is zero.
- Free particle ``q = t``, ``L = 1/2 qdot^2``: momentum ``p = qdot = 1`` and the
  translation Noether charge is conserved.
- 2-D isotropic oscillator on circular motion ``(cos, sin)``: the rotational
  Noether charge (angular momentum) is the constant ``w``.
All float64, torch/jax parity checked.
"""

from __future__ import annotations

import numpy as np
import torch
from _traj import jax_state, sho_specs, to_np, torch_state
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


def test_energy_of_harmonic_solution_is_constant() -> None:
    state = torch_state(sho_specs, W, T)
    H = to_np(tv.hamiltonian(state, _sho(("cos",))))
    assert np.allclose(H, 0.5 * W**2, atol=1e-10)


def test_energy_residual_zero_on_solution() -> None:
    state = torch_state(sho_specs, W, T)
    res = to_np(tv.hamiltons_equations_residual(state, _sho(("cos",))))
    assert np.allclose(res, 0.0, atol=1e-10)


def test_energy_residual_offsolution_matches_qdot_dot_el() -> None:
    # q = t: EL = w^2 t, qdot = 1  ->  residual = w^2 t.
    state = torch_state(sho_specs, W, T)
    res = to_np(tv.hamiltons_equations_residual(state, _sho(("lin",))))
    assert np.allclose(res, W**2 * T, atol=1e-10)


def test_free_particle_momentum_is_conserved() -> None:
    state = torch_state(sho_specs, W, T)
    p = to_np(tv.conjugate_momentum(state, _free(("lin",))))
    assert np.allclose(p[:, 0], 1.0, atol=1e-12)


def test_translation_noether_charge_is_momentum() -> None:
    state = torch_state(sho_specs, W, T)
    lag = _free(("lin",))
    gen = torch.ones(len(T), 1, dtype=torch.float64)  # translation X = 1
    Q = to_np(tv.noether_charge(state, lag, gen))
    p = to_np(tv.conjugate_momentum(state, lag))
    assert np.allclose(Q, p[:, 0], atol=1e-12)
    assert np.allclose(Q, 1.0, atol=1e-12)


def test_rotational_noether_charge_is_angular_momentum() -> None:
    # 2-D isotropic oscillator on (cos, sin); rotation X = (-q_y, q_x).
    state = torch_state(sho_specs, W, T)
    lag = _sho(("cos", "sin"))

    def rotation(q, qd, t):  # type: ignore[no-untyped-def]
        return torch.stack([-q[..., 1], q[..., 0]], dim=-1)

    Q = to_np(tv.noether_charge(state, lag, rotation))
    assert np.allclose(Q, W, atol=1e-10)


def test_hamiltonian_noether_cross_backend() -> None:
    lag = _sho(("cos", "sin"))
    ts, js = torch_state(sho_specs, W, T), jax_state(sho_specs, W, T)

    def rot_t(q, qd, t):  # type: ignore[no-untyped-def]
        return torch.stack([-q[..., 1], q[..., 0]], dim=-1)

    def rot_j(q, qd, t):  # type: ignore[no-untyped-def]
        import jax.numpy as jnp

        return jnp.stack([-q[..., 1], q[..., 0]], axis=-1)

    assert np.allclose(
        to_np(tv.hamiltonian(ts, lag)), to_np(jv.hamiltonian(js, lag)),
        rtol=1e-12, atol=1e-12,
    )
    assert np.allclose(
        to_np(tv.conjugate_momentum(ts, lag)), to_np(jv.conjugate_momentum(js, lag)),
        rtol=1e-12, atol=1e-12,
    )
    assert np.allclose(
        to_np(tv.noether_charge(ts, lag, rot_t)),
        to_np(jv.noether_charge(js, lag, rot_j)),
        rtol=1e-12, atol=1e-12,
    )
