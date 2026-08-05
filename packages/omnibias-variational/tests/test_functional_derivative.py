# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""The functional (variational) derivative via arbitrary-order Euler-Poisson.

- Pais-Uhlenbeck oscillator (order 2): ``delta S / delta q = q'''' +
  (w1^2 + w2^2) q'' + w1^2 w2^2 q`` -- zero on a normal mode, and equal to the
  hand-built fourth-order operator off a mode.
- Static Euler-Bernoulli beam (order 2): ``delta S / delta y = EI y'''' - rho``.
- Order-1 consistency: ``functional_derivative == -euler_lagrange_residual``.
All float64, torch/jax parity.
"""

from __future__ import annotations

import numpy as np
from _traj import to_np, torch_state
from omnibias.fields.torch.ops.basic import stack_components, vector_derivative
from omnibias.variational import Lagrangian
from omnibias.variational.torch import ops as tv

W1, W2 = 1.0, 1.7
W_OFF = 0.53  # neither normal mode
T = np.array([0.1, 0.4, 0.85, 1.3, 1.9], dtype=np.float64)


def _cos_specs(xp, omega):  # type: ignore[no-untyped-def]
    return {
        "q": (
            lambda t: xp.cos(omega * t),
            lambda t: -omega * xp.sin(omega * t),
            lambda t: -(omega**2) * xp.cos(omega * t),
            lambda t: omega**3 * xp.sin(omega * t),
            lambda t: omega**4 * xp.cos(omega * t),
        )
    }


def _sin_specs_y(xp, k):  # type: ignore[no-untyped-def]
    return {
        "y": (
            lambda t: xp.sin(k * t),
            lambda t: k * xp.cos(k * t),
            lambda t: -(k**2) * xp.sin(k * t),
            lambda t: -(k**3) * xp.cos(k * t),
            lambda t: k**4 * xp.sin(k * t),
        )
    }


def _pais_uhlenbeck(w1, w2):  # type: ignore[no-untyped-def]
    def fn(q, qd, qdd, t):  # type: ignore[no-untyped-def]
        return (
            0.5 * (qdd**2).sum(-1)
            - 0.5 * (w1**2 + w2**2) * (qd**2).sum(-1)
            + 0.5 * w1**2 * w2**2 * (q**2).sum(-1)
        )

    return Lagrangian(fn, dof=("q",), order=2)


def _beam(ei, rho):  # type: ignore[no-untyped-def]
    def fn(y, yp, ypp, t):  # type: ignore[no-untyped-def]
        return 0.5 * ei * (ypp**2).sum(-1) - rho * y.sum(-1)

    return Lagrangian(fn, dof=("y",), order=2)


def test_pais_uhlenbeck_zero_on_normal_mode() -> None:
    lag = _pais_uhlenbeck(W1, W2)
    for mode in (W1, W2):
        state = torch_state(_cos_specs, mode, T)
        fd = to_np(tv.functional_derivative(state, lag))
        assert np.allclose(fd, 0.0, atol=1e-9)


def test_pais_uhlenbeck_matches_fourth_order_operator() -> None:
    lag = _pais_uhlenbeck(W1, W2)
    state = torch_state(_cos_specs, W_OFF, T)
    fd = to_np(tv.functional_derivative(state, lag))[:, 0]
    q = to_np(stack_components(state, ("q",)))[:, 0]
    q2 = to_np(vector_derivative(state, ("q",), axis="t", order=2))[:, 0]
    q4 = to_np(vector_derivative(state, ("q",), axis="t", order=4))[:, 0]
    manual = q4 + (W1**2 + W2**2) * q2 + W1**2 * W2**2 * q
    assert np.max(np.abs(manual)) > 1e-3  # genuinely off a mode
    assert np.allclose(fd, manual, atol=1e-10)


def test_euler_bernoulli_beam_operator() -> None:
    # Static beam: L = 1/2 EI (y'')^2 - rho y  ->  delta S/delta y = EI y'''' - rho.
    # The "time" axis plays the role of the beam coordinate x; y(x) = sin(k x).
    ei, rho, k = 2.3, 0.7, 1.1
    lag = _beam(ei, rho)
    state = torch_state(_sin_specs_y, k, T)
    fd = to_np(tv.functional_derivative(state, lag))[:, 0]
    y4 = to_np(vector_derivative(state, ("y",), axis="t", order=4))[:, 0]
    manual = ei * y4 - rho
    assert np.max(np.abs(manual)) > 1e-3
    assert np.allclose(fd, manual, atol=1e-10)


def test_order1_consistency_with_euler_lagrange() -> None:
    lag = Lagrangian(lambda q, qd, t: 0.5 * (qd**2).sum(-1) - 0.5 * W1**2 * (q**2).sum(-1), dof=("q",))
    state = torch_state(_cos_specs, W_OFF, T)  # off the SHO frequency -> nonzero
    fd = to_np(tv.functional_derivative(state, lag))
    el = to_np(tv.euler_lagrange_residual(state, lag))
    assert np.max(np.abs(el)) > 1e-3
    assert np.allclose(fd, -el, rtol=1e-12, atol=1e-12)
    # and delta S/delta q = -(qddot + w^2 q)
    q = to_np(stack_components(state, ("q",)))
    q2 = to_np(vector_derivative(state, ("q",), axis="t", order=2))
    assert np.allclose(fd, -(q2 + W1**2 * q), atol=1e-10)


def test_functional_derivative_cross_backend() -> None:
    from _traj import jax_state
    from omnibias.variational.jax import ops as jv

    lag_t = _pais_uhlenbeck(W1, W2)
    lag_j = _pais_uhlenbeck(W1, W2)
    ts = torch_state(_cos_specs, W_OFF, T)
    js = jax_state(_cos_specs, W_OFF, T)
    assert np.allclose(
        to_np(tv.functional_derivative(ts, lag_t)),
        to_np(jv.functional_derivative(js, lag_j)),
        rtol=1e-12, atol=1e-12,
    )
