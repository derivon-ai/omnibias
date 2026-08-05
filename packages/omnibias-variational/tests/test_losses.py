# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Least-action training losses: direct (Ritz) and indirect (Euler-Lagrange).

- ``euler_lagrange_loss`` is zero on a solution and positive off it.
- Brachistochrone: the cycloid satisfies the brachistochrone Euler-Lagrange
  equation (indirect check on the true minimiser of the descent-time functional).
- Direct method: minimising ``action_minimization_loss`` over a Ritz (Fourier)
  trajectory recovers the harmonic-oscillator boundary-value solution.
All float64.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from _traj import AnalyticTrajField, sho_specs, to_np, torch_state
from omnibias.fields._core.components import ComponentSpec
from omnibias.fields._core.coords import CoordinateSpec
from omnibias.fields._core.quadrature import gauss_legendre
from omnibias.fields._core.sigma_cache import SigmaCache
from omnibias.fields._core.state import FieldState
from omnibias.fields.torch import _ops_dispatch
from omnibias.fields.torch.ops.integral import quadrature_nodes
from omnibias.variational import Lagrangian
from omnibias.variational.torch import ops as tv

W = 1.3
T = np.array([-0.7, -0.2, 0.4, 1.1, 1.9], dtype=np.float64)


def _sho(dof):  # type: ignore[no-untyped-def]
    return Lagrangian(
        lambda q, qd, t: 0.5 * (qd**2).sum(-1) - 0.5 * W**2 * (q**2).sum(-1),
        dof=dof,
    )


def test_action_minimization_loss_equals_action() -> None:
    rule = gauss_legendre([(0.0, 1.0)], 8)
    nodes = quadrature_nodes(rule, like=torch.zeros(1, dtype=torch.float64))
    state = torch_state(sho_specs, W, nodes[:, 0].numpy())
    lag = _sho(("cos",))
    loss = to_np(tv.action_minimization_loss(state, lag, rule=rule))
    act = to_np(tv.action(state, lag, rule=rule))
    assert np.allclose(loss, act)


def test_euler_lagrange_loss_zero_on_solution_positive_off() -> None:
    state = torch_state(sho_specs, W, T)
    on = float(to_np(tv.euler_lagrange_loss(state, _sho(("cos",)))))
    off = float(to_np(tv.euler_lagrange_loss(state, _sho(("lin",)))))
    assert on < 1e-18
    assert off > 1e-3


def test_brachistochrone_cycloid_satisfies_euler_lagrange() -> None:
    # Cycloid x = R(tau - sin tau), y = R(1 - cos tau); the brachistochrone
    # Lagrangian L(y, y') = sqrt((1 + y'^2) / y). The cycloid zeroes its EL.
    tau = np.linspace(0.5, 1.7 * math.pi, 7)
    R = 1.0
    x = R * (tau - np.sin(tau))
    y = R * (1.0 - np.cos(tau))
    yp = np.sin(tau) / (1.0 - np.cos(tau))
    ypp = -1.0 / (R * (1.0 - np.cos(tau)) ** 2)
    yt = torch.as_tensor(y, dtype=torch.float64)
    ypt = torch.as_tensor(yp, dtype=torch.float64)
    yppt = torch.as_tensor(ypp, dtype=torch.float64)
    specs = {"y": (lambda t: yt, lambda t: ypt, lambda t: yppt)}
    field = AnalyticTrajField(torch, _ops_dispatch, specs)
    state = field(torch.as_tensor(x[:, None], dtype=torch.float64))
    lag = Lagrangian(
        lambda q, qd, t: torch.sqrt((1.0 + qd[..., 0] ** 2) / q[..., 0]), dof=("y",),
    )
    res = to_np(tv.euler_lagrange_residual(state, lag))
    assert np.allclose(res, 0.0, atol=1e-9)


class _RitzTrajField:
    """Trainable trajectory ``q(t) = boundary + sum_k a_k sin(k pi s)``, ``s in [0, 1]``."""

    _omnibias_dispatch = "spectral"

    def __init__(self, coeffs, t0, t1, q_left, q_right):  # type: ignore[no-untyped-def]
        self.coeffs = coeffs
        self.t0, self.t1 = t0, t1
        self.q_left, self.q_right = q_left, q_right
        self.coordinate_spec = CoordinateSpec(("t",), time_axis="t")
        self.components = ComponentSpec(("q",))
        self._ops = _ops_dispatch

    def evaluate(self, coords):  # type: ignore[no-untyped-def]
        return FieldState(
            coords=coords, field=self, components=self.components,
            coordinate_spec=self.coordinate_spec, ops=self._ops,
            sigma_cache=SigmaCache(z=coords),
        )

    __call__ = evaluate

    def _s(self, state):  # type: ignore[no-untyped-def]
        return (state.coords[:, 0] - self.t0) / (self.t1 - self.t0)

    def value_component(self, state, name):  # type: ignore[no-untyped-def]
        s = self._s(state)
        base = self.q_left + (self.q_right - self.q_left) * s
        k = torch.arange(1, self.coeffs.shape[0] + 1, dtype=torch.float64)
        modes = torch.sin(k[None, :] * math.pi * s[:, None]) @ self.coeffs
        return base + modes

    def derivative(self, state, name, *, axis, order):  # type: ignore[no-untyped-def]
        s = self._s(state)
        length = self.t1 - self.t0
        k = torch.arange(1, self.coeffs.shape[0] + 1, dtype=torch.float64)
        w = k * math.pi / length
        if order == 1:
            dbase = (self.q_right - self.q_left) / length
            dmodes = (torch.cos(k[None, :] * math.pi * s[:, None]) * w[None, :]) @ self.coeffs
            return dbase + dmodes
        if order == 2:
            d2 = -(torch.sin(k[None, :] * math.pi * s[:, None]) * (w**2)[None, :]) @ self.coeffs
            return d2
        raise NotImplementedError

    def mixed_partial(self, state, name, axes, orders):  # type: ignore[no-untyped-def]
        raise NotImplementedError


def test_direct_method_recovers_harmonic_bvp() -> None:
    t0, t1, q_left, q_right = 0.0, 1.0, 1.0, 0.3
    coeffs = torch.zeros(6, dtype=torch.float64, requires_grad=True)
    field = _RitzTrajField(coeffs, t0, t1, q_left, q_right)
    lag = _sho(("q",))
    rule = gauss_legendre([(t0, t1)], 32)
    nodes = quadrature_nodes(rule, like=torch.zeros(1, dtype=torch.float64))

    opt = torch.optim.LBFGS([coeffs], lr=1.0, max_iter=300, line_search_fn="strong_wolfe")

    def closure():  # type: ignore[no-untyped-def]
        opt.zero_grad()
        loss = tv.action_minimization_loss(field(nodes), lag, rule=rule)
        loss.backward()
        return loss

    opt.step(closure)

    # Analytic BVP solution q(t) = A cos(w t) + B sin(w t), q(0)=1, q(1)=0.3.
    a_coef = q_left
    b_coef = (q_right - a_coef * math.cos(W)) / math.sin(W)
    tt = nodes[:, 0]
    exact = a_coef * torch.cos(W * tt) + b_coef * torch.sin(W * tt)
    got = field.value_component(field(nodes), "q").detach()
    assert torch.max(torch.abs(got - exact)).item() < 5e-3


def test_euler_lagrange_loss_cross_backend() -> None:
    import jax.numpy as jnp
    from _traj import jax_state
    from omnibias.variational.jax import ops as jv

    lag = _sho(("cos", "lin"))
    ts, js = torch_state(sho_specs, W, T), jax_state(sho_specs, W, T)
    t = to_np(tv.euler_lagrange_loss(ts, lag))
    j = to_np(jv.euler_lagrange_loss(js, lag))
    assert np.allclose(t, j, rtol=1e-12, atol=1e-12)
    _ = jnp  # keep import used
