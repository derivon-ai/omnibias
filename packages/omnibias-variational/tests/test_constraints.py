# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Constrained variational calculus.

- Bead on the unit circle (holonomic ``g = x^2 + y^2 - 1 = 0``): uniform circular
  motion ``(cos wt, sin wt)`` of a free particle is a constrained solution with
  the constant multiplier ``lambda = -w^2/2`` -- the constrained residual and the
  constraint both vanish, while the *unconstrained* Euler-Lagrange residual does
  not (the constraint force matters).
- Catenary (isoperimetric): minimising the potential ``int y sqrt(1+y'^2) dx`` at
  fixed length ``int sqrt(1+y'^2) dx`` gives ``y = lambda + a cosh(x/a)``; the
  augmented-Lagrangian Euler-Lagrange residual vanishes on it (and the plain one
  does not).
- torch/jax parity.
"""

from __future__ import annotations

import numpy as np
import torch
from _traj import AnalyticTrajField, jax_state, sho_specs, to_np, torch_state
from omnibias.variational import Constraint, Lagrangian
from omnibias.variational.jax import ops as jv
from omnibias.variational.torch import ops as tv

W = 1.3
T = np.array([-0.7, -0.2, 0.4, 1.1, 1.9], dtype=np.float64)
LAM, ACAT = 0.5, 1.0  # catenary y = LAM + ACAT cosh(x / ACAT)
TCAT = np.array([-1.0, -0.4, 0.2, 0.7, 1.3], dtype=np.float64)


def _free(dof):  # type: ignore[no-untyped-def]
    return Lagrangian(lambda q, qd, t: 0.5 * (qd**2).sum(-1), dof=dof)


def _circle(q, t):  # type: ignore[no-untyped-def]
    # g = x^2 + y^2 - 1, with an explicit trailing constraint axis.
    return (q[..., 0] ** 2 + q[..., 1] ** 2 - 1.0)[..., None]


def _pe():  # type: ignore[no-untyped-def]
    # Potential energy density of a hanging chain: L = y sqrt(1 + y'^2).
    return Lagrangian(lambda y, yp, t: (y * (1.0 + yp**2) ** 0.5).sum(-1), dof=("y",))


def _length():  # type: ignore[no-untyped-def]
    # Arc-length integrand (the isoperimetric constraint): g = sqrt(1 + y'^2).
    return Lagrangian(lambda y, yp, t: ((1.0 + yp**2) ** 0.5).sum(-1), dof=("y",))


def _cat_specs(xp):  # type: ignore[no-untyped-def]
    return {
        "y": (
            lambda t: LAM + ACAT * xp.cosh(t / ACAT),
            lambda t: xp.sinh(t / ACAT),
            lambda t: (1.0 / ACAT) * xp.cosh(t / ACAT),
        )
    }


def _torch_cat_state(t):  # type: ignore[no-untyped-def]
    from omnibias.fields.torch import _ops_dispatch

    field = AnalyticTrajField(torch, _ops_dispatch, _cat_specs(torch))
    return field(torch.as_tensor(t[:, None], dtype=torch.float64))


def _jax_cat_state(t):  # type: ignore[no-untyped-def]
    import jax.numpy as jnp
    from omnibias.fields.jax import _ops_dispatch

    field = AnalyticTrajField(jnp, _ops_dispatch, _cat_specs(jnp))
    return field(jnp.asarray(t[:, None], dtype=jnp.float64))


def test_bead_on_circle_is_constrained_solution() -> None:
    state = torch_state(sho_specs, W, T)
    lag = _free(("cos", "sin"))
    constraint = Constraint(_circle, count=1)
    lam = torch.full((len(T), 1), -(W**2) / 2.0, dtype=torch.float64)

    eom, g = tv.constrained_euler_lagrange_residual(state, lag, constraint, lam)
    assert np.allclose(to_np(eom), 0.0, atol=1e-10)
    assert np.allclose(to_np(g), 0.0, atol=1e-12)

    # The constraint force is essential: the free EL residual is far from zero.
    el = tv.euler_lagrange_residual(state, lag)
    assert np.max(np.abs(to_np(el))) > 1e-2


def test_catenary_is_isoperimetric_extremal() -> None:
    state = _torch_cat_state(TCAT)
    aug = tv.augmented_lagrangian(_pe(), _length(), LAM)
    res = to_np(tv.euler_lagrange_residual(state, aug))
    assert np.allclose(res, 0.0, atol=1e-9)

    # Without the multiplier, y = LAM + cosh(x) is not a plain-PE extremal.
    plain = to_np(tv.euler_lagrange_residual(state, _pe()))
    assert np.max(np.abs(plain)) > 1e-2


def test_constrained_residual_cross_backend() -> None:
    constraint = Constraint(_circle, count=1)
    ts = torch_state(sho_specs, W, T)
    js = jax_state(sho_specs, W, T)
    lag = _free(("cos", "sin"))
    lam_t = torch.full((len(T), 1), -(W**2) / 2.0, dtype=torch.float64)
    import jax.numpy as jnp

    lam_j = jnp.full((len(T), 1), -(W**2) / 2.0, dtype=jnp.float64)
    eom_t, g_t = tv.constrained_euler_lagrange_residual(ts, lag, constraint, lam_t)
    eom_j, g_j = jv.constrained_euler_lagrange_residual(js, lag, constraint, lam_j)
    assert np.allclose(to_np(eom_t), to_np(eom_j), rtol=1e-12, atol=1e-12)
    assert np.allclose(to_np(g_t), to_np(g_j), rtol=1e-12, atol=1e-12)


def test_augmented_lagrangian_cross_backend() -> None:
    aug = tv.augmented_lagrangian(_pe(), _length(), LAM)
    ts = _torch_cat_state(TCAT)
    js = _jax_cat_state(TCAT)
    assert np.allclose(
        to_np(tv.euler_lagrange_residual(ts, aug)),
        to_np(jv.euler_lagrange_residual(js, aug)),
        rtol=1e-12, atol=1e-12,
    )
