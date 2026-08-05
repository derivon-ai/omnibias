# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""CBF-row builders: analytic cross-checks, learned-Lagrangian path, parity."""

from __future__ import annotations

import numpy as np
from omnibias.control import CBFSpec

A1, A2, AMAX, R = 2.0, 2.0, 2.5, 1.0


def _analytic_double_integrator_rows(X):
    """Reference exponential-CBF row for the unit-mass double integrator (disc @ 0)."""
    p, v = X[:, :2], X[:, 2:]
    d = p
    b = np.sum(d * d, axis=1) - R * R
    dv = np.sum(d * v, axis=1)
    v2 = np.sum(v * v, axis=1)
    h_obs = 2 * v2 + 2 * (A1 + A2) * dv + A1 * A2 * b
    g_obs = -2 * d
    return g_obs, h_obs


def test_control_affine_matches_analytic_jax():
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    import omnibias.control.jax as cj

    def f(x):
        return jnp.array([x[2], x[3], 0.0, 0.0])

    def g(x):
        return jnp.array([[0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])

    def bar(x):
        return x[0] ** 2 + x[1] ** 2 - R * R

    X = np.random.default_rng(0).standard_normal((6, 4))
    G, h = cj.control_affine_cbf_rows(f, g, bar, jnp.asarray(X), CBFSpec((A1, A2), a_max=AMAX))
    g_obs, h_obs = _analytic_double_integrator_rows(X)
    assert G.shape == (6, 5, 2) and h.shape == (6, 5)
    assert np.max(np.abs(np.asarray(G[:, 0, :]) - g_obs)) < 1e-10
    assert np.max(np.abs(np.asarray(h[:, 0]) - h_obs)) < 1e-10
    assert np.max(np.abs(np.asarray(h[:, 1:]) - AMAX)) < 1e-12   # actuator box


def test_relative_degree_one():
    """Relative-degree-1 barrier ``h = c - x0`` gives ``L_f h + L_g h a + alpha h``."""
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    import omnibias.control.jax as cj

    # single integrator x_dot = a (n = d = 2); barrier h(x) = 1 - x0 (rel deg 1).
    def f(x):
        return jnp.zeros_like(x)

    def g(x):
        return jnp.eye(2)

    def bar(x):
        return 1.0 - x[0]

    X = np.random.default_rng(1).standard_normal((4, 2))
    G, h = cj.control_affine_cbf_rows(f, g, bar, jnp.asarray(X), CBFSpec((3.0,)))
    # L_f h = 0; L_g h = grad h . g = [-1, 0] -> row = -L_g h = [1, 0]; h_val = 3 (1 - x0)
    assert np.max(np.abs(np.asarray(G[:, 0, :]) - np.array([1.0, 0.0]))) < 1e-10
    assert np.max(np.abs(np.asarray(h[:, 0]) - 3.0 * (1.0 - X[:, 0]))) < 1e-10


def test_lagrangian_equals_control_affine():
    """A free-particle Lagrangian (mass M) reproduces the ``g = M^{-1}B`` rows."""
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    import omnibias.control.jax as cj
    from omnibias.variational import Lagrangian

    M = jnp.array([[1.0, 0.6], [0.6, 1.0]])
    Minv = jnp.linalg.inv(M)

    def Lfn(q, qd, t):
        return 0.5 * qd @ (M @ qd)

    L = Lagrangian(Lfn, dof=("q",))

    def bar(x):
        return x[0] ** 2 + x[1] ** 2 - R * R

    rng = np.random.default_rng(2)
    q = jnp.asarray(rng.standard_normal((5, 2)))
    qd = jnp.asarray(rng.standard_normal((5, 2)))
    t = jnp.zeros((5, 1))
    spec = CBFSpec((A1, A2), a_max=AMAX)
    Gl, hl = cj.lagrangian_cbf_rows(L, jnp.eye(2), bar, q, qd, t, spec)

    # reference: control-affine with f = [qd; 0], g = [0; M^{-1}]
    def f(x):
        return jnp.concatenate([x[2:], jnp.zeros(2)])

    def g(x):
        return jnp.concatenate([jnp.zeros((2, 2)), Minv], axis=0)

    X = jnp.concatenate([q, qd], axis=1)
    Gc, hc = cj.control_affine_cbf_rows(f, g, bar, X, spec)
    assert np.max(np.abs(np.asarray(Gl - Gc))) < 1e-9
    assert np.max(np.abs(np.asarray(hl - hc))) < 1e-9


def test_builders_torch_jax_parity():
    import jax

    jax.config.update("jax_enable_x64", True)
    import torch

    torch.set_default_dtype(torch.float64)
    import jax.numpy as jnp
    import omnibias.control.jax as cj
    import omnibias.control.torch as ct

    X = np.random.default_rng(4).standard_normal((6, 4))
    spec = CBFSpec((A1, A2), a_max=AMAX)

    def fj(x):
        return jnp.array([x[2], x[3], 0.0, 0.0])

    def gj(x):
        return jnp.array([[0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])

    def barj(x):
        return x[0] ** 2 + x[1] ** 2 - R * R

    G, h = cj.control_affine_cbf_rows(fj, gj, barj, jnp.asarray(X), spec)

    def ft(x):
        z = torch.zeros((), dtype=x.dtype)
        return torch.stack([x[2], x[3], z, z])

    def gt(x):
        return torch.tensor([[0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=x.dtype)

    def bart(x):
        return x[0] ** 2 + x[1] ** 2 - R * R

    Gt, ht = ct.control_affine_cbf_rows(ft, gt, bart, torch.tensor(X), spec)
    assert np.max(np.abs(np.asarray(G) - Gt.numpy())) < 1e-10
    assert np.max(np.abs(np.asarray(h) - ht.numpy())) < 1e-10
