# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""safe_rollout: the filter keeps the barrier non-negative; differentiable; parity."""

from __future__ import annotations

import numpy as np
from omnibias.control import CBFSpec, FilterSchedule

DT = 0.1
GAINS = (2.0, 2.0)
AMAX = 2.5
R = 1.0
GOAL = np.array([0.0, 3.0])


def _starts(n=32, seed=0):
    rng = np.random.default_rng(seed)
    px = rng.uniform(-1.0, 1.0, n)
    py = rng.uniform(-3.0, -2.6, n)
    v = 0.05 * rng.standard_normal((n, 2))
    return np.stack([px, py, v[:, 0], v[:, 1]], axis=1)


def test_filter_keeps_barrier_nonnegative():
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

    spec = CBFSpec(GAINS, a_max=AMAX)
    goal = jnp.asarray(GOAL)

    def policy(x):                              # greedy drive to goal (would collide)
        return 5.0 * (goal - x[:, :2]) - 3.0 * x[:, 2:]

    def step(x, a):
        v2 = x[:, 2:] + DT * a
        p2 = x[:, :2] + DT * v2
        return jnp.concatenate([p2, v2], axis=1)

    def rows_fn(x):
        return cj.control_affine_cbf_rows(f, g, bar, x, spec)

    x0 = jnp.asarray(_starts())

    # filtered: safe
    X, _A, _R = cj.safe_rollout(policy, step, rows_fn, x0, horizon=40, schedule=FilterSchedule())
    mb = np.asarray(cj.min_barrier(bar, X))
    assert mb.min() > -0.05                     # barrier stays ~>= 0 (exterior-penalty slack)

    # unfiltered greedy control: collides (drives straight through the disc)
    x = x0
    hit = False
    for _ in range(40):
        a = jnp.clip(policy(x), -AMAX, AMAX)
        v2 = x[:, 2:] + DT * a
        p2 = x[:, :2] + DT * v2
        x = jnp.concatenate([p2, v2], axis=1)
        if float(jnp.min(bar(x))) < -1e-3:
            hit = True
    assert hit


def test_rollout_differentiable():
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

    spec = CBFSpec(GAINS, a_max=AMAX)
    x0 = jnp.asarray(_starts(n=8))
    goal = jnp.asarray(GOAL)

    def step(x, a):
        v2 = x[:, 2:] + DT * a
        p2 = x[:, :2] + DT * v2
        return jnp.concatenate([p2, v2], axis=1)

    def rows_fn(x):
        return cj.control_affine_cbf_rows(f, g, bar, x, spec)

    def loss(W):
        def policy(x):
            feats = jnp.concatenate([goal - x[:, :2], x[:, 2:]], axis=1)
            return feats @ W
        X, _A, _R = cj.safe_rollout(policy, step, rows_fn, x0, horizon=15,
                                    schedule=FilterSchedule.fast())
        return jnp.mean(jnp.sum((X[-1, :, :2] - goal) ** 2, axis=1))

    W = jnp.asarray(np.random.default_rng(0).standard_normal((4, 2)) * 0.1)
    grad = jax.grad(loss)(W)
    assert np.all(np.isfinite(np.asarray(grad)))
    assert float(jnp.max(jnp.abs(grad))) > 0.0


def test_rollout_torch_jax_parity():
    import jax

    jax.config.update("jax_enable_x64", True)
    import torch

    torch.set_default_dtype(torch.float64)
    import jax.numpy as jnp
    import omnibias.control.jax as cj
    import omnibias.control.torch as ct

    spec = CBFSpec(GAINS, a_max=AMAX)
    x0 = _starts(n=10)
    W = np.random.default_rng(1).standard_normal((4, 2)) * 0.3

    def run_jax():
        goal = jnp.asarray(GOAL)

        def f(x):
            return jnp.array([x[2], x[3], 0.0, 0.0])

        def g(x):
            return jnp.array([[0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])

        def bar(x):
            return x[0] ** 2 + x[1] ** 2 - R * R

        Wj = jnp.asarray(W)

        def policy(x):
            return jnp.concatenate([goal - x[:, :2], x[:, 2:]], axis=1) @ Wj

        def step(x, a):
            v2 = x[:, 2:] + DT * a
            return jnp.concatenate([x[:, :2] + DT * v2, v2], axis=1)

        X, _, _ = cj.safe_rollout(policy, step, lambda x: cj.control_affine_cbf_rows(f, g, bar, x, spec),
                                  jnp.asarray(x0), horizon=12, schedule=FilterSchedule.fast())
        return np.asarray(X)

    def run_torch():
        goal = torch.tensor(GOAL)

        def f(x):
            z = torch.zeros((), dtype=x.dtype)
            return torch.stack([x[2], x[3], z, z])

        def g(x):
            return torch.tensor([[0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=x.dtype)

        def bar(x):
            return x[0] ** 2 + x[1] ** 2 - R * R

        Wt = torch.tensor(W)

        def policy(x):
            return torch.cat([goal - x[:, :2], x[:, 2:]], dim=1) @ Wt

        def step(x, a):
            v2 = x[:, 2:] + DT * a
            return torch.cat([x[:, :2] + DT * v2, v2], dim=1)

        X, _, _ = ct.safe_rollout(policy, step, lambda x: ct.control_affine_cbf_rows(f, g, bar, x, spec),
                                  torch.tensor(x0), horizon=12, schedule=FilterSchedule.fast())
        return X.detach().numpy()

    assert np.max(np.abs(run_jax() - run_torch())) < 1e-9
