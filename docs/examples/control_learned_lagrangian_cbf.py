# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Learned-dynamics CBF safety filter -- differentiable AND certified.

Run:

    pip install "omnibias-control[jax,lagrangian,verify]" optax
    python docs/examples/control_learned_lagrangian_cbf.py

A planar point robot has an *unknown, coupled* inertia ``M_true`` (pushing in x also
accelerates y). We (1) recover it with a **Lagrangian Neural Network** built on
``omnibias-variational`` (the inverse-dynamics residual
``|| inverse_dynamics(L_theta; q,qdot,qddot,t) - tau ||^2``), (2) turn the learned
Lagrangian into a control-affine CBF row with ``omnibias.control`` (the learned
``g = M_hat^{-1} B``), (3) train a policy *through* the differentiable ``cbf_filter``
via ``safe_rollout``, and (4) certify the recoverable set for the learned model with
``omnibias-verify``.

Payoff: a CBF built on the *naive* identity-mass model still collides under the true
coupled dynamics, while the *learned*-model filter is collision-free -- and the policy
trained through it also reaches the goal. Everything runs on CPU in well under a minute.
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402  (after x64 config)
import numpy as np  # noqa: E402
import optax  # noqa: E402
from omnibias.control import CBFSpec, FilterSchedule, certify_disc_recoverable  # noqa: E402
from omnibias.control.jax import (  # noqa: E402
    control_affine_cbf_rows,
    lagrangian_cbf_rows,
    min_barrier,
    safe_rollout,
)
from omnibias.variational import Lagrangian  # noqa: E402
from omnibias.variational.jax import ops as vops  # noqa: E402

# world / dynamics
CX, CY, R = 0.0, 0.0, 1.0
A_MAX, DT, T = 2.5, 0.1, 30
GAINS = (2.0, 2.0)
GOAL = jnp.array([0.0, 3.0])
SPEC = CBFSpec(GAINS, a_max=A_MAX)
M_TRUE = np.array([[1.0, 0.6], [0.6, 1.0]])   # unknown coupled inertia (B = I)


# --------------------------------------------------------------------------- #
# 1) Lagrangian Neural Network: learn the mass matrix from force/accel data
# --------------------------------------------------------------------------- #
def spd(theta):
    lc = jnp.array([[theta["l11"], 0.0], [theta["l21"], theta["l22"]]])
    return lc @ lc.T + 1e-4 * jnp.eye(2)


def make_lagrangian(theta):
    return Lagrangian(lambda q, qd, t: 0.5 * qd @ (spd(theta) @ qd), dof=("q",))


def train_lnn(seed=0, steps=300, lr=5e-2):
    rng = np.random.default_rng(seed)
    q = jnp.asarray(rng.standard_normal((512, 2)))
    qd = jnp.asarray(rng.standard_normal((512, 2)))
    tau = jnp.asarray(rng.uniform(-A_MAX, A_MAX, size=(512, 2)))
    qddot = jnp.asarray(np.asarray(tau) @ np.linalg.inv(M_TRUE).T)
    t = jnp.zeros((512, 1))
    theta = {"l11": jnp.asarray(1.3), "l21": jnp.asarray(0.0), "l22": jnp.asarray(0.8)}
    opt = optax.adam(lr)
    state = opt.init(theta)

    def loss(theta):
        lag = make_lagrangian(theta)
        return jnp.mean((vops.inverse_dynamics(lag, q, qd, qddot, t) - tau) ** 2)

    vg = jax.jit(jax.value_and_grad(loss))
    for _ in range(steps):
        _, g = vg(theta)
        updates, state = opt.update(g, state, theta)
        theta = optax.apply_updates(theta, updates)
    return make_lagrangian(theta), np.asarray(spd(theta))


# --------------------------------------------------------------------------- #
# 2) CBF rows for a chosen model  qddot = g a  (g = M^{-1} B, constant here)
# --------------------------------------------------------------------------- #
def rows_fn_for(g_mat):
    g_full = jnp.concatenate([jnp.zeros((2, 2)), jnp.asarray(g_mat)], axis=0)  # (4,2)

    def f(x):
        return jnp.array([x[2], x[3], 0.0, 0.0])

    def g(x):
        return g_full

    def bar(x):
        return (x[0] - CX) ** 2 + (x[1] - CY) ** 2 - R * R

    return lambda x: control_affine_cbf_rows(f, g, bar, x, SPEC)


def true_step(x, a):
    minv_b_true = jnp.asarray(np.linalg.solve(M_TRUE, np.eye(2)))
    acc = a @ minv_b_true.T
    v2 = x[:, 2:] + DT * acc
    return jnp.concatenate([x[:, :2] + DT * v2, v2], axis=1)


def barrier(x):
    return (x[0] - CX) ** 2 + (x[1] - CY) ** 2 - R * R


# --------------------------------------------------------------------------- #
# 3) policy + training through the filter
# --------------------------------------------------------------------------- #
def init_mlp(seed, sizes=(6, 64, 64, 2)):
    rng = np.random.default_rng(seed)
    return [
        (jnp.asarray(rng.standard_normal((a, b)) / np.sqrt(a)), jnp.zeros(b))
        for a, b in zip(sizes[:-1], sizes[1:], strict=True)
    ]


def mlp(params, x):
    for w, b in params[:-1]:
        x = jnp.tanh(x @ w + b)
    w, b = params[-1]
    return x @ w + b


def policy_fn(params):
    def policy(x):
        feats = jnp.concatenate([x[:, :2], x[:, 2:], GOAL - x[:, :2]], axis=1)
        return A_MAX * jnp.tanh(mlp(params, feats) / A_MAX)

    return policy


def starts(seed, n):
    rng = np.random.default_rng(seed)
    px = rng.uniform(-1.2, 1.2, n)
    py = rng.uniform(-3.2, -2.8, n)
    v = 0.1 * rng.standard_normal((n, 2))
    return jnp.asarray(np.stack([px, py, v[:, 0], v[:, 1]], axis=1))


def train_policy(seed, rows_fn, filtered, epochs=140, lr=3e-3):
    params = init_mlp(seed)
    x0 = starts(seed + 11, 64)
    opt = optax.adam(lr)
    state = opt.init(params)
    sched = FilterSchedule.fast()

    def loss(params):
        pol = policy_fn(params)
        if filtered:
            X, A, _ = safe_rollout(pol, true_step, rows_fn, x0, horizon=T, schedule=sched)
        else:                                   # clip only (no CBF)
            X, A = _clip_rollout(pol, x0)
        d2 = jnp.sum((X[:, :, :2] - GOAL) ** 2, axis=-1)
        return jnp.mean(d2) + 3.0 * jnp.mean(d2[-1]) + 1e-3 * jnp.mean(jnp.sum(A ** 2, -1))

    vg = jax.jit(jax.value_and_grad(loss))
    for _ in range(epochs):
        _, g = vg(params)
        updates, state = opt.update(g, state, params)
        params = optax.apply_updates(params, updates)
    return params


def _clip_rollout(policy, x0):
    def step(x, _):
        a = jnp.clip(policy(x), -A_MAX, A_MAX)
        x2 = true_step(x, a)
        return x2, (x2, a)

    _, (X, A) = jax.lax.scan(step, x0, None, length=T)
    return X, A


def evaluate(params, rows_fn, filtered):
    x0 = starts(777, 256)
    pol = policy_fn(params)
    if filtered:
        X, _, _ = safe_rollout(pol, true_step, rows_fn, x0, horizon=T, schedule=FilterSchedule())
    else:
        X, _ = _clip_rollout(pol, x0)
    mb = np.asarray(min_barrier(barrier, X))
    final = np.asarray(X[-1, :, :2])
    dist = np.linalg.norm(final - np.asarray(GOAL), axis=1)
    return {
        "collision_rate": float((mb < -1e-3).mean()),
        "success_rate": float((dist < 1.0).mean()),
        "min_barrier": float(mb.min()),
    }


def main():
    print("Learned-dynamics CBF (omnibias-control) -- LNN -> control-affine -> certified filter\n")
    lag, m_hat = train_lnn()
    err = float(np.linalg.norm(m_hat - M_TRUE))
    print(f"LNN inertia error ||M_hat - M_true||_F = {err:.2e}\n  M_hat = {np.round(m_hat, 3).tolist()}")

    g_hat = np.linalg.solve(m_hat, np.eye(2))       # learned M^{-1} B
    g_naive = np.eye(2)                             # naive identity-mass model

    # the package's lagrangian_cbf_rows reproduces the learned control-affine rows
    xs = starts(1, 6)
    Gl, hl = lagrangian_cbf_rows(lag, jnp.eye(2), barrier, xs[:, :2], xs[:, 2:], jnp.zeros((6, 1)), SPEC)
    Gc, hc = rows_fn_for(g_hat)(xs)
    assert float(jnp.max(jnp.abs(Gl - Gc))) < 1e-6, "lagrangian_cbf_rows must match control-affine"
    print("  lagrangian_cbf_rows == control_affine_cbf_rows(g=M_hat^-1 B): OK")

    rows_learned = rows_fn_for(g_hat)
    net_free = train_policy(0, rows_learned, filtered=False)   # trained w/o filter
    net_ours = train_policy(0, rows_learned, filtered=True)    # trained THROUGH the filter

    res = {
        "nominal (no filter)": evaluate(net_free, rows_learned, filtered=False),
        "filter@test (naive I)": evaluate(net_free, rows_fn_for(g_naive), filtered=True),
        "filter@test (learned)": evaluate(net_free, rows_learned, filtered=True),
        "ours (trained thru)": evaluate(net_ours, rows_learned, filtered=True),
    }
    print("\n  method                    collisions   success   min_barrier")
    for name, r in res.items():
        print(f"  {name:24s}  {r['collision_rate']:9.3f}  {r['success_rate']:8.3f}  {r['min_barrier']:11.3f}")

    # rigorous recoverable-set certificate for the LEARNED model
    print("\n  recoverable-set certificate (learned g = M_hat^-1 B), corridor py in [-2.5,-1.5]:")
    safe_speed = 0.0
    for vmax in (0.5, 1.0, 1.5):
        cert = certify_disc_recoverable((CX, CY), R, GAINS, A_MAX, [(-1.5, 1.5), (-2.5, -1.5)],
                                        vmax, g=g_hat, tol=1e-2)
        assert cert is not None
        if cert.certified:
            safe_speed = vmax
        print(f"    v_max={vmax}: f_lower={cert.f_lower:+.3f}  certified={cert.certified}")
    print(f"  => certified safe speed <= {safe_speed:.1f} for the learned model "
          f"(transfers to the true system up to ||M_hat-M_true|| = {err:.1e}).")

    assert err < 1e-3, "LNN should recover the inertia"
    assert res["ours (trained thru)"]["collision_rate"] == 0.0, "learned-model filter must be safe"
    assert res["ours (trained thru)"]["success_rate"] > 0.5, "trained-through policy should reach"
    assert res["nominal (no filter)"]["collision_rate"] > 0.1, "unfiltered nominal should collide"
    assert safe_speed >= 1.0, "low-speed corridor should certify"
    print("\nOK: learned dynamics -> certified-safe filter; training through it reaches the goal.")


if __name__ == "__main__":
    main()
