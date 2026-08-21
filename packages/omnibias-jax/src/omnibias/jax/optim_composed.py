# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Composed-curvature joint Newton (theory 08-02), JAX twin.

Bit-identical partner of :mod:`omnibias.torch.optim_composed` on the same
subspace Hessian floats. Enable 64-bit JAX before the first array if you
need torch parity (:mod:`omnibias.jax.precision`). Do not ``jit`` the
driver.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import cast

from omnibias.core.composed_curvature import (
    ComposedCurvatureConfig,
    ComposedCurvatureReport,
    SubspaceStep,
    chain_rule_mse_blocks,
    reject_full_parameter_jacobian,
    select_composed_step,
)
from omnibias.core.line_search import JetLineSearchConfig
from omnibias.jax.activations import get_activation
from omnibias.jax.jet import compose_jet, jet_to_tower
from omnibias.jax.line_search import jet_line_search

import jax
import jax.numpy as jnp
from jax import Array

ResidualFn = Callable[[Array, Array], Array]
Direction = tuple[Array, Array]


def _as_flat(t: Array) -> Array:
    return jnp.reshape(jnp.asarray(t), (-1,))


def _orthonormalize(cols: Sequence[Array], *, atol: float = 1e-12) -> Array:
    basis: list[Array] = []
    for raw in cols:
        w = jnp.reshape(jnp.asarray(raw), (-1,))
        for b in basis:
            w = w - jnp.dot(w, b) * b
        nrm = jnp.linalg.norm(w)
        if float(nrm) > atol:
            basis.append(w / nrm)
    if not basis:
        raise ValueError("directions are linearly dependent or zero")
    return jnp.stack(basis, axis=1)


def _unit_or_basis(vec: Array, index: int) -> Array:
    nrm = jnp.linalg.norm(vec)
    if float(nrm) > 1e-12:
        return cast(Array, vec / nrm)
    out = jnp.zeros_like(vec)
    return out.at[index % int(out.size)].set(1.0)


def default_directions(
    grad_flat: Array,
    n_prev: int,
    n_curr: int,
    n_directions: int,
) -> list[Direction]:
    """Prev-only gradient, curr-only gradient, then canonical remaining axes."""
    g_prev = grad_flat[:n_prev]
    g_curr = grad_flat[n_prev:]
    zero_prev = jnp.zeros_like(g_prev)
    zero_curr = jnp.zeros_like(g_curr)
    dirs: list[Direction] = [
        (_unit_or_basis(g_prev, 0), zero_curr),
        (zero_prev, _unit_or_basis(g_curr, 0)),
    ]
    total = n_prev + n_curr
    eye = jnp.eye(total, dtype=grad_flat.dtype)
    for i in range(total):
        if len(dirs) >= n_directions:
            break
        vec = eye[i]
        dirs.append((vec[:n_prev], vec[n_prev:]))
    return dirs[:n_directions]


def _matrix_floats(h: Array) -> tuple[tuple[float, ...], ...]:
    return tuple(tuple(float(x) for x in row) for row in h)


def _curr_only_indices(q: Array, n_prev: int, *, atol: float = 1e-10) -> list[int]:
    masses = jnp.linalg.norm(q[:n_prev, :], axis=0)
    return [i for i, m in enumerate(masses.tolist()) if m <= atol]


def scalar_nest_hessian(
    w: Array | float,
    v: Array | float,
    x: Array | float = 1.0,
) -> tuple[float, float, float]:
    """Order-2 ``compose_jet`` Hessian of the spec §5 scalar nest."""
    w_t = jnp.asarray(w)
    v_t = jnp.asarray(v)
    x_t = jnp.asarray(x, dtype=w_t.dtype)
    u0 = w_t * x_t
    zero = jnp.zeros_like(u0)
    u_jet = jnp.stack([u0, x_t.astype(u0.dtype), zero])
    spec = get_activation("tanh")
    fp = spec.fastpath
    if fp is None:
        raise ValueError("tanh fastpath is required for the scalar-nest Hessian")
    tower = jnp.stack([spec.forward(u0), fp(u0, 1), fp(u0, 2)])
    h_jet = compose_jet(u_jet, tower)
    derivs = jet_to_tower(h_jet)
    return chain_rule_mse_blocks(
        float(derivs[0]),
        float(derivs[1]),
        float(derivs[2]),
        float(v_t),
    )


def composed_block_hessian(
    residual_fn: ResidualFn,
    params_prev: Array,
    params_curr: Array,
    directions: Sequence[Direction] | None = None,
    *,
    config: ComposedCurvatureConfig | None = None,
) -> tuple[Array, Array, Array]:
    """Return ``(H_slice, H_joint, cross)`` in the ``k``-direction subspace."""
    cfg = config if config is not None else ComposedCurvatureConfig()
    prev = jnp.asarray(params_prev)
    curr = jnp.asarray(params_curr)
    n_prev = int(prev.size)
    n_curr = int(curr.size)
    n_params = n_prev + n_curr
    n_dir_claim = cfg.n_directions if directions is None else len(directions)
    reject_full_parameter_jacobian(n_dir_claim, n_params, cfg.allow_full)

    theta = jnp.concatenate([_as_flat(prev), _as_flat(curr)])

    def residual_flat(th: Array) -> Array:
        p = jnp.reshape(th[:n_prev], prev.shape)
        c = jnp.reshape(th[n_prev:], curr.shape)
        return jnp.reshape(jnp.asarray(residual_fn(p, c)), (-1,))

    def loss(th: Array) -> Array:
        r = residual_flat(th)
        return 0.5 * jnp.dot(r, r)

    used = directions
    if used is None:
        g = jax.grad(loss)(theta)
        used = default_directions(g, n_prev, n_curr, cfg.n_directions)
    cols = [
        jnp.concatenate(
            [
                _as_flat(d_prev).astype(theta.dtype),
                _as_flat(d_curr).astype(theta.dtype),
            ]
        )
        for d_prev, d_curr in used
    ]
    q = _orthonormalize(cols)

    def matvec(vec: Array) -> Array:
        if cfg.include_residual_hess:
            return cast(Array, jax.jvp(jax.grad(loss), (theta,), (vec,))[1])
        jv = jax.jvp(residual_flat, (theta,), (vec,))[1]
        vjp_fn = jax.vjp(residual_flat, theta)[1]
        return cast(Array, vjp_fn(jv)[0])

    k = int(q.shape[1])
    columns = [q.T @ matvec(q[:, j]) for j in range(k)]
    h_joint = 0.5 * (jnp.stack(columns, axis=1) + jnp.stack(columns, axis=1).T)

    curr_idx = _curr_only_indices(q, n_prev)
    if not curr_idx:
        embedded = jnp.concatenate(
            [jnp.zeros((n_prev, k), dtype=q.dtype), q[n_prev:, :]],
            axis=0,
        )
        q_slice = _orthonormalize([embedded[:, j] for j in range(k)])
    else:
        q_slice = q[:, jnp.asarray(curr_idx)]
    k_s = int(q_slice.shape[1])
    slice_cols = [q_slice.T @ matvec(q_slice[:, j]) for j in range(k_s)]
    h_slice = 0.5 * (jnp.stack(slice_cols, axis=1) + jnp.stack(slice_cols, axis=1).T)

    prev_idx = [
        i
        for i in range(k)
        if float(jnp.linalg.norm(q[n_prev:, i])) <= 1e-10
    ]
    if prev_idx and curr_idx:
        q_p = q[:, jnp.asarray(prev_idx)]
        q_c = q[:, jnp.asarray(curr_idx)]
        cross_cols = [q_p.T @ matvec(q_c[:, j]) for j in range(len(curr_idx))]
        cross = jnp.stack(cross_cols, axis=1)
    else:
        cross = jnp.zeros((0, k_s), dtype=q.dtype)
    return h_slice, h_joint, cross


def _apply_subspace_step(
    residual_fn: ResidualFn,
    prev: Array,
    curr: Array,
    q: Array,
    step: SubspaceStep,
) -> tuple[Array, Array, ComposedCurvatureReport]:
    n_prev = int(prev.size)
    theta = jnp.concatenate([_as_flat(prev), _as_flat(curr)])
    coeffs = jnp.asarray(step.coeffs, dtype=theta.dtype)
    direction = q @ coeffs
    nrm = float(jnp.linalg.norm(direction))
    if nrm == 0.0:
        report = ComposedCurvatureReport(
            lambda_min_slice=step.lambda_min_slice,
            lambda_min_joint=step.lambda_min_joint,
            escaped=step.escaped,
            step_norm=0.0,
        )
        return prev, curr, report

    unit = direction / nrm

    def loss_flat(th: Array) -> Array:
        p = jnp.reshape(th[:n_prev], prev.shape)
        c = jnp.reshape(th[n_prev:], curr.shape)
        r = jnp.reshape(jnp.asarray(residual_fn(p, c)), (-1,))
        return 0.5 * jnp.dot(r, r)

    ls = jet_line_search(
        loss_flat,
        theta,
        unit,
        config=JetLineSearchConfig(
            order=2,
            trust_radius=1.0,
            verify=True,
            max_step=1.0,
        ),
    )
    alpha = float(ls.step)
    if alpha == 0.0 and step.escaped:
        trial = 0.05
        f0 = float(loss_flat(theta))
        f_plus = float(loss_flat(theta + trial * unit))
        f_minus = float(loss_flat(theta - trial * unit))
        if f_plus < f0:
            alpha = trial
        elif f_minus < f0:
            alpha = -trial
    delta = alpha * unit
    new_theta = theta + delta
    new_prev = jnp.reshape(new_theta[:n_prev], prev.shape)
    new_curr = jnp.reshape(new_theta[n_prev:], curr.shape)
    report = ComposedCurvatureReport(
        lambda_min_slice=step.lambda_min_slice,
        lambda_min_joint=step.lambda_min_joint,
        escaped=step.escaped,
        step_norm=float(jnp.linalg.norm(delta)),
    )
    return new_prev, new_curr, report


def composed_curvature_step(
    residual_fn: ResidualFn,
    params_prev: Array,
    params_curr: Array,
    *,
    config: ComposedCurvatureConfig | None = None,
    directions: Sequence[Direction] | None = None,
) -> tuple[Array, Array, ComposedCurvatureReport]:
    """Joint damped Newton, or an escape step along a negative joint mode."""
    cfg = config if config is not None else ComposedCurvatureConfig()
    prev = jnp.asarray(params_prev)
    curr = jnp.asarray(params_curr)
    n_prev = int(prev.size)
    n_curr = int(curr.size)
    h_slice, h_joint, _cross = composed_block_hessian(
        residual_fn,
        prev,
        curr,
        directions,
        config=cfg,
    )
    theta = jnp.concatenate([_as_flat(prev), _as_flat(curr)])

    def residual_flat(th: Array) -> Array:
        p = jnp.reshape(th[:n_prev], prev.shape)
        c = jnp.reshape(th[n_prev:], curr.shape)
        return jnp.reshape(jnp.asarray(residual_fn(p, c)), (-1,))

    def loss(th: Array) -> Array:
        r = residual_flat(th)
        return 0.5 * jnp.dot(r, r)

    used = directions
    if used is None:
        g = jax.grad(loss)(theta)
        used = default_directions(g, n_prev, n_curr, cfg.n_directions)
    cols = [
        jnp.concatenate(
            [
                _as_flat(d_prev).astype(theta.dtype),
                _as_flat(d_curr).astype(theta.dtype),
            ]
        )
        for d_prev, d_curr in used
    ]
    q = _orthonormalize(cols)
    g_theta = jax.grad(loss)(theta)
    g_floats = tuple(float(jnp.dot(q[:, j], g_theta)) for j in range(int(q.shape[1])))
    step = select_composed_step(
        _matrix_floats(h_slice),
        _matrix_floats(h_joint),
        g_floats,
        cfg,
    )
    return _apply_subspace_step(residual_fn, prev, curr, q, step)


__all__ = [
    "ComposedCurvatureConfig",
    "ComposedCurvatureReport",
    "composed_block_hessian",
    "composed_curvature_step",
    "default_directions",
    "scalar_nest_hessian",
]
