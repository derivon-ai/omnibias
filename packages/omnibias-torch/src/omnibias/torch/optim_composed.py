# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Composed-curvature joint Newton (theory 08-02), PyTorch twin.

Builds the ``k``-direction subspace Hessian of a two-block residual
``1/2 ||r(W_{ell-1}, W_ell)||^2`` from Hessian-vector products. The
Gauss-Newton term ``J^T J`` is PSD; ``include_residual_hess=True`` keeps
``r · r''``, which contains ``sigma''`` from the founding bias collapse
(``delta -> 0``) and can make the joint block indefinite while the
``W_ell`` slice stays positive definite.

The subspace step (damped Newton vs unit negative-curvature direction)
is the shared core solver, so the jax twin is bit-identical on the same
floats. Length along that direction uses spec 03-12
(:func:`omnibias.torch.line_search.jet_line_search`) with ``verify=True``.

Do not wrap the driver in ``torch.compile``. The API refuses
``n_directions >= n_params`` unless ``allow_full=True``.
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
from omnibias.torch.activations.registry import get_activation
from omnibias.torch.jet import compose_jet, jet_to_tower
from omnibias.torch.line_search import jet_line_search

import torch
from torch import Tensor
from torch.func import grad, jvp, vjp

ResidualFn = Callable[[Tensor, Tensor], Tensor]
Direction = tuple[Tensor, Tensor]


def _as_flat(t: Tensor) -> Tensor:
    return torch.as_tensor(t).reshape(-1)


def _orthonormalize(cols: Sequence[Tensor], *, atol: float = 1e-12) -> Tensor:
    basis: list[Tensor] = []
    for raw in cols:
        w = torch.as_tensor(raw).reshape(-1).clone()
        for b in basis:
            w = w - torch.dot(w, b) * b
        nrm = torch.linalg.vector_norm(w)
        if float(nrm) > atol:
            basis.append(w / nrm)
    if not basis:
        raise ValueError("directions are linearly dependent or zero")
    return torch.stack(basis, dim=1)


def _unit_or_basis(vec: Tensor, index: int) -> Tensor:
    nrm = torch.linalg.vector_norm(vec)
    if float(nrm) > 1e-12:
        return cast(Tensor, vec / nrm)
    out = torch.zeros_like(vec)
    out[index % int(out.numel())] = 1.0
    return out


def default_directions(
    grad_flat: Tensor,
    n_prev: int,
    n_curr: int,
    n_directions: int,
) -> list[Direction]:
    """Prev-only gradient, curr-only gradient, then canonical remaining axes."""
    g_prev = grad_flat[:n_prev]
    g_curr = grad_flat[n_prev:]
    zero_prev = torch.zeros_like(g_prev)
    zero_curr = torch.zeros_like(g_curr)
    dirs: list[Direction] = [
        (_unit_or_basis(g_prev, 0), zero_curr),
        (zero_prev, _unit_or_basis(g_curr, 0)),
    ]
    total = n_prev + n_curr
    eye = torch.eye(total, dtype=grad_flat.dtype, device=grad_flat.device)
    for i in range(total):
        if len(dirs) >= n_directions:
            break
        vec = eye[i]
        dirs.append((vec[:n_prev], vec[n_prev:]))
    return dirs[:n_directions]


def _matrix_floats(h: Tensor) -> tuple[tuple[float, ...], ...]:
    rows = h.detach().cpu()
    return tuple(tuple(float(x) for x in row) for row in rows)


def _curr_only_indices(q: Tensor, n_prev: int, *, atol: float = 1e-10) -> list[int]:
    masses = torch.linalg.vector_norm(q[:n_prev, :], dim=0)
    return [i for i, m in enumerate(masses.tolist()) if m <= atol]


def scalar_nest_hessian(
    w: Tensor | float,
    v: Tensor | float,
    x: Tensor | float = 1.0,
) -> tuple[float, float, float]:
    """Order-2 ``compose_jet`` Hessian of the spec §5 scalar nest."""
    w_t = torch.as_tensor(w)
    v_t = torch.as_tensor(v)
    x_t = torch.as_tensor(x, dtype=w_t.dtype, device=w_t.device)
    u0 = w_t * x_t
    zero = torch.zeros_like(u0)
    u_jet = torch.stack([u0, x_t.to(dtype=u0.dtype), zero])
    spec = get_activation("tanh")
    fp = spec.fastpath
    if fp is None:
        raise ValueError("tanh fastpath is required for the scalar-nest Hessian")
    tower = torch.stack([spec.forward(u0), fp(u0, 1), fp(u0, 2)])
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
    params_prev: Tensor,
    params_curr: Tensor,
    directions: Sequence[Direction] | None = None,
    *,
    config: ComposedCurvatureConfig | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return ``(H_slice, H_joint, cross)`` in the ``k``-direction subspace.

    ``H_slice`` is the curr-only Rayleigh restriction (prev frozen).
    ``cross`` is the mixed prev-only / curr-only block when those columns
    exist; otherwise it is empty ``(0, k_c)``.
    """
    cfg = config if config is not None else ComposedCurvatureConfig()
    prev = torch.as_tensor(params_prev)
    curr = torch.as_tensor(params_curr)
    n_prev = int(prev.numel())
    n_curr = int(curr.numel())
    n_params = n_prev + n_curr
    n_dir_claim = cfg.n_directions if directions is None else len(directions)
    reject_full_parameter_jacobian(n_dir_claim, n_params, cfg.allow_full)

    theta = torch.cat([_as_flat(prev), _as_flat(curr)])

    def residual_flat(th: Tensor) -> Tensor:
        p = th[:n_prev].reshape(prev.shape)
        c = th[n_prev:].reshape(curr.shape)
        return torch.as_tensor(residual_fn(p, c)).reshape(-1)

    def loss(th: Tensor) -> Tensor:
        r = residual_flat(th)
        return 0.5 * torch.dot(r, r)

    used = directions
    if used is None:
        g = cast(Tensor, grad(loss)(theta))
        used = default_directions(g, n_prev, n_curr, cfg.n_directions)
    cols = [
        torch.cat([_as_flat(d_prev).to(dtype=theta.dtype, device=theta.device),
                   _as_flat(d_curr).to(dtype=theta.dtype, device=theta.device)])
        for d_prev, d_curr in used
    ]
    q = _orthonormalize(cols)

    def matvec(vec: Tensor) -> Tensor:
        if cfg.include_residual_hess:
            return cast(Tensor, jvp(grad(loss), (theta,), (vec,))[1])
        jv = jvp(residual_flat, (theta,), (vec,))[1]
        vjp_fn = vjp(residual_flat, theta)[1]
        return cast(Tensor, vjp_fn(jv)[0])

    k = int(q.shape[1])
    h_joint = q.new_zeros((k, k))
    for j in range(k):
        hv = matvec(q[:, j])
        h_joint[:, j] = q.T @ hv
    h_joint = 0.5 * (h_joint + h_joint.T)

    curr_idx = _curr_only_indices(q, n_prev)
    if not curr_idx:
        embedded = torch.cat(
            [theta.new_zeros((n_prev, k)), q[n_prev:, :]],
            dim=0,
        )
        q_slice = _orthonormalize([embedded[:, j] for j in range(k)])
    else:
        q_slice = q[:, curr_idx]
    k_s = int(q_slice.shape[1])
    h_slice = q.new_zeros((k_s, k_s))
    for j in range(k_s):
        hv = matvec(q_slice[:, j])
        h_slice[:, j] = q_slice.T @ hv
    h_slice = 0.5 * (h_slice + h_slice.T)

    prev_idx = [
        i
        for i in range(k)
        if float(torch.linalg.vector_norm(q[n_prev:, i])) <= 1e-10
    ]
    if prev_idx and curr_idx:
        q_p = q[:, prev_idx]
        q_c = q[:, curr_idx]
        cross = q.new_zeros((len(prev_idx), len(curr_idx)))
        for j in range(len(curr_idx)):
            hv = matvec(q_c[:, j])
            cross[:, j] = q_p.T @ hv
    else:
        cross = q.new_zeros((0, k_s))
    return h_slice, h_joint, cross


def _apply_subspace_step(
    residual_fn: ResidualFn,
    prev: Tensor,
    curr: Tensor,
    q: Tensor,
    step: SubspaceStep,
) -> tuple[Tensor, Tensor, ComposedCurvatureReport]:
    n_prev = int(prev.numel())
    theta = torch.cat([_as_flat(prev), _as_flat(curr)])
    coeffs = torch.as_tensor(step.coeffs, dtype=theta.dtype, device=theta.device)
    direction = q @ coeffs
    nrm = float(torch.linalg.vector_norm(direction))
    if nrm == 0.0:
        report = ComposedCurvatureReport(
            lambda_min_slice=step.lambda_min_slice,
            lambda_min_joint=step.lambda_min_joint,
            escaped=step.escaped,
            step_norm=0.0,
        )
        return prev, curr, report

    unit = direction / nrm

    def loss_flat(th: Tensor) -> Tensor:
        p = th[:n_prev].reshape(prev.shape)
        c = th[n_prev:].reshape(curr.shape)
        r = torch.as_tensor(residual_fn(p, c)).reshape(-1)
        return 0.5 * torch.dot(r, r)

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
    new_prev = new_theta[:n_prev].reshape(prev.shape)
    new_curr = new_theta[n_prev:].reshape(curr.shape)
    report = ComposedCurvatureReport(
        lambda_min_slice=step.lambda_min_slice,
        lambda_min_joint=step.lambda_min_joint,
        escaped=step.escaped,
        step_norm=float(torch.linalg.vector_norm(delta)),
    )
    return new_prev, new_curr, report


def composed_curvature_step(
    residual_fn: ResidualFn,
    params_prev: Tensor,
    params_curr: Tensor,
    *,
    config: ComposedCurvatureConfig | None = None,
    directions: Sequence[Direction] | None = None,
) -> tuple[Tensor, Tensor, ComposedCurvatureReport]:
    """Joint damped Newton, or an escape step along a negative joint mode."""
    cfg = config if config is not None else ComposedCurvatureConfig()
    prev = torch.as_tensor(params_prev)
    curr = torch.as_tensor(params_curr)
    n_prev = int(prev.numel())
    h_slice, h_joint, _cross = composed_block_hessian(
        residual_fn,
        prev,
        curr,
        directions,
        config=cfg,
    )
    theta = torch.cat([_as_flat(prev), _as_flat(curr)])

    def residual_flat(th: Tensor) -> Tensor:
        p = th[:n_prev].reshape(prev.shape)
        c = th[n_prev:].reshape(curr.shape)
        return torch.as_tensor(residual_fn(p, c)).reshape(-1)

    def loss(th: Tensor) -> Tensor:
        r = residual_flat(th)
        return 0.5 * torch.dot(r, r)

    used = directions
    if used is None:
        g = cast(Tensor, grad(loss)(theta))
        used = default_directions(g, n_prev, int(curr.numel()), cfg.n_directions)
    cols = [
        torch.cat([_as_flat(d_prev).to(dtype=theta.dtype, device=theta.device),
                   _as_flat(d_curr).to(dtype=theta.dtype, device=theta.device)])
        for d_prev, d_curr in used
    ]
    q = _orthonormalize(cols)
    g_joint = [(q[:, j] * cast(Tensor, grad(loss)(theta))).sum() for j in range(int(q.shape[1]))]
    g_floats = tuple(float(x) for x in g_joint)
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
