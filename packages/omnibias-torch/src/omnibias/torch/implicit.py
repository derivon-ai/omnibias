# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Implicit / DEQ Newton (theory 08-08), PyTorch twin.

Solve ``u = sigma(W u + x)`` to a named residual, then differentiate
the root by the implicit-function theorem with exact ``sigma'``
(bias collapse, ``delta -> 0``). One linear solve replaces unrolled
backprop through a fixed-point iteration.

The solver loop is a Python ``while`` (do not wrap in
``torch.compile``). JAX uses ``lax.while_loop``; see the twin
docstring. IFT is the chain rule at a fixed point, not an absence of
the chain rule. Not a global min and not CCF stretch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from omnibias.core.implicit import (
    DEQConfig,
    DEQNotContractive,
    DEQSolverUnknown,
    honesty_payload,
    reject_anderson,
    reject_deq_contraction,
)
from omnibias.torch.activations.registry import get_activation

import torch
from torch import Tensor


@dataclass(frozen=True)
class DEQResult:
    """Fixed point, residual, iteration count, and contraction bound."""

    u: Tensor
    residual: float
    n_iter: int
    spectral_radius_bound: float


def _as_W(W: Tensor) -> Tensor:
    w = torch.as_tensor(W)
    if w.ndim == 0:
        w = w.reshape(1, 1)
    if w.ndim != 2 or int(w.shape[0]) != int(w.shape[1]):
        raise ValueError(f"W must be a square matrix, got shape {tuple(w.shape)}")
    return w


def _as_x(x: Tensor, d: int) -> Tensor:
    t = torch.as_tensor(x)
    if t.ndim == 0:
        t = t.reshape(1)
    if t.ndim == 1:
        if int(t.shape[0]) != d:
            raise ValueError(f"x dim {int(t.shape[0])} must equal W size {d}")
        return t
    if int(t.shape[-1]) != d:
        raise ValueError(f"x trailing dim {int(t.shape[-1])} must equal W size {d}")
    return t


def _affine(u: Tensor, W: Tensor) -> Tensor:
    return cast(Tensor, torch.tensordot(u, W, dims=([-1], [-1])))


def _sigma(z: Tensor, spec: str) -> Tensor:
    act = get_activation(spec)
    return act.forward(z)


def _sigma_prime(z: Tensor, spec: str) -> Tensor:
    act = get_activation(spec)
    fp = act.fastpath
    if fp is None:
        raise ValueError(
            f"activation {spec!r} has no fastpath; exact sigma' is required"
        )
    return fp(z, 1)


def _residual(u: Tensor, W: Tensor, x: Tensor, spec: str) -> Tensor:
    return u - _sigma(_affine(u, W) + x, spec)


def _residual_norm(res: Tensor) -> float:
    return float(torch.linalg.vector_norm(res.reshape(-1)))


def _spectral_bound(u: Tensor, W: Tensor, x: Tensor, spec: str) -> float:
    z = _affine(u, W) + x
    sigp = _sigma_prime(z, spec).abs()
    row_l1 = W.abs().sum(dim=-1)
    per = sigp * row_l1
    return float(per.reshape(-1).max())


def _linear_map(u: Tensor, W: Tensor, spec: str, x: Tensor) -> Tensor:
    """``(I - D W) v`` operator applied by solving against ``rhs`` later."""
    z = _affine(u, W) + x
    return _sigma_prime(z, spec)


def _solve_A(sigp: Tensor, W: Tensor, rhs: Tensor) -> Tensor:
    """Solve ``(I - diag(sigp) W) delta = rhs``; batched when ``rhs`` is 2-D."""
    d = int(W.shape[0])
    eye = torch.eye(d, dtype=W.dtype, device=W.device)
    if rhs.ndim == 1:
        a = eye - sigp.reshape(d, 1) * W
        return cast(Tensor, torch.linalg.solve(a, rhs))
    n = int(rhs.shape[0])
    a = eye.unsqueeze(0) - sigp.reshape(n, d, 1) * W.unsqueeze(0)
    return cast(Tensor, torch.linalg.solve(a, rhs.reshape(n, d)))


def _newton_step(u: Tensor, W: Tensor, x: Tensor, spec: str) -> Tensor:
    res = _residual(u, W, x, spec)
    sigp = _linear_map(u, W, spec, x)
    return u + _solve_A(sigp, W, -res)


def _iterate_step(u: Tensor, W: Tensor, x: Tensor, spec: str) -> Tensor:
    return _sigma(_affine(u, W) + x, spec)


def _maybe_contract(
    u: Tensor, W: Tensor, x: Tensor, spec: str, config: DEQConfig
) -> float:
    bound = _spectral_bound(u, W, x, spec)
    reject_deq_contraction(bound, require=config.require_contraction)
    return bound


def deq_solve(
    W: Tensor,
    x: Tensor,
    spec: str = "tanh",
    *,
    config: DEQConfig | None = None,
) -> DEQResult:
    """Solve ``u = sigma(W u + x)`` by Newton or Banach iteration.

    torch deq_solve uses a Python while loop; do not wrap it in
    torch.compile. The IFT VJP is one linear solve, not unrolled BPTT.
    """
    cfg = config if config is not None else DEQConfig()
    reject_anderson(cfg.solver)
    w = _as_W(W)
    x_t = _as_x(x, int(w.shape[0])).to(dtype=w.dtype, device=w.device)
    u = torch.zeros_like(x_t)
    bound = _maybe_contract(u, w, x_t, spec, cfg)
    res = _residual_norm(_residual(u, w, x_t, spec))
    n_iter = 0
    while res > cfg.tol and n_iter < cfg.max_iter:
        if cfg.solver == "newton":
            u = _newton_step(u, w, x_t, spec)
        else:
            u = _iterate_step(u, w, x_t, spec)
        bound = _maybe_contract(u, w, x_t, spec, cfg)
        res = _residual_norm(_residual(u, w, x_t, spec))
        n_iter += 1
    return DEQResult(
        u=u,
        residual=res,
        n_iter=n_iter,
        spectral_radius_bound=bound,
    )


def deq_vjp(
    W: Tensor,
    x: Tensor,
    spec: str,
    g: Tensor,
    *,
    config: DEQConfig | None = None,
    u: Tensor | None = None,
) -> Tensor:
    """VJP of ``u*`` with respect to ``W`` through the IFT.

    ``g`` is the cotangent of ``u*`` (same shape as ``u``). The
    backward linear solve uses exact ``sigma'``, not unrolled BPTT.
    """
    cfg = config if config is not None else DEQConfig()
    w = _as_W(W)
    x_t = _as_x(x, int(w.shape[0])).to(dtype=w.dtype, device=w.device)
    if u is None:
        u_star = deq_solve(w, x_t, spec, config=cfg).u
    else:
        u_star = torch.as_tensor(u, dtype=w.dtype, device=w.device)
        if u_star.shape != x_t.shape:
            raise ValueError(
                f"u shape {tuple(u_star.shape)} must match x shape {tuple(x_t.shape)}"
            )
    g_t = torch.as_tensor(g, dtype=w.dtype, device=w.device)
    if g_t.shape != u_star.shape:
        raise ValueError(
            f"g shape {tuple(g_t.shape)} must match u shape {tuple(u_star.shape)}"
        )
    sigp = _linear_map(u_star, w, spec, x_t)
    # A^T v = g  with A = I - D W, A^T = I - W^T D
    if u_star.ndim == 1:
        d = int(w.shape[0])
        eye = torch.eye(d, dtype=w.dtype, device=w.device)
        at = eye - w.T * sigp.reshape(1, d)
        v = torch.linalg.solve(at, g_t)
        return torch.outer(v * sigp, u_star)
    n = int(u_star.shape[0])
    d = int(w.shape[0])
    eye = torch.eye(d, dtype=w.dtype, device=w.device)
    at = eye.unsqueeze(0) - w.T.unsqueeze(0) * sigp.reshape(n, 1, d)
    v = torch.linalg.solve(at, g_t)
    return torch.einsum("bi,bj->ij", v * sigp, u_star)


def deq_du_dW(
    W: Tensor,
    x: Tensor,
    spec: str = "tanh",
    *,
    config: DEQConfig | None = None,
    u: Tensor | None = None,
) -> Tensor:
    """IFT Jacobian ``du*/dW`` with shape ``(..., d, d, d)``.

    Indexing is ``[u_component, W_row, W_col]`` after any batch axis.
    """
    cfg = config if config is not None else DEQConfig()
    w = _as_W(W)
    x_t = _as_x(x, int(w.shape[0])).to(dtype=w.dtype, device=w.device)
    if u is None:
        u_star = deq_solve(w, x_t, spec, config=cfg).u
    else:
        u_star = torch.as_tensor(u, dtype=w.dtype, device=w.device)
    sigp = _linear_map(u_star, w, spec, x_t)
    d = int(w.shape[0])
    eye = torch.eye(d, dtype=w.dtype, device=w.device)
    if u_star.ndim == 1:
        a = eye - sigp.reshape(d, 1) * w
        ainv = torch.linalg.inv(a)
        return cast(
            Tensor,
            ainv.unsqueeze(-1) * sigp.reshape(1, d, 1) * u_star.reshape(1, 1, d),
        )
    n = int(u_star.shape[0])
    a = eye.unsqueeze(0) - sigp.reshape(n, d, 1) * w.unsqueeze(0)
    ainv = torch.linalg.inv(a)
    return cast(
        Tensor,
        ainv.unsqueeze(-1) * sigp.reshape(n, 1, d, 1) * u_star.reshape(n, 1, 1, d),
    )


__all__ = [
    "DEQConfig",
    "DEQNotContractive",
    "DEQResult",
    "DEQSolverUnknown",
    "deq_du_dW",
    "deq_solve",
    "deq_vjp",
    "honesty_payload",
]
