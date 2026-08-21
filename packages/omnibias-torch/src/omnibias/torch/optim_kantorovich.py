# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Kantorovich-accepted Newton (theory 08-04), PyTorch twin.

A Gauss–Newton or cubic trial is legal only when
:func:`omnibias.core.verified.kantorovich.kantorovich_accept_step`
returns a nonempty unique-zero ball of the **finite** residual map.
Torch produces the trial point and a float inverse ``A \approx DF^{-1}``;
the interval residual and Jacobian stay in
:mod:`omnibias.core.verified`.  Empty ball is a valid reject.

The sealed payload records ``continuum_pde_claim: false``.
``theorem_prover_verified`` is never asserted (sound-enclosure tier).
Do not wrap the driver in ``torch.compile``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from omnibias.core.verified.kantorovich import (
    CONTINUUM_PDE_CLAIM_KEY,
    FINITE_RESIDUAL_CLAIM,
    IntervalJac,
    IntervalMap,
    KantorovichAccept,
    polynomial_sqrt2_maps,
)
from omnibias.core.verified.kantorovich import (
    kantorovich_accept_step as _accept_core,
)
from omnibias.core.verified.kantorovich import (
    select_accepted_params as _select_core,
)

import torch
from torch import Tensor
from torch.func import jacrev

ResidualFn = Callable[[Tensor], Tensor]


def _as_float_vector(params: Tensor | Sequence[float]) -> list[float]:
    vec = torch.as_tensor(params).reshape(-1)
    return [float(v) for v in vec]


def _as_float_matrix(
    a_inv: Tensor | Sequence[Sequence[float]],
) -> list[list[float]]:
    mat = torch.as_tensor(a_inv)
    if mat.ndim == 0:
        return [[float(mat)]]
    if mat.ndim == 1:
        n = int(mat.numel())
        if n == 1:
            return [[float(mat.reshape(-1)[0])]]
        return []
    if mat.ndim != 2:
        return []
    rows, cols = int(mat.shape[0]), int(mat.shape[1])
    return [[float(mat[i, j]) for j in range(cols)] for i in range(rows)]


def approximate_inverse_jacobian(
    residual_fn: ResidualFn, params: Tensor
) -> list[list[float]]:
    """Float ``A \approx DF(theta)^{-1}`` at ``params``; empty if not square."""
    jac = jacrev(residual_fn)(params)
    mat = torch.as_tensor(jac)
    if mat.ndim == 1:
        mat = mat.reshape(1, -1)
    if mat.ndim != 2 or int(mat.shape[0]) != int(mat.shape[1]):
        return []
    try:
        inverse = torch.linalg.inv(mat)
    except (RuntimeError, ValueError):
        return []
    return _as_float_matrix(inverse)


def kantorovich_accept_step(
    residual_iv: IntervalMap,
    jac_iv: IntervalJac,
    a_inv: Tensor | Sequence[Sequence[float]],
    trial_params: Tensor | Sequence[float],
    *,
    lipschitz_df: float,
    r_max: float,
    claim: str = FINITE_RESIDUAL_CLAIM,
) -> KantorovichAccept:
    """Assemble NK bounds and consult the radii polynomial (core)."""
    return _accept_core(
        residual_iv,
        jac_iv,
        _as_float_matrix(a_inv),
        _as_float_vector(trial_params),
        lipschitz_df=float(lipschitz_df),
        r_max=float(r_max),
        claim=claim,
    )


def select_accepted_params(
    current: Tensor,
    trial: Tensor,
    decision: KantorovichAccept,
) -> Tensor:
    """Return ``trial`` on a unique-zero ball, otherwise keep ``current``."""
    chosen = _select_core(
        _as_float_vector(current), _as_float_vector(trial), decision
    )
    out = torch.as_tensor(chosen, dtype=current.dtype, device=current.device)
    return out.reshape(current.shape)


def kantorovich_gated_gauss_newton_step(
    residual_fn: ResidualFn,
    residual_iv: IntervalMap,
    jac_iv: IntervalJac,
    params: Tensor,
    *,
    lipschitz_df: float,
    r_max: float,
    damping: float = 1e-3,
    claim: str = FINITE_RESIDUAL_CLAIM,
) -> tuple[Tensor, KantorovichAccept]:
    """One Gauss–Newton trial, kept only if the unique-zero ball is nonempty."""
    from omnibias.torch.optim import GaussNewton

    gn = GaussNewton(damping=damping, max_line_search=8)
    trial, _info = gn.step(residual_fn, params)
    a_est = approximate_inverse_jacobian(residual_fn, trial)
    decision = kantorovich_accept_step(
        residual_iv,
        jac_iv,
        a_est,
        trial,
        lipschitz_df=lipschitz_df,
        r_max=r_max,
        claim=claim,
    )
    return select_accepted_params(params, trial, decision), decision


__all__ = [
    "CONTINUUM_PDE_CLAIM_KEY",
    "FINITE_RESIDUAL_CLAIM",
    "KantorovichAccept",
    "approximate_inverse_jacobian",
    "kantorovich_accept_step",
    "kantorovich_gated_gauss_newton_step",
    "polynomial_sqrt2_maps",
    "select_accepted_params",
]
