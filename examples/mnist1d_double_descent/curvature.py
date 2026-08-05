# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Exact-curvature instrumentation of the loss landscape.

Everything here rides on the exact matrix-free Hessian-vector product in
:mod:`omnibias.curvature.torch`. :func:`spectrum_snapshot` auto-switches by the
parameter count ``P``:

* narrow (``P <= dense_max_params``): materialise the exact ``(P, P)`` Hessian
  and read off the *full* eigenspectrum -- ``lambda_max``, ``lambda_min``, the
  trace, the Frobenius norm, the condition number, the negative-curvature
  fraction, and an effective rank;
* wide: estimate ``(lambda_min, lambda_max)`` by power iteration and the
  trace / Frobenius norm by the unbiased Hutchinson estimators.

:func:`dense_vs_matrix_free_gap` is the numerical-consistency check (the
matrix-free estimators must reproduce the dense oracle at a shared narrow width),
and :func:`exact_sharpness_gap` exposes the exact ascent-free SAM inner-max gap as
a scalar diagnostic.

Honesty: the HVP / dense Hessian are exact (autograd), and where the net uses the
Riccati (tanh) activation the sigma-tower entering them is the closed-form one.
``lambda_max`` by power iteration is exact only in the iteration limit; the
Hutchinson trace / Frobenius are unbiased stochastic estimators.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
from omnibias.curvature.torch import (
    HessianOperator,
    hessian_eigenvalue_extremes,
    hutchinson_frobenius_sq,
    hutchinson_trace,
    sam_sharpness_gap,
    top_eigenvalue,
)
from torch import Tensor

Params = list[Tensor]
_TOL = 1e-8


@dataclass
class CurvatureSnapshot:
    """One exact-curvature reading of ``loss`` at the current parameters."""

    method: str  # "dense" | "matrix_free"
    n_params: int
    grad_norm: float
    lambda_max: float
    lambda_min: float
    trace: float
    frobenius_sq: float
    condition_number: float
    neg_curvature_frac: float  # dense only, else NaN
    effective_rank: float  # dense only, else NaN

    def as_dict(self) -> dict[str, float | int | str]:
        return asdict(self)


def _generator(device: torch.device, seed: int | None) -> torch.Generator | None:
    if seed is None:
        return None
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    return gen


def _dense_from_operator(op: HessianOperator) -> Tensor:
    """Materialise the exact ``(P, P)`` Hessian by HVPs on the standard basis (reuses ``op``)."""
    params = op.params
    sizes = [p.numel() for p in params]
    total = int(sum(sizes))
    rows: list[Tensor] = []
    for i in range(total):
        e: list[Tensor] = []
        j = i
        for p, n in zip(params, sizes, strict=True):
            block = torch.zeros(n, dtype=p.dtype, device=p.device)
            if 0 <= j < n:
                block[j] = 1.0
            j -= n
            e.append(block.reshape(p.shape))
        hv = op.hvp(e, create_graph=False)
        rows.append(torch.cat([h.reshape(-1) for h in hv]).detach())
    return torch.stack(rows)


def _dense_metrics(eig: Tensor) -> tuple[float, float, float, float, float, float, float]:
    lam_max = float(eig[-1])
    lam_min = float(eig[0])
    trace = float(eig.sum())
    fro = float((eig * eig).sum())
    pos = eig[eig > _TOL]
    if pos.numel() > 0 and lam_max > 0.0:
        cond = float(lam_max / float(pos.min()))
    else:
        cond = math.inf
    neg_frac = float((eig < -_TOL).to(torch.float64).mean())
    abs_sum = float(eig.abs().sum())
    eff_rank = float((abs_sum * abs_sum) / (fro + _TOL)) if fro > 0.0 else 0.0
    return lam_max, lam_min, trace, fro, cond, neg_frac, eff_rank


def spectrum_snapshot(
    loss: Tensor,
    params: Params,
    *,
    dense_max_params: int = 1500,
    power_iters: int = 40,
    hutch_samples: int = 8,
    seed: int | None = 0,
) -> CurvatureSnapshot:
    """Exact-curvature reading of ``loss`` w.r.t. ``params`` (dense if small, else matrix-free)."""
    params = list(params)
    if not params:
        raise ValueError("params is empty")
    n_params = int(sum(p.numel() for p in params))
    op = HessianOperator(loss, params)
    grad_norm = float(op.grad_norm.detach())
    device = params[0].device

    if n_params <= dense_max_params:
        hess = _dense_from_operator(op)
        hess = 0.5 * (hess + hess.transpose(0, 1))
        eig = torch.linalg.eigvalsh(hess.to(torch.float64)).detach()
        lam_max, lam_min, trace, fro, cond, neg_frac, eff_rank = _dense_metrics(eig)
        method = "dense"
    else:
        gen = _generator(device, seed)
        lam_min, lam_max = hessian_eigenvalue_extremes(
            loss, params, iters=power_iters, generator=gen
        )
        trace = float(
            hutchinson_trace(loss, params, n_samples=hutch_samples, generator=gen).detach()
        )
        fro = float(
            hutchinson_frobenius_sq(loss, params, n_samples=hutch_samples, generator=gen).detach()
        )
        denom = max(abs(lam_min), _TOL)
        cond = float(lam_max / denom) if lam_max > 0.0 else math.inf
        neg_frac = math.nan
        eff_rank = math.nan
        method = "matrix_free"

    return CurvatureSnapshot(
        method=method,
        n_params=n_params,
        grad_norm=grad_norm,
        lambda_max=lam_max,
        lambda_min=lam_min,
        trace=trace,
        frobenius_sq=fro,
        condition_number=cond,
        neg_curvature_frac=neg_frac,
        effective_rank=eff_rank,
    )


def dense_vs_matrix_free_gap(
    loss: Tensor, params: Params, *, power_iters: int = 60, seed: int = 0
) -> float:
    """Relative gap between the dense ``lambda_max`` oracle and the power-iteration estimate.

    Used to validate that the matrix-free estimators reproduce the exact spectrum
    at a shared (narrow) width. Returns ``|lam_dense - lam_free| / |lam_dense|``.
    """
    params = list(params)
    op = HessianOperator(loss, params)
    hess = _dense_from_operator(op)
    hess = 0.5 * (hess + hess.transpose(0, 1))
    lam_dense = float(torch.linalg.eigvalsh(hess.to(torch.float64))[-1])
    gen = _generator(params[0].device, seed)
    lam_free = float(top_eigenvalue(loss, params, iters=power_iters, generator=gen))
    return abs(lam_dense - lam_free) / max(abs(lam_dense), _TOL)


def exact_sharpness_gap(
    loss: Tensor, params: Params, *, rho: float = 0.05, iters: int = 30, seed: int = 0
) -> float:
    """Exact ascent-free SAM inner-max gap ``rho|g| + rho^2/2 max(lam_max, 0)`` (a scalar)."""
    params = list(params)
    gen = _generator(params[0].device, seed)
    gap = sam_sharpness_gap(loss, params, rho=rho, iters=iters, generator=gen, differentiable=False)
    return float(gap.detach())


__all__ = [
    "CurvatureSnapshot",
    "dense_vs_matrix_free_gap",
    "exact_sharpness_gap",
    "spectrum_snapshot",
]
