# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""PyTorch Newton projection on an equality locus (theory 01-09).

``newton_project`` is Gauss-Newton in the forward pass. Its gradient uses the
implicit function theorem (``dx* / d theta = - DF^+ dF/d theta``), not the
unrolled iteration, so memory does not scale with iteration count.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal, cast

import torch
import torch.nn as nn
from omnibias.core.locus import (
    EqualitySystem,
    NewtonResult,
    UnitTerm,
    dF_d_weights,
    pseudoinverse,
)
from omnibias.core.locus import (
    jacobian as jacobian_core,
)
from omnibias.core.locus import (
    newton_project as newton_project_core,
)
from omnibias.core.locus import (
    residual as residual_core,
)
from omnibias.core.tanh_method import PDESpec, TravellingWaveAnsatz, verify_exact
from torch import Tensor


def _as_tuple(x: Tensor) -> tuple[float, ...]:
    return tuple(float(v) for v in x.detach().reshape(-1).tolist())


def _with_weights(sys: EqualitySystem, weights: Sequence[float]) -> EqualitySystem:
    if len(weights) != sys.n_terms:
        raise ValueError("weights length must match the number of units")
    terms = tuple(
        UnitTerm(t.order, float(w), t.normal, t.bias)
        for t, w in zip(sys.terms, weights, strict=True)
    )
    return EqualitySystem(terms, sys.base)


class _NewtonIFT(torch.autograd.Function):
    """Forward: Gauss-Newton. Backward: IFT on the landing point."""

    @staticmethod
    def forward(
        ctx: Any,
        x0: Tensor,
        weights: Tensor,
        sys: EqualitySystem,
        max_iter: int,
        tol: float,
    ) -> Tensor:
        sys_w = _with_weights(sys, _as_tuple(weights))
        result = newton_project_core(sys_w, _as_tuple(x0), max_iter=max_iter, tol=tol)
        x_star = torch.tensor(result.point, dtype=x0.dtype, device=x0.device)
        ctx.sys = sys
        ctx.save_for_backward(x_star, weights)
        return x_star

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> tuple[Tensor | None, Tensor | None, None, None, None]:
        x_star, weights = ctx.saved_tensors
        sys_w = _with_weights(ctx.sys, _as_tuple(weights))
        xs = _as_tuple(x_star)
        df = jacobian_core(sys_w, xs)
        pinv = pseudoinverse(df)  # D x m
        dfdc = dF_d_weights(sys_w, xs)  # m x n_c
        g = _as_tuple(grad_output)
        # dL/dc = - (dF/dc)^T (DF^+)^T g
        # (DF^+)^T g is m-vector: pinv^T g
        m = len(df)
        n_c = sys_w.n_terms
        pinv_t_g = tuple(sum(pinv[k][j] * g[k] for k in range(sys_w.dim)) for j in range(m))
        grad_w = torch.zeros(n_c, dtype=weights.dtype, device=weights.device)
        for j in range(n_c):
            acc = 0.0
            for i in range(m):
                acc -= dfdc[i][j] * pinv_t_g[i]
            grad_w[j] = acc
        # x0 is a seed; IFT treats the landing point as independent of x0
        # once the iteration has converged (projection onto the manifold).
        grad_x0 = torch.zeros_like(x_star)
        return grad_x0, grad_w, None, None, None


def newton_project(
    sys: EqualitySystem,
    x0: Tensor,
    *,
    weights: Tensor | None = None,
    max_iter: int = 20,
    tol: float = 1e-12,
) -> Tensor:
    """Gauss-Newton projection. Differentiable through the IFT, not the loop."""
    if weights is None:
        w = torch.tensor(
            [t.weight for t in sys.terms],
            dtype=x0.dtype,
            device=x0.device,
        )
    else:
        w = weights
    return cast(Tensor, _NewtonIFT.apply(x0, w, sys, max_iter, tol))


def newton_project_unrolled(
    sys: EqualitySystem,
    x0: Tensor,
    *,
    weights: Tensor,
    max_iter: int = 20,
    tol: float = 1e-12,
) -> Tensor:
    """Unrolled Gauss-Newton kept on the autodiff graph (G5 comparison only).

    Implemented for codimension 1 (the IFT-vs-unrolled gate). Higher
    codimension should use :func:`newton_project`.
    """
    if sys.codimension != 1:
        raise ValueError("newton_project_unrolled is the G5 probe for m=1")
    from omnibias.torch.activations import get_activation

    act = get_activation(sys.base)
    x = x0.clone()
    normals = [
        torch.tensor(t.normal, dtype=x.dtype, device=x.device) for t in sys.terms
    ]
    for _ in range(max_iter):
        zs = []
        for t, nrm in zip(sys.terms, normals, strict=True):
            zs.append((x * nrm).sum() + t.bias)
        sigmas = [cast(Tensor, act.fastpath(z, t.order)) for t, z in zip(sys.terms, zs, strict=True)]
        sigmas_p = [
            cast(Tensor, act.fastpath(z, t.order + 1)) for t, z in zip(sys.terms, zs, strict=True)
        ]
        f = weights[0] * sigmas[0] - weights[1] * sigmas[1]
        if float(f.detach().abs()) <= tol:
            return x
        df = weights[0] * sigmas_p[0] * normals[0] - weights[1] * sigmas_p[1] * normals[1]
        gram = (df * df).sum()
        x = x - df * (f / gram)
    return x


def residual(sys: EqualitySystem, x: Tensor) -> Tensor:
    return torch.tensor(residual_core(sys, _as_tuple(x)), dtype=x.dtype, device=x.device)


def jacobian(sys: EqualitySystem, x: Tensor) -> Tensor:
    return torch.tensor(jacobian_core(sys, _as_tuple(x)), dtype=x.dtype, device=x.device)


def locus_tangent_tensor(sys: EqualitySystem, x: Tensor) -> Tensor:
    from omnibias.core.locus import locus_tangent as _lt

    basis = _lt(sys, _as_tuple(x))
    if not basis:
        return torch.zeros((0, sys.dim), dtype=x.dtype, device=x.device)
    return torch.tensor(basis, dtype=x.dtype, device=x.device)


def newton_result(
    sys: EqualitySystem, x0: Tensor, *, max_iter: int = 20, tol: float = 1e-12
) -> NewtonResult:
    return newton_project_core(sys, _as_tuple(x0), max_iter=max_iter, tol=tol)


@dataclass
class LocusOutput:
    """Required diagnostics. The locus is a constraint manifold, not a PDE solver."""

    point: Tensor
    branch: Tensor
    condition: Tensor
    converged: Tensor


class EqualityLocusLayer(nn.Module):
    """Gauss-Newton projection with IFT backward. Always returns branch / condition."""

    def __init__(
        self,
        system: EqualitySystem,
        *,
        branch: int | Literal["nearest", "all"] = "nearest",
        max_iter: int = 20,
        tol: float = 1e-12,
        require_transversal: bool = True,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.system = system
        self.branch_mode = branch
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.require_transversal = bool(require_transversal)
        self._dtype = torch.get_default_dtype() if dtype is None else dtype

    def forward(self, x0: Tensor) -> LocusOutput:
        result = newton_result(self.system, x0, max_iter=self.max_iter, tol=self.tol)
        point = newton_project(self.system, x0, max_iter=self.max_iter, tol=self.tol)
        sigma_min = float(result.condition)
        cond = 1.0 / max(sigma_min, 1e-30)
        ok = bool(result.converged)
        if self.require_transversal:
            ok = ok and bool(result.transversal)
        return LocusOutput(
            point=point,
            branch=torch.tensor(result.branch, dtype=torch.int64, device=x0.device),
            condition=torch.tensor(cond, dtype=x0.dtype, device=x0.device),
            converged=torch.tensor(ok, dtype=torch.bool, device=x0.device),
        )


class AnsatzSolutionField(nn.Module):
    """Level-2: equality inside a named tanh-class. Not a general PDE solver."""

    def __init__(self, pde: PDESpec, ansatz: TravellingWaveAnsatz) -> None:
        super().__init__()
        if not verify_exact(pde, ansatz):
            raise ValueError(
                f"ansatz {ansatz.kind!r} failed symbolic verification for {pde.name!r}; "
                "no claim is made outside a named ansatz class"
            )
        self.pde = pde
        self.ansatz = ansatz

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        from omnibias.core.tanh_method import evaluate_ansatz

        xs = x.detach().reshape(-1).tolist()
        ts = t.detach().reshape(-1).tolist()
        vals = [
            evaluate_ansatz(self.ansatz, float(xi), float(ti))
            for xi, ti in zip(xs, ts, strict=True)
        ]
        return torch.tensor(vals, dtype=x.dtype, device=x.device).reshape(x.shape)

    def certificate(self) -> dict[str, object]:
        return {
            "class": self.pde.name,
            "verified": True,
            "level3_general_solver": False,
        }


__all__ = [
    "AnsatzSolutionField",
    "EqualityLocusLayer",
    "LocusOutput",
    "jacobian",
    "locus_tangent_tensor",
    "newton_project",
    "newton_project_unrolled",
    "newton_result",
    "residual",
]
