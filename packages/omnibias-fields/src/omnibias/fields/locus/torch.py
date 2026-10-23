# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Host Gauss-Newton root landing and first-order isolated-root sensitivities.

The iteration does not generally compute a nearest point. Implicit derivatives
are supported only at converged, regular square roots. Underdetermined roots
need a solution-selection rule; use an affine projection or differentiate the
explicit iteration when that is the intended map.
"""

from __future__ import annotations

import math
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
    smallest_singular,
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
from torch.autograd.function import once_differentiable


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


def _check_inputs(sys: EqualitySystem, x0: Tensor, weights: Tensor, max_iter: int, tol: float) -> None:
    if x0.ndim != 1 or x0.numel() != sys.dim:
        raise ValueError("x0 must be a vector with system.dim entries")
    if weights.ndim != 1 or weights.numel() != sys.n_terms:
        raise ValueError("weights must be a vector with system.n_terms entries")
    if not x0.is_floating_point() or not weights.is_floating_point():
        raise ValueError("x0 and weights must be real floating-point tensors")
    if not bool(torch.isfinite(x0).all()) or not bool(torch.isfinite(weights).all()):
        raise ValueError("x0 and weights must be finite")
    if max_iter < 1 or not math.isfinite(tol) or tol <= 0:
        raise ValueError("max_iter must be positive and tol must be finite and positive")


def _require_isolated_root(sys: EqualitySystem, result: NewtonResult, tol: float) -> None:
    if sys.codimension != sys.dim:
        raise ValueError(
            "implicit differentiation requires a regular isolated square root; "
            "an underdetermined locus needs a solution-selection rule"
        )
    if not result.converged or any(not math.isfinite(v) for v in result.point):
        raise ValueError("implicit differentiation requires a converged root")
    f = residual_core(sys, result.point)
    if any(not math.isfinite(v) for v in f) or math.hypot(*f) > tol:
        raise ValueError("implicit differentiation requires a converged root")
    sigma_min = smallest_singular(jacobian_core(sys, result.point))
    if not math.isfinite(sigma_min) or sigma_min <= 1e-10:
        raise ValueError("implicit differentiation requires a regular root (Jacobian rank lost)")


class _NewtonIFT(torch.autograd.Function):
    """Forward: host root landing. Backward: first-order square-root IFT."""

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
        _require_isolated_root(sys_w, result, tol)
        x_star = torch.tensor(result.point, dtype=x0.dtype, device=x0.device)
        ctx.sys = sys
        ctx.save_for_backward(x_star, weights)
        return x_star

    @staticmethod
    @once_differentiable
    def backward(ctx: Any, grad_output: Tensor) -> tuple[Tensor | None, Tensor | None, None, None, None]:
        x_star, weights = ctx.saved_tensors
        sys_w = _with_weights(ctx.sys, _as_tuple(weights))
        xs = _as_tuple(x_star)
        df = jacobian_core(sys_w, xs)
        dfdc = dF_d_weights(sys_w, xs)  # m x n_c
        # Solve the square adjoint directly; forming the normal-equation Gram
        # matrix would unnecessarily square its condition number.
        solve_dtype = torch.float64 if torch.float64 in (x_star.dtype, weights.dtype) else torch.float32
        jac = torch.tensor(df, dtype=solve_dtype, device=x_star.device)
        parameter_jac = torch.tensor(dfdc, dtype=solve_dtype, device=x_star.device)
        adjoint = torch.linalg.solve(jac.T, grad_output.to(dtype=solve_dtype))
        grad_w = (-(parameter_jac.T @ adjoint)).to(dtype=weights.dtype, device=weights.device)
        # A regular isolated root is locally independent of its initial seed
        # within the same attraction basin. This is not true for a manifold.
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
    """Land on a root from ``x0`` using host Gauss-Newton.

    Gradient-enabled calls require a converged, regular square root and provide
    first-order implicit sensitivities. The seed derivative is then locally
    zero within its attraction basin. Forward-only calls also accept
    underdetermined systems. There is no nearest-point or branch guarantee.
    """
    if weights is None:
        w = torch.tensor(
            [t.weight for t in sys.terms],
            dtype=x0.dtype,
            device=x0.device,
        )
    else:
        w = weights
    _check_inputs(sys, x0, w, max_iter, tol)
    if not torch.is_grad_enabled() or not (x0.requires_grad or w.requires_grad):
        sys_w = _with_weights(sys, _as_tuple(w))
        result = newton_project_core(sys_w, _as_tuple(x0), max_iter=max_iter, tol=tol)
        return torch.tensor(result.point, dtype=x0.dtype, device=x0.device)
    return cast(Tensor, _NewtonIFT.apply(x0, w, sys, max_iter, tol))


def newton_project_unrolled(
    sys: EqualitySystem,
    x0: Tensor,
    *,
    weights: Tensor,
    max_iter: int = 20,
    tol: float = 1e-12,
) -> Tensor:
    """Differentiate the executed codimension-one Gauss-Newton iteration.

    This is the iteration map, including its seed dependence, rather than an
    implicit root derivative or a nearest-point projection. The stopping
    decision is discrete and its derivative is not included.
    """
    if sys.codimension != 1:
        raise ValueError("newton_project_unrolled is the G5 probe for m=1")
    _check_inputs(sys, x0, weights, max_iter, tol)
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
        if not bool(torch.isfinite(gram).detach()) or float(gram.detach()) <= 1e-20:
            raise ValueError("Gauss-Newton iteration requires a regular Jacobian")
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
    """Host root-landing diagnostics, without an attached derivative."""
    weights = x0.new_tensor([term.weight for term in sys.terms])
    _check_inputs(sys, x0, weights, max_iter, tol)
    return newton_project_core(sys, _as_tuple(x0), max_iter=max_iter, tol=tol)


@dataclass
class LocusOutput:
    """Required diagnostics. The locus is a constraint manifold, not a PDE solver."""

    point: Tensor
    branch: Tensor
    condition: Tensor
    converged: Tensor


class EqualityLocusLayer(nn.Module):
    """Seed-selected root landing with diagnostics and restricted IFT backward.

    ``branch`` diagnostics describe the returned root; they do not establish
    branch completeness or nearest-point selection.
    """

    def __init__(
        self,
        system: EqualitySystem,
        *,
        branch: Literal["seed"] = "seed",
        max_iter: int = 20,
        tol: float = 1e-12,
        require_transversal: bool = True,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if branch != "seed":
            raise ValueError("only branch='seed' is implemented; nearest/all/numbered guarantees are unsupported")
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
