# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""JAX Newton projection on an equality locus (theory 01-09).

Forward is Gauss-Newton (host, matching the torch landing point in float64).
The implicit-function-theorem VJP is :func:`newton_ift_vjp`; differentiating
through the unrolled loop is refused.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import jax.numpy as jnp
from jax import Array
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


def _as_tuple(x: Array) -> tuple[float, ...]:
    return tuple(float(v) for v in x.reshape(-1).tolist())


def _with_weights(sys: EqualitySystem, weights: Sequence[float]) -> EqualitySystem:
    if len(weights) != sys.n_terms:
        raise ValueError("weights length must match the number of units")
    terms = tuple(
        UnitTerm(t.order, float(w), t.normal, t.bias)
        for t, w in zip(sys.terms, weights, strict=True)
    )
    return EqualitySystem(terms, sys.base)


def newton_project(
    sys: EqualitySystem,
    x0: Array,
    *,
    weights: Array | None = None,
    max_iter: int = 20,
    tol: float = 1e-12,
) -> Array:
    """Gauss-Newton projection onto the locus."""
    w = [t.weight for t in sys.terms] if weights is None else _as_tuple(weights)
    sys_w = _with_weights(sys, w)
    result = newton_project_core(sys_w, _as_tuple(x0), max_iter=max_iter, tol=tol)
    return jnp.asarray(result.point, dtype=x0.dtype)


def newton_ift_vjp(
    sys: EqualitySystem,
    x_star: Array,
    weights: Array,
    g: Array,
) -> Array:
    """``dL/dc`` via ``dx*/dc = -DF^+ dF/dc``. Memory is independent of iteration count."""
    sys_w = _with_weights(sys, _as_tuple(weights))
    xs = _as_tuple(x_star)
    df = jacobian_core(sys_w, xs)
    pinv = pseudoinverse(df)
    dfdc = dF_d_weights(sys_w, xs)
    gv = _as_tuple(g)
    m = len(df)
    pinv_t_g = tuple(sum(pinv[k][j] * gv[k] for k in range(sys_w.dim)) for j in range(m))
    grad_w = []
    for j in range(sys_w.n_terms):
        acc = 0.0
        for i in range(m):
            acc -= dfdc[i][j] * pinv_t_g[i]
        grad_w.append(acc)
    return jnp.asarray(grad_w, dtype=weights.dtype)


def residual(sys: EqualitySystem, x: Array) -> Array:
    return jnp.asarray(residual_core(sys, _as_tuple(x)), dtype=x.dtype)


def jacobian(sys: EqualitySystem, x: Array) -> Array:
    return jnp.asarray(jacobian_core(sys, _as_tuple(x)), dtype=x.dtype)


def newton_result(
    sys: EqualitySystem, x0: Array, *, max_iter: int = 20, tol: float = 1e-12
) -> NewtonResult:
    return newton_project_core(sys, _as_tuple(x0), max_iter=max_iter, tol=tol)


@dataclass
class LocusOutput:
    point: Array
    branch: Array
    condition: Array
    converged: Array


def equality_locus_apply(
    sys: EqualitySystem,
    x0: Array,
    *,
    max_iter: int = 20,
    tol: float = 1e-12,
    require_transversal: bool = True,
) -> LocusOutput:
    result = newton_result(sys, x0, max_iter=max_iter, tol=tol)
    point = newton_project(sys, x0, max_iter=max_iter, tol=tol)
    sigma_min = float(result.condition)
    cond = 1.0 / max(sigma_min, 1e-30)
    ok = bool(result.converged)
    if require_transversal:
        ok = ok and bool(result.transversal)
    return LocusOutput(
        point=point,
        branch=jnp.asarray(result.branch, dtype=jnp.int64),
        condition=jnp.asarray(cond, dtype=x0.dtype),
        converged=jnp.asarray(ok),
    )


__all__ = [
    "LocusOutput",
    "equality_locus_apply",
    "jacobian",
    "newton_ift_vjp",
    "newton_project",
    "newton_result",
    "residual",
]
