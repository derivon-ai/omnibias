# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""JAX arrays around a host Gauss-Newton root-landing implementation.

These functions are eager host operations, not JIT/vmap/autodiff kernels.
The explicit first-order weight VJP :func:`newton_ift_vjp` is valid only at
regular isolated square roots. Underdetermined loci require a selection rule.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import jax.numpy as jnp
from jax import Array
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


def _check_seed(sys: EqualitySystem, x0: Array, max_iter: int, tol: float) -> None:
    if x0.ndim != 1 or x0.size != sys.dim:
        raise ValueError("x0 must be a vector with system.dim entries")
    if not jnp.issubdtype(x0.dtype, jnp.floating):
        raise ValueError("x0 must be a real floating-point array")
    if max_iter < 1 or not math.isfinite(tol) or tol <= 0:
        raise ValueError("max_iter must be positive and tol must be finite and positive")
    if any(not math.isfinite(v) for v in _as_tuple(x0)):
        raise ValueError("x0 must be finite")


def newton_project(
    sys: EqualitySystem,
    x0: Array,
    *,
    weights: Array | None = None,
    max_iter: int = 20,
    tol: float = 1e-12,
) -> Array:
    """Eager host root landing from a seed, without a nearest-point guarantee.

    Differentiation or JIT tracing through this host wrapper is unsupported.
    Use :func:`newton_ift_vjp` explicitly for eligible root sensitivities.
    """
    _check_seed(sys, x0, max_iter, tol)
    if weights is not None and (weights.ndim != 1 or weights.size != sys.n_terms):
        raise ValueError("weights must be a vector with system.n_terms entries")
    if weights is not None and not jnp.issubdtype(weights.dtype, jnp.floating):
        raise ValueError("weights must be a real floating-point array")
    w = [t.weight for t in sys.terms] if weights is None else _as_tuple(weights)
    sys_w = _with_weights(sys, w)
    result = newton_project_core(sys_w, _as_tuple(x0), max_iter=max_iter, tol=tol)
    return jnp.asarray(result.point, dtype=x0.dtype)


def newton_ift_vjp(
    sys: EqualitySystem,
    x_star: Array,
    weights: Array,
    g: Array,
    *,
    tol: float = 1e-12,
) -> Array:
    """Explicit first-order ``dL/dc = -F_c.T solve(F_x.T, g)`` at a root.

    The supplied point must satisfy the residual tolerance and have a regular
    square Jacobian. This eager host operation has no JIT/autodiff contract.
    A seed derivative is locally zero only for an isolated root in one basin.
    """
    if sys.codimension != sys.dim:
        raise ValueError(
            "implicit differentiation requires a regular isolated square root; "
            "an underdetermined locus needs a solution-selection rule"
        )
    if not math.isfinite(tol) or tol <= 0:
        raise ValueError("tol must be finite and positive")
    if x_star.ndim != 1 or x_star.size != sys.dim:
        raise ValueError("x_star must be a vector with system.dim entries")
    if weights.ndim != 1 or weights.size != sys.n_terms:
        raise ValueError("weights must be a vector with system.n_terms entries")
    if g.ndim != 1 or g.size != sys.dim:
        raise ValueError("g must be a vector with system.dim entries")
    if any(not jnp.issubdtype(value.dtype, jnp.floating) for value in (x_star, weights, g)):
        raise ValueError("x_star, weights and g must be real floating-point arrays")
    if any(not math.isfinite(value) for value in _as_tuple(g)):
        raise ValueError("g must be finite")
    sys_w = _with_weights(sys, _as_tuple(weights))
    xs = _as_tuple(x_star)
    f = residual_core(sys_w, xs)
    if any(not math.isfinite(v) for v in xs + f) or math.hypot(*f) > tol:
        raise ValueError("implicit differentiation requires a converged root")
    df = jacobian_core(sys_w, xs)
    sigma_min = smallest_singular(df)
    if not math.isfinite(sigma_min) or sigma_min <= 1e-10:
        raise ValueError("implicit differentiation requires a regular root (Jacobian rank lost)")
    dfdc = dF_d_weights(sys_w, xs)
    adjoint = jnp.linalg.solve(jnp.asarray(df, dtype=weights.dtype).T, g)
    return -jnp.asarray(dfdc, dtype=weights.dtype).T @ adjoint


def residual(sys: EqualitySystem, x: Array) -> Array:
    return jnp.asarray(residual_core(sys, _as_tuple(x)), dtype=x.dtype)


def jacobian(sys: EqualitySystem, x: Array) -> Array:
    return jnp.asarray(jacobian_core(sys, _as_tuple(x)), dtype=x.dtype)


def newton_result(
    sys: EqualitySystem, x0: Array, *, max_iter: int = 20, tol: float = 1e-12
) -> NewtonResult:
    """Eager host root-landing diagnostics, without an attached derivative."""
    _check_seed(sys, x0, max_iter, tol)
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
    branch: Literal["seed"] = "seed",
    max_iter: int = 20,
    tol: float = 1e-12,
    require_transversal: bool = True,
) -> LocusOutput:
    """Eager host root landing with diagnostics; ``branch`` is descriptive."""
    if branch != "seed":
        raise ValueError("only branch='seed' is implemented; nearest/all/numbered guarantees are unsupported")
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
