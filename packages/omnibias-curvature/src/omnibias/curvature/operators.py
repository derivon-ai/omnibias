# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Matrix-free symmetric solves with residual diagnostics and implicit gradients.

PCG is numerical. ``custom_linear_solve`` differentiates its defining linear
system, so gradients are meaningful only when the reported solve converges.
"""

from __future__ import annotations

from collections.abc import Callable
from math import isfinite
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

MatVec = Callable[[Array], Array]
State = tuple[Array, Array, Array, Array, Array, Array]


class PCGResult(NamedTuple):
    solution: Array
    residual_norm: Array
    relative_residual: Array
    iterations: Array
    converged: Array
    breakdown: Array


def pcg_solve(
    matvec: MatVec,
    rhs: Array,
    *,
    preconditioner: MatVec | None = None,
    rtol: float = 1e-8,
    atol: float = 0.0,
    max_iterations: int = 200,
) -> PCGResult:
    """Solve A x=b for a user-supplied SPD operator without assembling A.

    The preconditioner applies an SPD approximate inverse. Invalid curvature,
    nonfinite iterates and exhausted budgets are explicit nonconvergence.
    Options are static under JIT. Reverse derivatives use the symmetric
    implicit solve rather than differentiating an adaptive iteration count.
    """
    if rhs.ndim != 1 or rhs.size == 0 or jnp.issubdtype(rhs.dtype, jnp.complexfloating):
        raise ValueError("rhs must be a nonempty real vector")
    if rtol < 0 or atol < 0 or not isfinite(rtol + atol) or max_iterations < 1:
        raise ValueError("finite nonnegative tolerances and positive iteration budget required")
    apply_m: MatVec = (lambda x: x) if preconditioner is None else preconditioner

    def solve(op: MatVec, b: Array) -> tuple[Array, tuple[Array, Array]]:
        x = jnp.zeros_like(b)
        r = b
        z = apply_m(r)
        p = z
        rz = jnp.vdot(r, z).real
        target = jnp.maximum(atol, rtol * jnp.linalg.norm(b))
        invalid = (~jnp.all(jnp.isfinite(b))) | (~jnp.all(jnp.isfinite(z)))
        initial: State = (jnp.asarray(0), x, r, p, rz, invalid)

        def cond(state: State) -> Array:
            k, _, res, _, _, bad = state
            return jnp.asarray((k < max_iterations) & (jnp.linalg.norm(res) > target) & (~bad))

        def body(state: State) -> State:
            k, x, r, p, rz, bad = state
            ap = op(p)
            denominator = jnp.vdot(p, ap).real
            valid = (denominator > 0) & (rz > 0) & jnp.isfinite(denominator) & jnp.isfinite(rz)
            alpha = jnp.where(valid, rz / jnp.where(valid, denominator, 1.0), 0.0)
            xn = x + alpha * p
            rn = r - alpha * ap
            zn = apply_m(rn)
            rzn = jnp.vdot(rn, zn).real
            beta = jnp.where(valid, rzn / jnp.where(valid, rz, 1.0), 0.0)
            pn = zn + beta * p
            bad = bad | (~valid) | (~jnp.all(jnp.isfinite(xn))) | (~jnp.all(jnp.isfinite(rn)))
            return k + 1, xn, rn, pn, rzn, bad

        final = jax.lax.while_loop(cond, body, initial)
        return final[1], (final[0], final[-1])

    x, (iterations, breakdown) = jax.lax.custom_linear_solve(
        matvec, rhs, solve=solve, symmetric=True, has_aux=True
    )
    residual = jnp.linalg.norm(matvec(x) - rhs)
    norm = jnp.linalg.norm(rhs)
    relative = residual / jnp.where(norm > 0, norm, 1.0)
    converged = (residual <= jnp.maximum(atol, rtol * norm)) & (~breakdown) & jnp.isfinite(residual)
    return PCGResult(x, residual, relative, iterations, converged, breakdown)


__all__ = ["MatVec", "PCGResult", "pcg_solve"]
