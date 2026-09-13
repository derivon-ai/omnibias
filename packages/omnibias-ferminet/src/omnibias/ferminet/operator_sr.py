# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Live-pytree QGT actions and matrix-free stochastic reconfiguration (JAX).

This is a numerical covariance of the supplied weighted observations. Sampling
uncertainty is separate from the PCG residual. The dense SR API is unchanged.
Requires the existing ``curvature`` extra.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from jax import Array
from jax.flatten_util import ravel_pytree
from omnibias.curvature.operators import PCGResult, pcg_solve

LogPsi = Callable[[Any, Array], Array]


@dataclass(frozen=True)
class QGTOperator:
    """Symmetric real QGT on real parameters, including complex log amplitudes."""

    matvec: Callable[[Array], Array]
    energy_gradient: Callable[[Array], Array]
    dimension: int
    sample_count: int
    weights: Array
    sampling_kind: str


def qgt_operator(
    log_psi_fn: LogPsi,
    params: Any,
    samples: Array,
    *,
    weights: Array | None = None,
    chunk_size: int | None = None,
    sampling_kind: str = "monte_carlo",
) -> QGTOperator:
    """Covariance action by centered JVP/VJP; no N×P or P×P Jacobian.

    Weights are nonnegative and normalized by their sum. For traced weights,
    callers must validate that condition before JIT. Nontraced invalid weights
    are rejected. ``exact_enumeration`` labels a finite distribution, not a
    claim that floating-point summation is rigorous.
    """
    flat, unravel = ravel_pytree(params)
    if (
        samples.ndim < 1
        or samples.shape[0] < 1
        or flat.size < 1
        or jnp.issubdtype(flat.dtype, jnp.complexfloating)
    ):
        raise ValueError("nonempty samples and real parameter pytree required")
    if sampling_kind not in ("monte_carlo", "exact_enumeration", "quadrature"):
        raise ValueError("unknown sampling kind")
    n = int(samples.shape[0])
    size = n if chunk_size is None else chunk_size
    if size < 1:
        raise ValueError("chunk_size must be positive")
    w = jnp.ones(n, dtype=flat.dtype) if weights is None else jnp.asarray(weights, dtype=flat.dtype)
    if w.shape != (n,):
        raise ValueError("one weight per sample required")
    if not isinstance(w, jax.core.Tracer):
        import numpy as np

        host = np.asarray(w)
        if not np.all(np.isfinite(host)) or np.any(host < 0) or np.sum(host) <= 0:
            raise ValueError("finite nonnegative weights with positive sum required")
    w = w / jnp.sum(w)
    chunks = [(samples[i : i + size], w[i : i + size], i) for i in range(0, n, size)]

    def parts(theta: Array, batch: Array) -> tuple[Array, Array]:
        result = jax.vmap(lambda x: log_psi_fn(unravel(theta), x))(batch)
        if result.ndim != 1:
            raise ValueError("log_psi_fn must return a scalar per sample")
        return jnp.real(result), jnp.imag(result)

    def action(v: Array) -> Array:
        if v.shape != flat.shape:
            raise ValueError("QGT vector dimension mismatch")
        tangent = []
        mean_r = jnp.zeros((), dtype=flat.dtype)
        mean_i = jnp.zeros_like(mean_r)
        for batch, weight, _ in chunks:
            _, (dr, di) = jax.jvp(partial(parts, batch=batch), (flat,), (v,))
            tangent.append((dr, di))
            mean_r += jnp.dot(weight, dr)
            mean_i += jnp.dot(weight, di)
        result = jnp.zeros_like(flat)
        for (batch, weight, _), (dr, di) in zip(chunks, tangent, strict=True):
            _, pullback = jax.vjp(partial(parts, batch=batch), flat)
            result += pullback((weight * (dr - mean_r), weight * (di - mean_i)))[0]
        return result

    def gradient(energies: Array) -> Array:
        if energies.shape != (n,):
            raise ValueError("one local energy per sample required")
        centered = energies - jnp.dot(w, energies)
        result = jnp.zeros_like(flat)
        for batch, weight, i in chunks:
            _, pullback = jax.vjp(partial(parts, batch=batch), flat)
            e = centered[i : i + len(batch)]
            result += 2 * pullback((weight * jnp.real(e), weight * jnp.imag(e)))[0]
        return result

    return QGTOperator(action, gradient, int(flat.size), n, w, sampling_kind)


class OperatorSRResult(NamedTuple):
    params: Any
    energy_mean: Array
    energy_gradient: Array
    solve: PCGResult
    accepted: Array


def matrixfree_sr_step(
    log_psi_fn: LogPsi,
    params: Any,
    samples: Array,
    local_energies: Array,
    *,
    weights: Array | None = None,
    damping: float = 1e-3,
    learning_rate: float = 1.0,
    chunk_size: int | None = None,
    rtol: float = 1e-8,
    max_iterations: int = 200,
    sampling_kind: str = "monte_carlo",
) -> OperatorSRResult:
    """Update a live pytree; a failed linear solve leaves parameters unchanged.

    Sampling remains the caller's existing sampler/local-energy pipeline. This
    also accepts finite enumerated systems, without forcing a position walker.
    """
    if (
        damping < 0
        or not math.isfinite(damping)
        or learning_rate < 0
        or not math.isfinite(learning_rate)
    ):
        raise ValueError("finite nonnegative damping and learning rate required")
    op = qgt_operator(
        log_psi_fn,
        params,
        samples,
        weights=weights,
        chunk_size=chunk_size,
        sampling_kind=sampling_kind,
    )
    grad = op.energy_gradient(local_energies)
    solved = pcg_solve(
        lambda v: op.matvec(v) + damping * v, grad, rtol=rtol, max_iterations=max_iterations
    )
    flat, unravel = ravel_pytree(params)
    proposed = flat - learning_rate * solved.solution
    updated = jnp.where(solved.converged, proposed, flat)
    return OperatorSRResult(
        unravel(updated),
        jnp.real(jnp.dot(op.weights, local_energies)),
        grad,
        solved,
        solved.converged,
    )


__all__ = ["OperatorSRResult", "QGTOperator", "matrixfree_sr_step", "qgt_operator"]
