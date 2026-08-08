# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""NTK-rebalance helpers (jax twin)."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any

import jax
import jax.numpy as jnp
from jax import Array


def ntk_balanced_loss(
    losses: dict[str, Array],
    *,
    ntk_traces: dict[str, Array] | None = None,
    epsilon: float = 1e-12,
) -> tuple[Array, dict[str, float]]:
    """Combine multiple loss terms with NTK-balanced weights."""
    if not losses:
        raise ValueError("ntk_balanced_loss: empty losses dict")
    if ntk_traces is None:
        weights = {k: 1.0 for k in losses}
    else:
        if set(ntk_traces) != set(losses):
            raise ValueError(
                f"ntk_traces keys {sorted(ntk_traces)!r} do not match "
                f"losses keys {sorted(losses)!r}"
            )
        log_t: dict[str, float] = {}
        for k, t in ntk_traces.items():
            t_val = float(jnp.maximum(t, epsilon))
            log_t[k] = math.log(t_val)
        mean_log = sum(log_t.values()) / len(log_t)
        weights = {k: math.exp(mean_log - log_t[k]) for k in losses}

    total: Array | None = None
    for k, L in losses.items():
        term = float(weights[k]) * L
        total = term if total is None else total + term
    assert total is not None
    return total, {k: float(weights[k]) for k in losses}


def estimate_ntk_trace(
    loss_fn: Callable[[Any], Array],
    params: Any,
) -> Array:
    """Cheap NTK trace estimator: ``sum_p (d loss / d p)^2``."""
    grads = jax.grad(loss_fn)(params)
    leaves = jax.tree_util.tree_leaves(grads)
    out = jnp.zeros(
        (),
        dtype=jnp.result_type(*[jnp.asarray(g) for g in leaves])
        if leaves
        else jnp.float32,
    )
    for g in leaves:
        out = out + jnp.sum(g * g)
    return out


def _flatten_grads(grads: Any) -> Array:
    leaves = jax.tree_util.tree_leaves(grads)
    if not leaves:
        raise ValueError("params must contain at least one trainable leaf")
    return jnp.concatenate([jnp.ravel(g) for g in leaves])


def empirical_jacobian(
    residual_fn: Callable[[Any], Array],
    params: Any,
) -> Array:
    """Empirical Jacobian ``J = dr/dtheta`` with shape ``(n_out, n_params)``.

    ``residual_fn`` must accept the parameter pytree and return the residual
  vector.
    """
    residual0 = residual_fn(params).reshape(-1)
    n_out = int(residual0.shape[0])

    def row(m: int) -> Array:
        def fm(p: Any) -> Array:
            return residual_fn(p).reshape(-1)[m]

        return _flatten_grads(jax.grad(fm)(params))

    return jnp.stack([row(m) for m in range(n_out)], axis=0)


def ntk_eigenspectrum(
    residual_fn: Callable[[Any], Array],
    params: Any,
    *,
    n_eigen: int = 16,
) -> Array:
    """Leading empirical-NTK eigenvalues (squared singular values of ``J``)."""
    j = empirical_jacobian(residual_fn, params)
    s = jnp.linalg.svd(j, compute_uv=False)
    evals = s * s
    k = min(int(n_eigen), int(evals.shape[0]))
    idx = jnp.argsort(evals)[::-1]
    return evals[idx][:k]


def ntk_tail_head_index(eigenvalues: Array, *, n_head: int = 4) -> float:
    """Legacy tail/head eigenvalue ratio."""
    ev = jnp.asarray(eigenvalues).reshape(-1)
    if ev.size < 2:
        return 1.0
    n_head = max(1, min(int(n_head), ev.size // 2))
    head = jnp.mean(ev[:n_head])
    tail = jnp.mean(ev[n_head:])
    if float(head) <= 0.0:
        return 0.0
    return float(jnp.clip(tail / head, 0.0, 1.0))


def fourier_mode_learning_rates(
    residual_fn: Callable[[Any], Array],
    params: Any,
    *,
    coords: Array,
    modes: Sequence[int],
    L: float = 1.0,
    window_axis: int = 0,
) -> Array:
    """Per-Fourier-mode NTK learning-rate proxy on a uniform 1-D grid."""
    if not modes:
        raise ValueError("modes must be non-empty")
    residual = residual_fn(params).reshape(-1)
    x = coords[:, int(window_axis)]
    n = int(x.shape[0])
    if n != int(residual.shape[0]):
        raise ValueError(
            f"coords rows {n} must match residual length {int(residual.shape[0])}"
        )
    rates: list[Array] = []
    two_pi = 2.0 * math.pi
    for k in modes:
        phi = jnp.sin(two_pi * float(k) * x / float(L))
        phi = phi / (jnp.linalg.norm(phi) + 1e-12)

        def scalar(p: Any, phi: Array = phi) -> Array:
            return jnp.dot(residual_fn(p).reshape(-1), phi)

        flat = _flatten_grads(jax.grad(scalar)(params))
        rates.append(jnp.sum(flat * flat))
    return jnp.stack(rates)


def kernel_task_alignment(
    mode_rates: Array,
    task_coeffs: Sequence[float],
) -> float:
    """Cosine alignment between task energy and per-mode kernel capacity."""
    rates = jnp.asarray(mode_rates).reshape(-1)
    task = jnp.asarray(list(task_coeffs), dtype=rates.dtype)
    if rates.shape[0] != task.shape[0]:
        raise ValueError(
            f"mode_rates length {rates.shape[0]} != task_coeffs {task.shape[0]}"
        )
    num = float(jnp.sum(rates * task))
    den = float(jnp.linalg.norm(rates) * jnp.linalg.norm(task))
    if den <= 0.0:
        return 0.0
    return num / den


def spectral_bias_index(
    mode_rates: Array,
    *,
    n_low: int = 2,
    n_high: int = 2,
) -> float:
    """Low- versus high-frequency NTK response (smaller => stronger bias)."""
    rates = jnp.asarray(mode_rates).reshape(-1)
    if rates.size < 2:
        return 1.0
    n_low = max(1, min(int(n_low), rates.size - 1))
    n_high = max(1, min(int(n_high), rates.size - n_low))
    low = jnp.mean(rates[:n_low])
    high = jnp.mean(rates[-n_high:])
    if float(high) <= 0.0:
        return 0.0
    return float(jnp.clip(low / high, 0.0, 1.0))


__all__ = [
    "empirical_jacobian",
    "estimate_ntk_trace",
    "fourier_mode_learning_rates",
    "kernel_task_alignment",
    "ntk_balanced_loss",
    "ntk_eigenspectrum",
    "ntk_tail_head_index",
    "spectral_bias_index",
]
