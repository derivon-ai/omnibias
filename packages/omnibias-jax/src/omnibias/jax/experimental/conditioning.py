# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""How well determined is a damped Gauss-Newton step?

The dual step ``J^T (K + d I)^{-1} r`` with ``K = J J^T`` inherits the accuracy of ``K`` multiplied by the condition number
of ``K + d I``, which is at most ``(lambda_max + d) / d``.  Two correct programs whose ``K`` agree to 1e-14 can therefore
return steps that differ by 1e-3 when ``d`` is small against ``lambda_max``: the difference is the problem's, not a defect
of either program.  This module makes that visible and, on request, bounds it.

* :func:`condition_report` estimates ``lambda_max`` by power iteration (a lower bound, typically within a few percent for
  the dominant-mode spectra of Gram matrices) and also reports the rigorous Gershgorin bound, then gives the condition
  bound of ``K + d I`` and the step error it predicts, ``kappa * ulps * eps`` where ``ulps`` is the assumed accuracy of
  ``K`` in units of the dtype's machine epsilon (64, from 1.2e-14 measured between independent float64 programs).
* :func:`damped_solve` solves ``(K + d I) alpha = r``; with a ``"floor"`` policy it first raises ``d`` to the smallest value
  whose predicted step error is within the tolerance (never below twice the accuracy of ``K``: a float32 problem cannot
  be asked for a 1e-6 step).  The floor changes the step, deliberately: it is the damping at
  which the step is reproducible to that tolerance.  ``"report"`` never changes a result.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any

from omnibias.core.conditioning import (
    MODES,
    ConditioningPolicy,
    damping_floor_value,
    effective_tolerance_value,
    resolve_conditioning,
    step_status,
)

import jax
import jax.numpy as jnp

Array = jax.Array

__all__ = [
    "ConditionReport",
    "ConditioningPolicy",
    "MODES",
    "condition_report",
    "damped_solve",
    "damping_floor",
    "effective_tolerance",
    "estimate_lambda_max",
    "floored_damping",
    "resolve_conditioning",
]


def _finfo(dtype: Any) -> Any:
    return jnp.finfo(dtype)  # type: ignore[no-untyped-call,unused-ignore]


@partial(
    jax.tree_util.register_dataclass,
    data_fields=[
        "lambda_max",
        "lambda_max_upper",
        "damping",
        "effective_damping",
        "condition_estimate",
        "expected_step_error",
        "floor",
        "floor_applied",
    ],
    meta_fields=["step_tolerance", "requested_tolerance", "mode"],
)
@dataclass(frozen=True)
class ConditionReport:
    """Diagnostics of one damped solve (arrays; ``to_config`` fetches them as floats)."""

    lambda_max: Array  # power-iteration estimate (a lower bound)
    lambda_max_upper: Array  # rigorous: min(trace, Gershgorin row sum)
    damping: Array
    effective_damping: Array
    condition_estimate: Array  # (lambda_max + effective_damping) / effective_damping
    expected_step_error: Array  # condition_estimate * ulps * eps
    floor: Array  # the damping at which expected_step_error equals step_tolerance
    floor_applied: Array
    step_tolerance: float  # the tolerance in force (see effective_tolerance)
    requested_tolerance: float
    mode: str

    def to_config(self) -> dict[str, Any]:
        def f(a: Array) -> float:
            return float(jnp.max(a))

        expected = f(self.expected_step_error)
        status = step_status(expected, self.step_tolerance)
        return {
            "mode": self.mode,
            "status": status,
            "lambda_max": f(self.lambda_max),
            "lambda_max_upper": f(self.lambda_max_upper),
            "damping": f(self.damping),
            "effective_damping": f(self.effective_damping),
            "condition_estimate": f(self.condition_estimate),
            "expected_step_relative_error": expected,
            "damping_floor": f(self.floor),
            "floor_applied": bool(jnp.any(self.floor_applied)),
            "step_tolerance": self.step_tolerance,
            "requested_tolerance": self.requested_tolerance,
        }


def estimate_lambda_max(K: Array, iters: int = 16) -> Array:
    """Power-iteration estimate of the largest eigenvalue of the symmetric positive semi-definite ``K`` (a lower bound)."""
    n = K.shape[0]
    v0 = jnp.cos(1.2345 * jnp.arange(n, dtype=K.dtype) + 0.5)
    v0 = v0 / jnp.linalg.norm(v0)

    def body(_: int, v: Array) -> Array:
        w = K @ v
        return w / jnp.maximum(jnp.linalg.norm(w), _finfo(K.dtype).tiny)

    v = jax.lax.fori_loop(0, iters, body, v0)
    return jnp.maximum(v @ (K @ v), 0.0)


def effective_tolerance(dtype: Any, policy: ConditioningPolicy) -> float:
    """The step tolerance the data can support: the requested one, but never below twice the accuracy of ``K`` itself."""
    return effective_tolerance_value(float(_finfo(dtype).eps), policy)


def damping_floor(lam_max: Array, dtype: Any, policy: ConditioningPolicy) -> Array:
    """The smallest damping with predicted step error ``<= tolerance``: ``lambda_max * c / (tolerance - c)``.

    ``c = ulps * eps`` is the assumed relative accuracy of ``K``, and the tolerance is :func:`effective_tolerance`
    (a request below ``2 c`` is raised to it: a step cannot be more reproducible than the matrix it comes from).
    """
    return lam_max * (damping_floor_value(1.0, float(_finfo(dtype).eps), policy))


def condition_report(
    K: Array, damping: Array | float, policy: ConditioningPolicy
) -> ConditionReport:
    """Diagnose ``K + damping I`` (``damping`` scalar or a vector of values); applies the floor if ``policy`` says so."""
    d = jnp.asarray(damping, K.dtype)
    lam = estimate_lambda_max(K, policy.power_iters)
    upper = jnp.minimum(jnp.trace(K), jnp.max(jnp.sum(jnp.abs(K), axis=1)))
    floor = damping_floor(lam, K.dtype, policy)
    d_eff = jnp.maximum(d, floor) if policy.mode == "floor" else d
    eps = _finfo(K.dtype).eps
    kappa = (lam + d_eff) / jnp.maximum(d_eff, _finfo(K.dtype).tiny)
    return ConditionReport(
        lambda_max=lam,
        lambda_max_upper=upper,
        damping=d,
        effective_damping=d_eff,
        condition_estimate=kappa,
        expected_step_error=kappa * policy.kernel_error_ulps * eps,
        floor=floor,
        floor_applied=d_eff > d,
        step_tolerance=effective_tolerance(K.dtype, policy),
        requested_tolerance=policy.step_tolerance,
        mode=policy.mode,
    )


def floored_damping(K: Array, damping: Array | float, policy: ConditioningPolicy | None) -> Array:
    """``damping`` (scalar or vector), raised to the policy's floor when the policy is ``"floor"``; else unchanged."""
    d = jnp.asarray(damping, K.dtype)
    if policy is not None and policy.mode == "floor":
        lam = estimate_lambda_max(K, policy.power_iters)
        d = jnp.maximum(d, damping_floor(lam, K.dtype, policy))
    return d


def damped_solve(
    K: Array, r: Array, damping: Array | float, policy: ConditioningPolicy | None = None
) -> tuple[Array, Array]:
    """Solve ``(K + d I) alpha = r`` for scalar ``d``; returns ``(alpha, effective_damping)``.

    Cholesky as the engines always did; with a ``"floor"`` policy ``d`` is first raised to :func:`damping_floor`.
    Without a policy, or with ``"off"``/``"report"``, the result is bit-identical to the plain factorisation.
    """
    d = floored_damping(K, damping, policy)
    factor = jax.scipy.linalg.cho_factor(K + d * jnp.eye(K.shape[0], dtype=K.dtype))
    return jax.scipy.linalg.cho_solve(factor, r), d
