# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""How well determined is a damped Gauss-Newton step? (PyTorch)

The policy and the floor formulas live in :mod:`omnibias.core.conditioning` and are shared with the JAX backend; this
module estimates ``lambda_max`` of the Gram matrix on tensors and solves the damped system, optionally with the damping
raised to the reproducibility floor.
"""

from __future__ import annotations

from typing import Any

from omnibias.core.conditioning import (
    MODES,
    ConditioningPolicy,
    damping_floor_value,
    effective_tolerance_value,
    resolve_conditioning,
    step_status,
)

import torch
from torch import Tensor

__all__ = [
    "ConditioningPolicy",
    "MODES",
    "condition_report",
    "damped_solve",
    "estimate_lambda_max",
    "floored_damping",
    "resolve_conditioning",
]


def estimate_lambda_max(K: Tensor, iters: int = 16) -> Tensor:
    """Power-iteration estimate of the largest eigenvalue of the symmetric positive semi-definite ``K`` (a lower bound)."""
    n = K.shape[0]
    v = torch.cos(1.2345 * torch.arange(n, dtype=K.dtype, device=K.device) + 0.5)
    v = v / torch.linalg.vector_norm(v)
    tiny = torch.finfo(K.dtype).tiny
    for _ in range(iters):
        w = K @ v
        v = w / torch.clamp(torch.linalg.vector_norm(w), min=tiny)
    return torch.clamp(v @ (K @ v), min=0.0)


def floored_damping(K: Tensor, damping: Any, policy: ConditioningPolicy | None) -> Tensor:
    """``damping`` raised to the policy's floor when the policy is ``"floor"``; else unchanged."""
    d = torch.as_tensor(damping, dtype=K.dtype, device=K.device)
    if policy is not None and policy.mode == "floor":
        lam = estimate_lambda_max(K, policy.power_iters)
        eps = float(torch.finfo(K.dtype).eps)
        d = torch.maximum(d, lam * damping_floor_value(1.0, eps, policy))
    return d


def condition_report(K: Tensor, damping: Any, policy: ConditioningPolicy) -> dict[str, Any]:
    """Diagnose ``K + damping I``; the keys are those of the JAX ``ConditionReport.to_config()``."""
    d = torch.as_tensor(damping, dtype=K.dtype, device=K.device)
    lam = estimate_lambda_max(K, policy.power_iters)
    upper = torch.minimum(torch.trace(K), torch.max(torch.sum(torch.abs(K), dim=1)))
    eps = float(torch.finfo(K.dtype).eps)
    floor = lam * damping_floor_value(1.0, eps, policy)
    d_eff = torch.maximum(d, floor) if policy.mode == "floor" else d
    kappa = (lam + d_eff) / torch.clamp(d_eff, min=torch.finfo(K.dtype).tiny)
    expected = float(torch.max(kappa * policy.kernel_error_ulps * eps))
    tol = effective_tolerance_value(eps, policy)
    return {
        "mode": policy.mode,
        "status": step_status(expected, tol),
        "lambda_max": float(lam),
        "lambda_max_upper": float(upper),
        "damping": float(torch.max(d)),
        "effective_damping": float(torch.max(d_eff)),
        "condition_estimate": float(torch.max(kappa)),
        "expected_step_relative_error": expected,
        "damping_floor": float(floor),
        "floor_applied": bool(torch.any(d_eff > d)),
        "step_tolerance": tol,
        "requested_tolerance": policy.step_tolerance,
    }


def damped_solve(
    K: Tensor, r: Tensor, damping: Any, policy: ConditioningPolicy | None = None
) -> tuple[Tensor, Tensor]:
    """Solve ``(K + d I) alpha = r``; returns ``(alpha, effective_damping)`` (Cholesky, as the JAX engines do)."""
    d = floored_damping(K, damping, policy)
    factor = torch.linalg.cholesky(K + d * torch.eye(K.shape[0], dtype=K.dtype, device=K.device))
    return torch.cholesky_solve(r[:, None], factor)[:, 0], d
