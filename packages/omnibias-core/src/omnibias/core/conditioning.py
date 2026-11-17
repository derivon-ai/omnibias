# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""The conditioning policy and floor formulas of damped Gauss-Newton steps, independent of any tensor framework.

The dual step ``J^T (K + d I)^{-1} r`` with ``K = J J^T`` inherits the accuracy of ``K`` multiplied by the condition number
of ``K + d I``, which is at most ``(lambda_max + d) / d``. With ``c = ulps * eps`` the assumed relative accuracy of ``K``
the predicted step error is ``kappa * c``, and the smallest damping whose prediction meets a tolerance ``tol`` is
``lambda_max * c / (tol - c)``. The JAX and PyTorch front ends estimate ``lambda_max`` on their own arrays and use the
same policy and formulas, so a plan and its report mean the same on either backend.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "ConditioningPolicy",
    "MODES",
    "damping_floor_value",
    "effective_tolerance_value",
    "resolve_conditioning",
    "step_status",
]

MODES = ("off", "report", "floor")


@dataclass(frozen=True)
class ConditioningPolicy:
    """``mode``: ``"off"``, ``"report"`` (diagnose, never change a result) or ``"floor"`` (raise the damping to the floor).

    ``step_tolerance`` is the relative step error the floor guarantees (under the model in the module docstring);
    ``kernel_error_ulps`` the assumed relative accuracy of ``K`` in machine epsilons; ``power_iters`` the iterations of
    the ``lambda_max`` estimate.
    """

    mode: str = "report"
    step_tolerance: float = 1e-6
    kernel_error_ulps: float = 64.0
    power_iters: int = 16

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"conditioning mode must be one of {MODES}, got {self.mode!r}")
        if not (self.step_tolerance > 0 and self.kernel_error_ulps > 0 and self.power_iters >= 1):
            raise ValueError(
                "step_tolerance and kernel_error_ulps must be positive and power_iters >= 1"
            )


def resolve_conditioning(policy: str | ConditioningPolicy | None) -> ConditioningPolicy:
    """``None`` means ``"off"``; a string selects the mode with default tolerances."""
    if policy is None:
        return ConditioningPolicy("off")
    if isinstance(policy, ConditioningPolicy):
        return policy
    return ConditioningPolicy(str(policy))


def effective_tolerance_value(eps: float, policy: ConditioningPolicy) -> float:
    """The step tolerance the data can support: the requested one, never below twice the accuracy of ``K`` itself."""
    c = policy.kernel_error_ulps * float(eps)
    return max(policy.step_tolerance, 2.0 * c)


def damping_floor_value(lam_max: float, eps: float, policy: ConditioningPolicy) -> float:
    """The smallest damping with predicted step error ``<= tolerance``: ``lambda_max * c / (tolerance - c)``."""
    c = policy.kernel_error_ulps * float(eps)
    tol = effective_tolerance_value(eps, policy)
    return float(lam_max) * (c / (tol - c))


def step_status(expected_error: float, tolerance: float) -> str:
    """``"ok"``, ``"conditioning_limited"`` (a bounded but larger error) or ``"unreliable"`` (no digits promised)."""
    if expected_error <= tolerance:
        return "ok"
    return "conditioning_limited" if expected_error < 0.1 else "unreliable"
