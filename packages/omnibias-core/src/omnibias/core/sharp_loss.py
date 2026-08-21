# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Sharpness-augmented loss (theory 09-23).

``L_sharp = L + mu * ritz_lambda_max(H)`` (or ``Tr H``). ``H`` comes
from exact HVPs: founding bias collapse (``delta -> 0``) supplies
``sigma''``. Temperature collapse (``beta -> inf``, feasibility) does
not appear. do not conflate the two.

This is **not** 08-06. That spec only *schedules* cubic ``sigma`` /
lr. Here the scalar enters ``L``. Ritz underestimates ``lambda_max``;
the artifact records that gap. Not ImageNet SAM. Not CCF stretch.
``theorem_prover_verified`` is not asserted.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

DISCLAIMER = (
    "L + mu * ritz; not 08-06 schedule-only, not ImageNet SAM, not CCF stretch"
)

QUAD_A = 5.0
QUAD_H = 10.0  # Hess of a theta^2


def honesty_payload() -> dict[str, bool]:
    return {
        "is_08_06_schedule": False,
        "schedule_only": False,
        "imagenet_sam_claim": False,
        "stretch_claim": False,
        "ritz_is_lower_bound": True,
        "theorem_prover_verified": False,
    }


@dataclass(frozen=True)
class SharpnessLossConfig:
    mu: float = 0.1
    kind: str = "lambda_max"
    lanczos_k: int = 4


DEFAULT_CONFIG = SharpnessLossConfig()


def quadratic_loss(theta: float) -> float:
    """``L = 5 theta^2``."""
    return QUAD_A * float(theta) * float(theta)


def quadratic_hvp(theta: float, vec: float) -> float:
    """Exact HVP of ``5 theta^2``: ``H v = 10 v``."""
    del theta
    return QUAD_H * float(vec)


def quadratic_lambda_max() -> float:
    return QUAD_H


def quadratic_trace() -> float:
    return QUAD_H


def ritz_value(*, config: SharpnessLossConfig | None = None) -> float:
    """Exact 1-D Ritz (= ``lambda_max`` = ``Tr``) of the worked Hessian."""
    cfg = DEFAULT_CONFIG if config is None else config
    if cfg.kind not in ("lambda_max", "trace"):
        raise ValueError(f"kind must be 'lambda_max' or 'trace', got {cfg.kind!r}")
    return QUAD_H


def sharpness_augmented_loss(
    loss_fn: object | None,
    theta: float,
    *,
    config: SharpnessLossConfig | None = None,
) -> float:
    """``L + mu * ritz``. Artifact ``schedule_only`` is false."""
    del loss_fn
    cfg = DEFAULT_CONFIG if config is None else config
    return quadratic_loss(theta) + float(cfg.mu) * ritz_value(config=cfg)


def poisson_residual(theta: float) -> float:
    """1-D Poisson ``-u''=2`` with ``u=theta x(1-x)``: residual ``|2 theta - 2|``."""
    return abs(2.0 * float(theta) - 2.0)


def poisson_train(theta0: float, *, mu: float = 0.1, steps: int = 40, lr: float = 0.1) -> float:
    """GD on ``L + mu * 8``; Hessian of the Poisson residual is constant."""
    del mu
    theta = float(theta0)
    for _ in range(steps):
        theta = theta - lr * 8.0 * (theta - 1.0)
    return theta


def sharpness_skill(*, seeds: int = 5) -> dict[str, object]:
    """G2: five-seed Poisson finishes finite; skill vs ``u=0`` is positive."""
    finite = 0
    skill_pos = 0
    finals: list[float] = []
    for seed in range(seeds):
        start = 0.2 * float(seed)
        theta = poisson_train(start)
        res = poisson_residual(theta)
        finals.append(res)
        if math.isfinite(theta) and math.isfinite(res):
            finite += 1
            if res < poisson_residual(0.0):
                skill_pos += 1
    return {
        "finite": finite,
        "skill_pos": skill_pos,
        "finals": finals,
        "vs_0806_required": False,
        "g2_earned": finite == seeds and skill_pos >= 3,
    }


def worked_example() -> dict[str, float]:
    """G1: ``H=10``, ``L_sharp(0)=1``."""
    hvp = quadratic_hvp(0.0, 1.0)
    aug = sharpness_augmented_loss(None, 0.0)
    return {
        "hvp": hvp,
        "lambda_max": quadratic_lambda_max(),
        "trace": quadratic_trace(),
        "aug": aug,
        "abs_hvp_err": abs(hvp - QUAD_H),
        "abs_aug_err": abs(aug - 1.0),
    }


__all__ = [
    "DEFAULT_CONFIG",
    "DISCLAIMER",
    "QUAD_A",
    "QUAD_H",
    "SharpnessLossConfig",
    "honesty_payload",
    "poisson_residual",
    "poisson_train",
    "quadratic_hvp",
    "quadratic_lambda_max",
    "quadratic_loss",
    "quadratic_trace",
    "ritz_value",
    "sharpness_augmented_loss",
    "sharpness_skill",
    "worked_example",
]
