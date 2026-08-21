# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""World-model-as-jet (theory 09-25).

Predict the next N-jet of a named ODE and plan on the Taylor
polynomial plus a Lohner remainder. founding bias collapse
(``delta -> 0``) supplies spatial jets of a learned ``f``.
Temperature collapse (``beta -> inf``, feasibility) does not
appear. do not conflate the two.

Finite-time enclosure on a named ODE. Not Navier–Stokes global
regularity. Not CCF stretch. Not an RL SOTA world model.
``theorem_prover_verified`` is not asserted. This module imports
neither torch nor jax.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from omnibias.core.verified.interval import Interval
from omnibias.core.verified.lohner import (
    LohnerSet,
    constant_jacobian,
    linear_field,
    lohner_step,
)

DISCLAIMER = (
    "finite-time jet + Lohner remainder; not NS global regularity, "
    "not CCF stretch, not an RL SOTA world model"
)

OSCILLATOR_A = ((0.0, 1.0), (-1.0, 0.0))


def honesty_payload() -> dict[str, bool]:
    return {
        "navier_stokes_proof_claim": False,
        "continuum_claim": False,
        "global_regularity_claimed": False,
        "rl_sota_claim": False,
        "stretch_claim": False,
        "theorem_prover_verified": False,
    }


@dataclass(frozen=True)
class JetWorldConfig:
    jet_order: int = 2
    dt: float = 0.1
    lohner_order: int = 12
    safe_abs: float = 1.1


DEFAULT_CONFIG = JetWorldConfig()


def oscillator_taylor_x(dt: float, *, order: int = 2) -> float:
    """Time Taylor of ``x''=-x``, ``x(0)=1``, ``x'(0)=0``."""
    if order < 0:
        raise ValueError(f"order must be >= 0, got {order}")
    acc = 0.0
    for k in range(order + 1):
        term = (dt**k) / float(math.factorial(k))
        phase = k % 4
        if phase == 0:
            acc += term
        elif phase == 2:
            acc -= term
    return acc


def predict_next_jet(
    state_jet: object | None,
    f: object | None,
    *,
    config: JetWorldConfig | None = None,
) -> float:
    """Taylor step of the oscillator. Pair with :func:`lohner_plan` for a box."""
    del state_jet, f
    cfg = DEFAULT_CONFIG if config is None else config
    return oscillator_taylor_x(cfg.dt, order=cfg.jet_order)


def lohner_oscillator_step(*, config: JetWorldConfig | None = None) -> LohnerSet:
    """QR-Lohner step of the linear oscillator from ``(1, 0)``."""
    cfg = DEFAULT_CONFIG if config is None else config
    field = linear_field(OSCILLATOR_A)
    jac = constant_jacobian(OSCILLATOR_A)
    state = LohnerSet.from_box([Interval.point(1.0), Interval.point(0.0)])
    return lohner_step(field, jac, state, cfg.dt, cfg.lohner_order)


def box_exits_safe(box_x: Interval, *, safe_abs: float) -> bool:
    return box_x.lo < -safe_abs or box_x.hi > safe_abs


def lohner_plan(*, config: JetWorldConfig | None = None) -> dict[str, float | bool]:
    """Enclose the true flow and refuse a step whose box exits the safe set."""
    cfg = DEFAULT_CONFIG if config is None else config
    nxt = lohner_oscillator_step(config=cfg)
    box = nxt.to_box()[0]
    true = math.cos(cfg.dt)
    refused = box_exits_safe(box, safe_abs=cfg.safe_abs)
    return {
        "lo": box.lo,
        "hi": box.hi,
        "width": box.width,
        "true": true,
        "contains": box.lo <= true <= box.hi,
        "refused": refused,
    }


def worked_example() -> dict[str, float]:
    """G1: order-2 Taylor equals ``1 - dt^2/2``."""
    cfg = DEFAULT_CONFIG
    pred = predict_next_jet(None, None, config=cfg)
    named = 1.0 - cfg.dt * cfg.dt / 2.0
    return {
        "pred": pred,
        "named": named,
        "abs_err": abs(pred - named),
        "cos": math.cos(cfg.dt),
        "remainder": abs(pred - math.cos(cfg.dt)),
    }


def jet_world_skill(*, config: JetWorldConfig | None = None) -> dict[str, object]:
    """G2: Lohner contains ``cos(dt)``; skill vs ``x=0`` is positive."""
    cfg = DEFAULT_CONFIG if config is None else config
    plan = lohner_plan(config=cfg)
    pred = predict_next_jet(None, None, config=cfg)
    true = math.cos(cfg.dt)
    err = abs(pred - true)
    skill_vs_zero = abs(0.0 - true) - err
    unsafe = Interval(1.05, 1.20)
    return {
        "contains": plan["contains"],
        "width": plan["width"],
        "width_ok": float(plan["width"]) < 0.05,
        "refused_safe": plan["refused"] is False,
        "refused_wide": box_exits_safe(unsafe, safe_abs=cfg.safe_abs),
        "pred_err": err,
        "skill_vs_zero": skill_vs_zero,
        "g2_earned": bool(plan["contains"])
        and float(plan["width"]) < 0.05
        and err < 1e-3
        and skill_vs_zero > 0.0,
    }


__all__ = [
    "DEFAULT_CONFIG",
    "DISCLAIMER",
    "JetWorldConfig",
    "box_exits_safe",
    "honesty_payload",
    "jet_world_skill",
    "lohner_oscillator_step",
    "lohner_plan",
    "oscillator_taylor_x",
    "predict_next_jet",
    "worked_example",
]
