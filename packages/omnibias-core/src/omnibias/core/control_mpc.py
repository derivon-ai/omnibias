# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Receding-horizon jet MPC on a directional restriction (theory 08-12).

The unconstrained receding-horizon law is the first control of
:func:`~omnibias.core.control_lqr.scalar_finite_horizon_lqr` (theory
08-11). This module applies that ``u_0`` subject to an input box and an
optional Lagrange trust radius of the Taylor model, then **recedes**
(only ``u_0`` is applied). The jet is founding bias collapse (``delta -> 0``).
Temperature collapse (``beta -> inf``, feasibility) does not appear.
do not conflate the two.

This is receding LQR-with-box on a 1-D jet, not a plant / chemical-plant
MPC, not a general nonlinear programme, and not a global minimum of a
deep nest. Faà di Bruno remains the chain rule that *forms* the jet.
Activation Riccati is not a prediction model. Not CCF stretch.
``theorem_prover_verified`` is not asserted.
"""

from __future__ import annotations

import math
import random
import statistics
from collections.abc import Sequence
from dataclasses import dataclass

from omnibias.core.control_lqr import JetLQRConfig, lqr_step_from_derivatives
from omnibias.core.line_search import (
    certified_truncation_radius,
    lagrange_remainder_bound,
)

DISCLAIMER = (
    "receding jet-MPC: first control of scalar LQR plus box / remainder; "
    "not plant MPC, not a general QP, not a global min, not CCF stretch"
)


def honesty_payload() -> dict[str, bool]:
    return {
        "global_min_claimed": False,
        "plant_mpc_claimed": False,
        "general_qp_claimed": False,
        "algebraic_riccati_claimed": False,
        "activation_riccati_as_model": False,
        "stretch_claim": False,
        "continuum_claimed": False,
        "theorem_prover_verified": False,
        "skips_chain_rule": False,
    }


@dataclass(frozen=True)
class JetMPCConfig:
    """Receding-horizon box on the 08-11 first control.

    Parameters
    ----------
    q, r, qf, horizon
        Passed through to the 08-11 Riccati. ``horizon`` is the
        *planned* horizon; only the first control is applied.
    u_max
        Absolute input box on the applied control.
    next_derivative_bound
        Optional claimed ``|phi^(N+1)|`` on the applied segment.
    trust_atol
        If set with ``next_derivative_bound``, clip ``|u|`` to the
        certified Lagrange radius with this remainder tolerance.
    refuse_uncertified
        If true, refuse when the remainder upper bound of the applied
        step exceeds ``remainder_tol``.
    remainder_tol
        Positive finite remainder threshold.
    """

    q: float = 0.0
    r: float = 0.0
    qf: float = 1.0
    horizon: int = 1
    u_max: float = 2.0
    next_derivative_bound: float | None = None
    trust_atol: float | None = None
    refuse_uncertified: bool = False
    remainder_tol: float = 1.0


DEFAULT_CONFIG = JetMPCConfig()


@dataclass(frozen=True)
class JetMPCReport:
    step: float
    unconstrained_step: float
    gain: float
    error: float
    curvature: float
    horizon: int
    u_max: float
    trust_radius: float | None
    remainder_hi: float | None
    receding: bool
    refused: bool
    reason: str


def _require_finite(name: str, value: float) -> float:
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return out


def _validate_config(config: JetMPCConfig) -> JetMPCConfig:
    q = _require_finite("q", config.q)
    r = _require_finite("r", config.r)
    qf = _require_finite("qf", config.qf)
    u_max = _require_finite("u_max", config.u_max)
    rem_tol = _require_finite("remainder_tol", config.remainder_tol)
    if q < 0.0 or r < 0.0 or qf < 0.0:
        raise ValueError(f"LQR costs must be >= 0, got q={q}, r={r}, qf={qf}")
    if config.horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {config.horizon}")
    if u_max <= 0.0:
        raise ValueError(f"u_max must be > 0, got {u_max}")
    if rem_tol <= 0.0:
        raise ValueError(f"remainder_tol must be > 0, got {rem_tol}")
    bound = config.next_derivative_bound
    if bound is not None:
        bound_f = _require_finite("next_derivative_bound", bound)
        if bound_f < 0.0:
            raise ValueError(f"next_derivative_bound must be >= 0, got {bound_f}")
        bound = bound_f
    trust = config.trust_atol
    if trust is not None:
        trust_f = _require_finite("trust_atol", trust)
        if trust_f <= 0.0:
            raise ValueError(f"trust_atol must be > 0, got {trust_f}")
        if bound is None:
            raise ValueError("trust_atol requires next_derivative_bound")
        trust = trust_f
    if config.refuse_uncertified and bound is None:
        raise ValueError("refuse_uncertified=True requires next_derivative_bound")
    return JetMPCConfig(
        q=q,
        r=r,
        qf=qf,
        horizon=int(config.horizon),
        u_max=u_max,
        next_derivative_bound=bound,
        trust_atol=trust,
        refuse_uncertified=config.refuse_uncertified,
        remainder_tol=rem_tol,
    )


def mpc_step_from_derivatives(
    derivatives: Sequence[float],
    *,
    config: JetMPCConfig | None = None,
) -> JetMPCReport:
    """Receding first control of 08-11 LQR, boxed and optionally trusted."""
    cfg = _validate_config(DEFAULT_CONFIG if config is None else config)
    if not derivatives:
        raise ValueError("derivatives must be non-empty")
    used = [_require_finite(f"derivatives[{k}]", v) for k, v in enumerate(derivatives)]
    lqr = lqr_step_from_derivatives(
        used,
        config=JetLQRConfig(
            q=cfg.q,
            r=cfg.r,
            qf=cfg.qf,
            horizon=cfg.horizon,
            s_max=cfg.u_max,
        ),
    )
    unconstrained = lqr.control
    limit = cfg.u_max
    trust_radius: float | None = None
    order = len(used) - 1
    if cfg.trust_atol is not None:
        assert cfg.next_derivative_bound is not None
        trust_radius = certified_truncation_radius(
            cfg.next_derivative_bound, order, atol=cfg.trust_atol, max_step=cfg.u_max
        )
        limit = min(limit, trust_radius)
    step = max(-limit, min(limit, unconstrained))
    remainder_hi: float | None = None
    if cfg.refuse_uncertified:
        assert cfg.next_derivative_bound is not None
        rem = lagrange_remainder_bound(cfg.next_derivative_bound, step, order)
        remainder_hi = float(rem.hi)
        if remainder_hi > cfg.remainder_tol:
            return JetMPCReport(
                step=0.0,
                unconstrained_step=unconstrained,
                gain=lqr.gain,
                error=lqr.error,
                curvature=lqr.curvature,
                horizon=cfg.horizon,
                u_max=cfg.u_max,
                trust_radius=trust_radius,
                remainder_hi=remainder_hi,
                receding=True,
                refused=True,
                reason="uncertified_model",
            )
    return JetMPCReport(
        step=step,
        unconstrained_step=unconstrained,
        gain=lqr.gain,
        error=lqr.error,
        curvature=lqr.curvature,
        horizon=cfg.horizon,
        u_max=cfg.u_max,
        trust_radius=trust_radius,
        remainder_hi=remainder_hi,
        receding=True,
        refused=False,
        reason="ok" if step == unconstrained else "boxed",
    )


def worked_example() -> dict[str, float | bool]:
    """G1: Newton recovery, input box, receding flag."""
    free = mpc_step_from_derivatives(
        (1.0, -2.0, 2.0),
        config=JetMPCConfig(q=0.0, r=0.0, qf=1.0, horizon=1, u_max=4.0),
    )
    boxed = mpc_step_from_derivatives(
        (1.0, -2.0, 2.0),
        config=JetMPCConfig(q=0.0, r=0.0, qf=1.0, horizon=1, u_max=0.25),
    )
    return {
        "step": free.step,
        "abs_step_err": abs(free.step - 1.0),
        "boxed_step": boxed.step,
        "boxed": boxed.reason == "boxed",
        "receding": free.receding and boxed.receding,
        "newton_recovered": abs(free.step - 1.0) < 1e-15,
    }


def mpc_skill(*, n: int = 20, seed: int = 0) -> dict[str, object]:
    """G2: random bowls plus a box clip and a remainder refuse."""
    rng = random.Random(seed)
    errs: list[float] = []
    finite = True
    cfg = JetMPCConfig(q=0.0, r=0.0, qf=1.0, horizon=1, u_max=4.0)
    for _ in range(n):
        center = rng.uniform(0.25, 0.75)
        report = mpc_step_from_derivatives(
            (center * center, -2.0 * center, 2.0), config=cfg
        )
        if not math.isfinite(report.step):
            finite = False
        errs.append(abs(report.step - center))
    boxed = mpc_step_from_derivatives(
        (1.0, -2.0, 2.0),
        config=JetMPCConfig(q=0.0, r=0.0, qf=1.0, horizon=1, u_max=0.25),
    )
    refused = mpc_step_from_derivatives(
        (1.0, 1.0, 1.0),
        config=JetMPCConfig(
            q=0.0,
            r=0.0,
            qf=1.0,
            horizon=1,
            u_max=1.0,
            next_derivative_bound=math.e,
            refuse_uncertified=True,
            remainder_tol=0.1,
        ),
    )
    median = statistics.median(errs) if errs else math.inf
    return {
        "median_err": median,
        "all_finite": finite,
        "boxed": boxed.reason == "boxed" and boxed.step == 0.25,
        "refused": refused.refused and refused.reason == "uncertified_model",
        "g2_earned": bool(
            finite
            and median < 1e-12
            and boxed.reason == "boxed"
            and refused.refused
        ),
    }


__all__ = [
    "DEFAULT_CONFIG",
    "DISCLAIMER",
    "JetMPCConfig",
    "JetMPCReport",
    "honesty_payload",
    "mpc_skill",
    "mpc_step_from_derivatives",
    "worked_example",
]
