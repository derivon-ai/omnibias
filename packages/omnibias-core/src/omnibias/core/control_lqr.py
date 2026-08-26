# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Finite-horizon LQR on a directional jet (theory 08-11).

A scalar restriction ``phi(s) = L(theta + s d)`` is known through its
Taylor jet (founding bias collapse, ``delta -> 0``). The local linear
error dynamics are

    e_{k+1} = e_k + phi''(0) u_k,    e_0 = phi'(0),

an integrator scaled by curvature. Finite-horizon discrete LQR with
costs ``Q, R, Qf`` yields the first control ``u_0 = -K_0 e_0``, applied
as the signed step along ``d``.

This is a **scalar discrete Riccati recursion**, not the infinite-horizon
algebraic Riccati / DARE, and not the activation Riccati
(``sigma' = s(1-s)``). Temperature collapse (``beta -> inf``,
feasibility) does not appear. do not conflate the two.

``R = 0``, horizon ``1``, ``Qf = 1`` recovers the Newton step
``-phi'/phi''`` when ``phi'' != 0``. Local 1-D restriction. Not a plant
LQR, not MPC, not a global minimum of a deep nest. Faà di Bruno remains
the chain rule that *forms* the jet. Not CCF stretch.
``theorem_prover_verified`` is not asserted.
"""

from __future__ import annotations

import math
import random
import statistics
from collections.abc import Sequence
from dataclasses import dataclass

DISCLAIMER = (
    "local 1-D finite-horizon LQR on a directional jet; scalar discrete "
    "Riccati, not DARE, not activation Riccati, not a plant LQR, not MPC, "
    "not a global min, not CCF stretch"
)

_SINGULAR = 1e-18


def honesty_payload() -> dict[str, bool]:
    return {
        "global_min_claimed": False,
        "plant_lqr_claimed": False,
        "algebraic_riccati_claimed": False,
        "activation_riccati_as_gain": False,
        "mpc_claimed": False,
        "stretch_claim": False,
        "continuum_claimed": False,
        "theorem_prover_verified": False,
        "skips_chain_rule": False,
    }


@dataclass(frozen=True)
class JetLQRConfig:
    """Scalar finite-horizon LQR costs on the directional error.

    Parameters
    ----------
    q, r, qf
        Running state cost, control effort, and terminal cost. ``r`` and
        ``q`` / ``qf`` must be finite with ``r >= 0``, ``q >= 0``,
        ``qf >= 0``.
    horizon
        Number of discrete stages ``N >= 1``. Only the first control is
        applied (a later MPC spec recedes this).
    s_max
        Absolute step clip along ``d``.
    """

    q: float = 0.0
    r: float = 0.0
    qf: float = 1.0
    horizon: int = 1
    s_max: float = 2.0


DEFAULT_CONFIG = JetLQRConfig()


@dataclass(frozen=True)
class JetLQRReport:
    step: float
    gain: float
    error: float
    curvature: float
    control: float
    terminal_cost: float
    horizon: int
    refused: bool
    reason: str


def _require_finite(name: str, value: float) -> float:
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return out


def _validate_config(config: JetLQRConfig) -> JetLQRConfig:
    q = _require_finite("q", config.q)
    r = _require_finite("r", config.r)
    qf = _require_finite("qf", config.qf)
    s_max = _require_finite("s_max", config.s_max)
    if q < 0.0 or r < 0.0 or qf < 0.0:
        raise ValueError(f"LQR costs must be >= 0, got q={q}, r={r}, qf={qf}")
    if config.horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {config.horizon}")
    if s_max <= 0.0:
        raise ValueError(f"s_max must be > 0, got {s_max}")
    return JetLQRConfig(q=q, r=r, qf=qf, horizon=int(config.horizon), s_max=s_max)


def scalar_finite_horizon_lqr(
    a: float,
    b: float,
    q: float,
    r: float,
    qf: float,
    horizon: int,
) -> tuple[float, float]:
    """Backward scalar Riccati. Returns ``(K_0, P_0)``.

    Dynamics ``x+ = a x + b u``. Costs ``q, r, qf``. Raises if
    ``R + B^T P B`` is singular.
    """
    a_f = _require_finite("a", a)
    b_f = _require_finite("b", b)
    q_f = _require_finite("q", q)
    r_f = _require_finite("r", r)
    qf_f = _require_finite("qf", qf)
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")
    if q_f < 0.0 or r_f < 0.0 or qf_f < 0.0:
        raise ValueError("LQR costs must be >= 0")
    p_next = qf_f
    gain = 0.0
    for _ in range(int(horizon)):
        denom = r_f + b_f * b_f * p_next
        if abs(denom) < _SINGULAR:
            raise ValueError(
                "LQR Riccati solve is singular (R + B^T P B ~ 0); "
                "need r>0 or nonzero curvature"
            )
        gain = (b_f * p_next * a_f) / denom
        p_next = q_f + a_f * a_f * p_next - gain * (b_f * p_next * a_f)
        if not math.isfinite(p_next) or not math.isfinite(gain):
            raise ValueError("LQR Riccati produced a non-finite cost-to-go")
    return gain, p_next


def lqr_step_from_derivatives(
    derivatives: Sequence[float],
    *,
    config: JetLQRConfig | None = None,
) -> JetLQRReport:
    """One LQR step from ``phi^(k)(0)``. Needs ``phi'`` and ``phi''``."""
    cfg = _validate_config(DEFAULT_CONFIG if config is None else config)
    if not derivatives:
        raise ValueError("derivatives must be non-empty")
    used = [_require_finite(f"derivatives[{k}]", v) for k, v in enumerate(derivatives)]
    if len(used) < 3:
        raise ValueError("jet-LQR needs phi, phi', and phi'' (three derivatives)")
    error = used[1]
    curvature = used[2]
    gain, p0 = scalar_finite_horizon_lqr(
        1.0, curvature, cfg.q, cfg.r, cfg.qf, cfg.horizon
    )
    control = -gain * error
    if not math.isfinite(control):
        raise ValueError(f"LQR control is not finite, got {control!r}")
    step = max(-cfg.s_max, min(cfg.s_max, control))
    return JetLQRReport(
        step=step,
        gain=gain,
        error=error,
        curvature=curvature,
        control=control,
        terminal_cost=p0,
        horizon=cfg.horizon,
        refused=False,
        reason="ok",
    )


def worked_example() -> dict[str, float | bool]:
    """G1: ``phi(s)=(s-1)^2`` Newton recovery and ``R>0`` shrinkage."""
    newton = lqr_step_from_derivatives(
        (1.0, -2.0, 2.0),
        config=JetLQRConfig(q=0.0, r=0.0, qf=1.0, horizon=1, s_max=4.0),
    )
    damped = lqr_step_from_derivatives(
        (1.0, -2.0, 2.0),
        config=JetLQRConfig(q=0.0, r=4.0, qf=1.0, horizon=1, s_max=4.0),
    )
    return {
        "step": newton.step,
        "abs_step_err": abs(newton.step - 1.0),
        "gain": newton.gain,
        "damped_step": damped.step,
        "damped_smaller": abs(damped.step) < abs(newton.step) - 1e-15,
        "newton_recovered": abs(newton.step - 1.0) < 1e-15,
    }


def lqr_skill(*, n: int = 20, seed: int = 0) -> dict[str, object]:
    """G2: random bowls recover Newton; singular ``R=0, phi''=0`` raises."""
    rng = random.Random(seed)
    errs: list[float] = []
    finite = True
    cfg = JetLQRConfig(q=0.0, r=0.0, qf=1.0, horizon=1, s_max=4.0)
    for _ in range(n):
        center = rng.uniform(0.25, 0.75)
        report = lqr_step_from_derivatives(
            (center * center, -2.0 * center, 2.0), config=cfg
        )
        if not math.isfinite(report.step):
            finite = False
        errs.append(abs(report.step - center))
    singular = False
    try:
        lqr_step_from_derivatives(
            (1.0, -1.0, 0.0),
            config=JetLQRConfig(q=0.0, r=0.0, qf=1.0, horizon=1),
        )
    except ValueError as exc:
        singular = "singular" in str(exc)
    median = statistics.median(errs) if errs else math.inf
    return {
        "median_err": median,
        "all_finite": finite,
        "singular_raised": singular,
        "g2_earned": finite and median < 1e-12 and singular,
    }


__all__ = [
    "DEFAULT_CONFIG",
    "DISCLAIMER",
    "JetLQRConfig",
    "JetLQRReport",
    "honesty_payload",
    "lqr_skill",
    "lqr_step_from_derivatives",
    "scalar_finite_horizon_lqr",
    "worked_example",
]
