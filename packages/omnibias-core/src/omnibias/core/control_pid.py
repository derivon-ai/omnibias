# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Jet-PID optimizer on a directional restriction (theory 08-10).

A scalar restriction ``phi(s) = L(theta + s d)`` is known through its
Taylor jet (founding bias collapse, ``delta -> 0``). The tracking error
is the directional derivative ``e(s) = phi'(s)``. PID is evaluated on
that jet:

* **P** -- ``e(0) = phi'(0)``
* **I** -- exact fundamental-theorem integral of the *Taylor model*
  ``p`` of ``phi``: ``I = p(h) - p(0)`` (``model_window``), or the same
  increment accumulated along applied steps (``running``). This is not a
  discrete sum of past gradients.
* **D** -- ``e'(0) = phi''(0)``

Control is ``u = kp P + ki I + kd D``. The signed step along ``d`` is
``s* = clip(-u, -s_max, s_max)``. Temperature collapse (``beta -> inf``,
feasibility) does not appear. do not conflate the two.

This is a local 1-D restriction trainer, not a plant PID, not LQR, not
MPC, and not a global minimum of a deep nest. Faà di Bruno remains the
chain rule that *forms* the jet. The I term is exact on the model;
truncation remainder is real and may refuse a step. Activation Riccati
(``sigma' = s(1-s)``) is not a gain formula. Not CCF stretch.
``theorem_prover_verified`` is not asserted.
"""

from __future__ import annotations

import math
import random
import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from omnibias.core.line_search import (
    lagrange_remainder_bound,
    poly_eval,
    taylor_coeffs_from_derivatives,
)

DISCLAIMER = (
    "local 1-D jet-PID on a directional restriction; not a plant PID, "
    "not LQR, not MPC, not a global min, not CCF stretch"
)

IntegralMode = Literal["model_window", "running"]


def honesty_payload() -> dict[str, bool]:
    return {
        "global_min_claimed": False,
        "plant_pid_claimed": False,
        "lqr_claimed": False,
        "mpc_claimed": False,
        "algebraic_riccati_claimed": False,
        "stretch_claim": False,
        "continuum_claimed": False,
        "theorem_prover_verified": False,
        "skips_chain_rule": False,
    }


@dataclass(frozen=True)
class JetPIDConfig:
    """Gains and integral policy for one jet-PID step.

    Parameters
    ----------
    kp, ki, kd
        Finite PID gains. ``kd != 0`` requires ``phi''(0)`` (at least
        three derivatives).
    window
        Design window ``h`` for ``integral_mode="model_window"``.
        ``I = p(h) - p(0)``.
    s_max
        Absolute step clip along ``d``. Must be positive and finite.
    integral_mode
        ``model_window`` uses the local Taylor FTC over ``window``.
        ``running`` adds the FTC increment of the *applied* step to
        ``running_integral``.
    next_derivative_bound
        Optional claimed ``|phi^(N+1)|`` bound on the segment. Used only
        when ``refuse_uncertified_integral`` is true.
    refuse_uncertified_integral
        If true, refuse (step ``0``) when the Lagrange remainder of
        ``p`` on the integral segment exceeds ``remainder_tol``.
    remainder_tol
        Positive finite threshold on the remainder upper bound.
    """

    kp: float = 1.0
    ki: float = 0.0
    kd: float = 0.0
    window: float = 1.0
    s_max: float = 1.0
    integral_mode: IntegralMode = "model_window"
    next_derivative_bound: float | None = None
    refuse_uncertified_integral: bool = False
    remainder_tol: float = 1.0


DEFAULT_CONFIG = JetPIDConfig()


@dataclass(frozen=True)
class JetPIDReport:
    step: float
    proportional: float
    integral: float
    derivative: float
    control: float
    model_value: float
    integral_mode: str
    running_integral: float
    remainder_hi: float | None
    refused: bool
    reason: str


def poly_antiderivative(coeffs: Sequence[float]) -> list[float]:
    """Formal antiderivative of ``sum_k coeffs[k] s^k`` with constant 0."""
    if not coeffs:
        raise ValueError("coeffs must be non-empty")
    out = [0.0]
    for k, raw in enumerate(coeffs):
        val = float(raw)
        if not math.isfinite(val):
            raise ValueError(f"coeff[{k}] must be finite, got {raw!r}")
        out.append(val / float(k + 1))
    return out


def model_definite_integral(coeffs: Sequence[float], lo: float, hi: float) -> float:
    """Exact definite integral of the Taylor *model* ``p`` on ``[lo, hi]``."""
    anti = poly_antiderivative(coeffs)
    return poly_eval(anti, hi) - poly_eval(anti, lo)


def model_ftc_increment(coeffs: Sequence[float], lo: float, hi: float) -> float:
    r"""Exact FTC of the model error ``e = p'``: ``p(hi) - p(lo)``."""
    if not coeffs:
        raise ValueError("coeffs must be non-empty")
    lo_f = float(lo)
    hi_f = float(hi)
    if not math.isfinite(lo_f) or not math.isfinite(hi_f):
        raise ValueError(f"integral limits must be finite, got [{lo!r}, {hi!r}]")
    return poly_eval(coeffs, hi_f) - poly_eval(coeffs, lo_f)


def _require_finite(name: str, value: float) -> float:
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return out


def _validate_config(config: JetPIDConfig) -> JetPIDConfig:
    kp = _require_finite("kp", config.kp)
    ki = _require_finite("ki", config.ki)
    kd = _require_finite("kd", config.kd)
    window = _require_finite("window", config.window)
    s_max = _require_finite("s_max", config.s_max)
    if s_max <= 0.0:
        raise ValueError(f"s_max must be > 0, got {s_max}")
    if config.integral_mode not in ("model_window", "running"):
        raise ValueError(f"unknown integral_mode {config.integral_mode!r}")
    rem_tol = _require_finite("remainder_tol", config.remainder_tol)
    if rem_tol <= 0.0:
        raise ValueError(f"remainder_tol must be > 0, got {rem_tol}")
    bound = config.next_derivative_bound
    if bound is not None:
        bound_f = _require_finite("next_derivative_bound", bound)
        if bound_f < 0.0:
            raise ValueError(f"next_derivative_bound must be >= 0, got {bound_f}")
        bound = bound_f
    if config.refuse_uncertified_integral and bound is None:
        raise ValueError(
            "refuse_uncertified_integral=True requires next_derivative_bound"
        )
    return JetPIDConfig(
        kp=kp,
        ki=ki,
        kd=kd,
        window=window,
        s_max=s_max,
        integral_mode=config.integral_mode,
        next_derivative_bound=bound,
        refuse_uncertified_integral=config.refuse_uncertified_integral,
        remainder_tol=rem_tol,
    )


def pid_step_from_derivatives(
    derivatives: Sequence[float],
    *,
    config: JetPIDConfig | None = None,
    running_integral: float = 0.0,
) -> JetPIDReport:
    """One jet-PID step from ``phi^(k)(0)`` for ``k = 0 .. N``.

    Raises ``ValueError`` on empty / non-finite jets, missing ``phi'``,
    or ``kd != 0`` without ``phi''``.
    """
    cfg = _validate_config(DEFAULT_CONFIG if config is None else config)
    if not derivatives:
        raise ValueError("derivatives must be non-empty")
    used = [_require_finite(f"derivatives[{k}]", v) for k, v in enumerate(derivatives)]
    if len(used) < 2:
        raise ValueError("jet-PID needs phi and phi' (at least two derivatives)")
    if cfg.kd != 0.0 and len(used) < 3:
        raise ValueError("kd != 0 requires phi''(0) (at least three derivatives)")
    i_state = _require_finite("running_integral", running_integral)

    coeffs = taylor_coeffs_from_derivatives(used)
    proportional = used[1]
    derivative = used[2] if len(used) >= 3 else 0.0
    if cfg.integral_mode == "model_window":
        integral = model_ftc_increment(coeffs, 0.0, cfg.window)
        integral_segment = cfg.window
    else:
        integral = i_state
        integral_segment = 0.0

    remainder_hi: float | None = None
    if cfg.refuse_uncertified_integral:
        assert cfg.next_derivative_bound is not None
        order = len(used) - 1
        rem = lagrange_remainder_bound(
            cfg.next_derivative_bound, integral_segment, order
        )
        remainder_hi = float(rem.hi)
        if remainder_hi > cfg.remainder_tol:
            return JetPIDReport(
                step=0.0,
                proportional=proportional,
                integral=integral,
                derivative=derivative,
                control=0.0,
                model_value=used[0],
                integral_mode=cfg.integral_mode,
                running_integral=i_state,
                remainder_hi=remainder_hi,
                refused=True,
                reason="uncertified_integral",
            )

    control = cfg.kp * proportional + cfg.ki * integral + cfg.kd * derivative
    if not math.isfinite(control):
        raise ValueError(f"PID control is not finite, got {control!r}")
    raw = -control
    step = max(-cfg.s_max, min(cfg.s_max, raw))
    running = i_state + model_ftc_increment(coeffs, 0.0, step)
    return JetPIDReport(
        step=step,
        proportional=proportional,
        integral=integral,
        derivative=derivative,
        control=control,
        model_value=used[0],
        integral_mode=cfg.integral_mode,
        running_integral=running,
        remainder_hi=remainder_hi,
        refused=False,
        reason="ok",
    )


def worked_example() -> dict[str, float | bool]:
    """G1: ``phi(s) = (s-1)^2`` from ``s=0`` with ``kp=0.5`` steps to 1."""
    # phi = 1 - 2s + s^2  =>  phi(0)=1, phi'=-2, phi''=2
    report = pid_step_from_derivatives(
        (1.0, -2.0, 2.0),
        config=JetPIDConfig(kp=0.5, ki=0.0, kd=0.0, s_max=2.0),
    )
    damped = pid_step_from_derivatives(
        (1.0, -2.0, 2.0),
        config=JetPIDConfig(kp=1.0, ki=0.0, kd=0.5, s_max=2.0),
    )
    overshoot = pid_step_from_derivatives(
        (1.0, -2.0, 2.0),
        config=JetPIDConfig(kp=1.0, ki=0.0, kd=0.0, s_max=4.0),
    )
    return {
        "step": report.step,
        "abs_step_err": abs(report.step - 1.0),
        "proportional": report.proportional,
        "damped_step": damped.step,
        "overshoot_step": overshoot.step,
        "damped_is_newton": abs(damped.step - 1.0) < 1e-15,
    }


def pid_skill(*, n: int = 20, seed: int = 0) -> dict[str, object]:
    """G2: random bowls ``(s-c)^2`` plus a ``kd``-without-``phi''`` reject."""
    rng = random.Random(seed)
    errs: list[float] = []
    finite = True
    for _ in range(n):
        center = rng.uniform(0.25, 0.75)
        # phi(s)=(s-c)^2 => phi(0)=c^2, phi'=-2c, phi''=2
        report = pid_step_from_derivatives(
            (center * center, -2.0 * center, 2.0),
            config=JetPIDConfig(kp=0.5, ki=0.0, kd=0.0, s_max=2.0),
        )
        if not math.isfinite(report.step):
            finite = False
        errs.append(abs(report.step - center))
    kd_raised = False
    try:
        pid_step_from_derivatives(
            (1.0, -2.0),
            config=JetPIDConfig(kp=0.0, ki=0.0, kd=1.0),
        )
    except ValueError as exc:
        kd_raised = "three derivatives" in str(exc)
    median = statistics.median(errs) if errs else math.inf
    return {
        "median_err": median,
        "all_finite": finite,
        "kd_raised": kd_raised,
        "g2_earned": finite and median < 1e-12 and kd_raised,
    }


__all__ = [
    "DEFAULT_CONFIG",
    "DISCLAIMER",
    "JetPIDConfig",
    "JetPIDReport",
    "honesty_payload",
    "model_definite_integral",
    "model_ftc_increment",
    "pid_skill",
    "pid_step_from_derivatives",
    "poly_antiderivative",
    "worked_example",
]
