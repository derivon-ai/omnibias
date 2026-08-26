# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Plant PID layer with closed-form I and D (theory 09-29).

The measurement is an activation of time, ``y(t) = sigma(alpha t + beta)``.
The tracking error is ``e = r - y``. Then

* **P** -- ``e(t)``
* **D** -- ``de/dt = -alpha sigma'(z)`` from the founding bias collapse
  tower (``delta -> 0``)
* **I** -- exact FTC ``r (t-t0) - (S(z)-S(z0))/alpha`` when
  ``alpha != 0``, or ``e (t-t0)`` when ``alpha = 0``. ``S' = sigma``
  is the unused ``integral`` role.

This is a **plant controller**, not the 08-10 jet-PID *trainer*.
Temperature collapse (``beta -> inf``, feasibility) does not appear.
do not conflate the two. Not cruise-control SOTA. Not CCF stretch.
``theorem_prover_verified`` is not asserted.
"""

from __future__ import annotations

import math
import random
import statistics
from dataclasses import dataclass
from typing import Literal

from omnibias.core.ftc import sigmoid, softplus

DISCLAIMER = (
    "plant PID layer: exact I/D on an activation-of-time error; "
    "not 08-10, not cruise SOTA, not CCF stretch"
)

Family = Literal["sigmoid", "tanh"]


def honesty_payload() -> dict[str, bool]:
    return {
        "trainer_claimed": False,
        "cruise_sota_claimed": False,
        "stretch_claim": False,
        "continuum_claimed": False,
        "theorem_prover_verified": False,
        "skips_chain_rule": False,
        "discrete_sum_i": False,
    }


@dataclass(frozen=True)
class PlantPIDConfig:
    kp: float = 1.0
    ki: float = 0.0
    kd: float = 0.0
    setpoint: float = 0.5
    alpha: float = 1.0
    beta: float = 0.0
    t0: float = 0.0
    family: Family = "sigmoid"


DEFAULT_CONFIG = PlantPIDConfig()


@dataclass(frozen=True)
class PlantPIDReport:
    control: float
    proportional: float
    integral: float
    derivative: float
    measurement: float
    error: float
    family: str


def _require_finite(name: str, value: float) -> float:
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return out


def log_cosh(z: float) -> float:
    """Stable ``log(cosh z)`` = ``|z| + log1p(exp(-2|z|)) - log(2)``."""
    az = abs(_require_finite("z", z))
    return az + math.log1p(math.exp(-2.0 * az)) - math.log(2.0)


def activation_value(z: float, family: Family) -> float:
    if family == "sigmoid":
        return sigmoid(z)
    if family == "tanh":
        return math.tanh(z)
    raise ValueError(f"unknown family {family!r}")


def activation_prime(z: float, family: Family) -> float:
    """``sigma'(z)`` from the Riccati identity, not a finite difference."""
    if family == "sigmoid":
        s = sigmoid(z)
        return s * (1.0 - s)
    if family == "tanh":
        t = math.tanh(z)
        return 1.0 - t * t
    raise ValueError(f"unknown family {family!r}")


def activation_integral(z: float, family: Family) -> float:
    """Antiderivative ``S`` with ``S' = sigma`` (the ``integral`` role)."""
    if family == "sigmoid":
        return softplus(z)
    if family == "tanh":
        return log_cosh(z)
    raise ValueError(f"unknown family {family!r}")


def error_integral(
    t: float,
    t0: float,
    *,
    setpoint: float,
    alpha: float,
    beta: float,
    family: Family,
) -> float:
    """Exact FTC of ``r - sigma(alpha s + beta)`` on ``[t0, t]``."""
    t_f = _require_finite("t", t)
    t0_f = _require_finite("t0", t0)
    r = _require_finite("setpoint", setpoint)
    a = _require_finite("alpha", alpha)
    b = _require_finite("beta", beta)
    z = a * t_f + b
    z0 = a * t0_f + b
    if a == 0.0:
        return (r - activation_value(b, family)) * (t_f - t0_f)
    return r * (t_f - t0_f) - (activation_integral(z, family) - activation_integral(z0, family)) / a


def plant_pid(
    t: float,
    *,
    config: PlantPIDConfig | None = None,
) -> PlantPIDReport:
    """PID control at time ``t`` for ``y = sigma(alpha t + beta)``."""
    cfg = DEFAULT_CONFIG if config is None else config
    if cfg.family not in ("sigmoid", "tanh"):
        raise ValueError(f"unknown family {cfg.family!r}")
    kp = _require_finite("kp", cfg.kp)
    ki = _require_finite("ki", cfg.ki)
    kd = _require_finite("kd", cfg.kd)
    t_f = _require_finite("t", t)
    alpha = _require_finite("alpha", cfg.alpha)
    beta = _require_finite("beta", cfg.beta)
    _require_finite("setpoint", cfg.setpoint)
    _require_finite("t0", cfg.t0)
    z = alpha * t_f + beta
    y = activation_value(z, cfg.family)
    error = cfg.setpoint - y
    deriv = -alpha * activation_prime(z, cfg.family)
    integral = error_integral(
        t_f,
        cfg.t0,
        setpoint=cfg.setpoint,
        alpha=alpha,
        beta=beta,
        family=cfg.family,
    )
    control = kp * error + ki * integral + kd * deriv
    if not math.isfinite(control):
        raise ValueError(f"PID control is not finite, got {control!r}")
    return PlantPIDReport(
        control=control,
        proportional=error,
        integral=integral,
        derivative=deriv,
        measurement=y,
        error=error,
        family=cfg.family,
    )


def _trap_error_integral(config: PlantPIDConfig, t: float, *, panels: int = 8000) -> float:
    if panels < 1:
        raise ValueError("panels must be >= 1")
    width = t - config.t0
    if width == 0.0:
        return 0.0
    h = width / float(panels)

    def err_at(s: float) -> float:
        return config.setpoint - activation_value(config.alpha * s + config.beta, config.family)

    acc = 0.5 * (err_at(config.t0) + err_at(t))
    for i in range(1, panels):
        acc += err_at(config.t0 + i * h)
    return acc * h


def worked_example() -> dict[str, float | bool]:
    """G1: sigmoid window ``[-1, 0]``, exact I vs trapezoid, exact D."""
    cfg = PlantPIDConfig(
        kp=0.0, ki=1.0, kd=1.0, setpoint=0.5, alpha=1.0, beta=0.0, t0=-1.0
    )
    report = plant_pid(0.0, config=cfg)
    trap = _trap_error_integral(cfg, 0.0)
    expected_d = -activation_prime(0.0, "sigmoid")
    return {
        "integral": report.integral,
        "ftc_residual": abs(report.integral - trap),
        "derivative": report.derivative,
        "d_residual": abs(report.derivative - expected_d),
        "measurement": report.measurement,
        "control": report.control,
    }


def plant_pid_skill(*, n: int = 16, seed: int = 0) -> dict[str, object]:
    """G2: random windows, both families; unknown family raises."""
    rng = random.Random(seed)
    residuals: list[float] = []
    finite = True
    for _ in range(n):
        family: Family = "tanh" if rng.random() < 0.5 else "sigmoid"
        cfg = PlantPIDConfig(
            kp=0.0,
            ki=1.0,
            kd=0.0,
            setpoint=rng.uniform(-0.2, 0.8),
            alpha=rng.choice((-1.5, -0.5, 0.0, 0.75, 2.0)),
            beta=rng.uniform(-0.5, 0.5),
            t0=rng.uniform(-1.0, -0.2),
            family=family,
        )
        t = cfg.t0 + rng.uniform(0.2, 1.2)
        report = plant_pid(t, config=cfg)
        if not math.isfinite(report.integral):
            finite = False
        residuals.append(abs(report.integral - _trap_error_integral(cfg, t)))
    unknown = False
    try:
        plant_pid(0.0, config=PlantPIDConfig(family="relu"))  # type: ignore[arg-type]
    except ValueError as exc:
        unknown = "unknown family" in str(exc)
    median = statistics.median(residuals) if residuals else math.inf
    return {
        "median_ftc_residual": median,
        "all_finite": finite,
        "unknown_raised": unknown,
        "g2_earned": finite and median < 1e-8 and unknown,
    }


__all__ = [
    "DEFAULT_CONFIG",
    "DISCLAIMER",
    "PlantPIDConfig",
    "PlantPIDReport",
    "activation_integral",
    "activation_prime",
    "activation_value",
    "error_integral",
    "honesty_payload",
    "log_cosh",
    "plant_pid",
    "plant_pid_skill",
    "worked_example",
]
