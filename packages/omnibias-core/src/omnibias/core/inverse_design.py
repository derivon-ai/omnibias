# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Newton-on-input inverse design (theory 09-22).

``r(x) = f(x) - y`` is driven by exact ``sigma'`` from founding
bias collapse (``delta -> 0``). Temperature collapse
(``beta -> inf``, feasibility) does not appear. do not conflate
the two.

This inverts the net as a map on ``x``. It is not 08-03 (that
inverts a *layer* for a hidden target). Not a global inverse.
Not CCF stretch. Optional 08-04 is a finite-map ball, not a
continuum theorem. ``theorem_prover_verified`` is not asserted.
"""

from __future__ import annotations

import math
import random
import statistics
from dataclasses import dataclass

from omnibias.core.verified.interval import Interval
from omnibias.core.verified.kantorovich import (
    IntervalJac,
    IntervalMap,
    kantorovich_accept_step,
)

DISCLAIMER = (
    "local Newton-on-x; not 08-03 layer invert, not a global inverse, "
    "not CCF stretch"
)


def honesty_payload() -> dict[str, bool]:
    return {
        "global_inverse_claimed": False,
        "layer_invert_claimed": False,
        "stretch_claim": False,
        "continuum_claimed": False,
        "theorem_prover_verified": False,
        "n_roots_unknown": True,
    }


@dataclass(frozen=True)
class InverseDesignConfig:
    tol: float = 1e-10
    min_sigma_prime: float = 1e-4
    require_ball: bool = False
    max_steps: int = 16
    sat_abs: float = 0.999
    scale: float = 2.0
    r_max: float = 0.5


DEFAULT_CONFIG = InverseDesignConfig()


@dataclass(frozen=True)
class InverseDesignReport:
    x: float
    y: float
    residual: float
    n_steps: int
    sigma_prime: float
    f_prime: float
    ball_accepted: bool | None
    n_roots_unknown: bool = True


def tanh_scale(x: float, scale: float) -> float:
    return math.tanh(scale * x)


def tanh_scale_prime(x: float, scale: float) -> tuple[float, float]:
    """``(sigma', f')`` with ``sigma' = sech^2(scale x)``, ``f' = scale sigma'``."""
    val = math.tanh(scale * x)
    sigma_p = 1.0 - val * val
    return sigma_p, scale * sigma_p


def tanh_scale_target(y: float, scale: float) -> float:
    if abs(y) >= 1.0:
        raise ValueError(f"tanh target requires |y| < 1, got {y}")
    return 0.5 * math.log((1.0 + y) / (1.0 - y)) / scale


def _tanh_maps(y: float, scale: float) -> tuple[IntervalMap, IntervalJac, float]:
    y_iv = Interval.point(float(y))

    def func(xs: list[Interval]) -> list[Interval]:
        mid = float(xs[0].lo + xs[0].hi) / 2.0
        return [Interval.point(math.tanh(scale * mid)) - y_iv]

    def jacobian(xs: list[Interval]) -> list[list[Interval]]:
        mid = float(xs[0].lo + xs[0].hi) / 2.0
        _, f_p = tanh_scale_prime(mid, scale)
        return [[Interval.point(f_p)]]

    return func, jacobian, 8.0 * abs(scale) / 2.0


def invert_input(
    f: object | None,
    y: float,
    x0: float,
    *,
    config: InverseDesignConfig | None = None,
    family: str = "tanh_scale",
) -> InverseDesignReport:
    """Newton-on-``x``. Raises if ``|sigma'|`` is tiny or ``y`` is saturated."""
    del f
    if family != "tanh_scale":
        raise ValueError(f"unknown family {family!r}")
    cfg = DEFAULT_CONFIG if config is None else config
    target = float(y)
    if abs(target) >= cfg.sat_abs:
        raise ValueError(
            f"saturated target |y|={abs(target)} >= sat_abs={cfg.sat_abs}; "
            "local invert refuses to silently diverge"
        )
    if abs(target) >= 1.0:
        raise ValueError(f"tanh target requires |y| < 1, got {target}")
    x = float(x0)
    sigma_p = 1.0
    f_p = cfg.scale
    steps = 0
    residual = tanh_scale(x, cfg.scale) - target
    while steps < cfg.max_steps:
        steps += 1
        sigma_p, f_p = tanh_scale_prime(x, cfg.scale)
        if abs(sigma_p) < cfg.min_sigma_prime:
            raise ValueError(
                f"|sigma'|={abs(sigma_p)} < min_sigma_prime={cfg.min_sigma_prime}"
            )
        residual = tanh_scale(x, cfg.scale) - target
        if abs(residual) < cfg.tol:
            break
        x = x - residual / f_p
    residual = tanh_scale(x, cfg.scale) - target
    ball_accepted: bool | None = None
    if cfg.require_ball:
        func, jac, lip = _tanh_maps(target, cfg.scale)
        a_inv = [[1.0 / f_p]] if abs(f_p) >= 1e-18 else []
        decision = kantorovich_accept_step(
            func, jac, a_inv, [x], lipschitz_df=lip, r_max=cfg.r_max
        )
        ball_accepted = decision.accepted
        if not decision.accepted:
            raise ValueError("08-04 unique-zero ball is empty; invert halted")
    return InverseDesignReport(
        x=x,
        y=target,
        residual=residual,
        n_steps=steps,
        sigma_prime=sigma_p,
        f_prime=f_p,
        ball_accepted=ball_accepted,
    )


def worked_example() -> dict[str, float]:
    """G1: ``tanh(2x) = 0.5`` from ``x=0``."""
    report = invert_input(None, 0.5, 0.0)
    target = tanh_scale_target(0.5, 2.0)
    return {
        "x": report.x,
        "target": target,
        "residual": abs(report.residual),
        "abs_x_err": abs(report.x - target),
        "sigma_prime": report.sigma_prime,
        "f_prime": report.f_prime,
    }


def invert_skill(*, n: int = 20, seed: int = 0) -> dict[str, object]:
    """G2: random ``y`` in ``(-0.8, 0.8)`` plus a saturated reject."""
    rng = random.Random(seed)
    errs: list[float] = []
    finite = True
    for _ in range(n):
        y = rng.uniform(-0.8, 0.8)
        report = invert_input(None, y, 0.0)
        if not math.isfinite(report.x) or not math.isfinite(report.residual):
            finite = False
        errs.append(abs(report.x - tanh_scale_target(y, 2.0)))
    saturated_raised = False
    try:
        invert_input(None, 0.999, 0.0)
    except ValueError as exc:
        saturated_raised = "saturat" in str(exc).lower()
    median = statistics.median(errs) if errs else math.inf
    return {
        "median_err": median,
        "all_finite": finite,
        "saturated_raised": saturated_raised,
        "g2_earned": finite and median < 1e-10 and saturated_raised,
    }


__all__ = [
    "DEFAULT_CONFIG",
    "DISCLAIMER",
    "InverseDesignConfig",
    "InverseDesignReport",
    "honesty_payload",
    "invert_input",
    "invert_skill",
    "tanh_scale",
    "tanh_scale_prime",
    "tanh_scale_target",
    "worked_example",
]
