# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Kantorovich homotopy continuation (theory 09-20).

A path of finite residual maps ``H(theta, tau)``. Each knot takes a
Newton trial and keeps it only if the 08-04 unique-zero ball is
nonempty. An empty ball is a halt, not a forged root.

Jets / ``sigma''`` on a nest use founding bias collapse
(``delta -> 0``). Temperature collapse (``beta -> inf``, feasibility)
does not appear. do not conflate the two.

Not a rewrite of one-step 08-04. Not CCF stretch. Not a continuum
PDE. ``theorem_prover_verified`` is not asserted.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

from omnibias.core.verified.interval import Interval
from omnibias.core.verified.kantorovich import (
    IntervalJac,
    IntervalMap,
    KantorovichAccept,
    kantorovich_accept_step,
)

DISCLAIMER = (
    "Kantorovich homotopy: 08-04 filter on a tau-path; empty ball is a "
    "halt, not a continuum theorem and not CCF stretch"
)


def honesty_payload() -> dict[str, bool]:
    return {
        "stretch_claim": False,
        "continuum_claimed": False,
        "theorem_prover_verified": False,
        "empty_ball_is_reject": True,
    }


@dataclass(frozen=True)
class HomotopyConfig:
    n_knots: int = 8
    require_ball: bool = True
    newton_iters: int = 8
    r_max: float = 0.5
    residual_tol: float = 1e-12


DEFAULT_CONFIG = HomotopyConfig()


@dataclass(frozen=True)
class HomotopyReport:
    theta: float
    tau_final: float
    halted: bool
    balls: tuple[bool, ...]
    residual: float
    claimed_root: bool


def quadratic_h(theta: float, tau: float) -> float:
    """``H = theta - 1 - tau theta^2``."""
    return theta - 1.0 - tau * theta * theta


def quadratic_dh(theta: float, tau: float) -> float:
    return 1.0 - 2.0 * tau * theta


def quadratic_root(tau: float) -> float:
    """Branch near ``1`` for ``0 < tau <= 0.25``."""
    if tau <= 0.0:
        return 1.0
    disc = 1.0 - 4.0 * tau
    if disc < 0.0:
        raise ValueError(f"no real root for tau={tau}")
    return (1.0 - math.sqrt(disc)) / (2.0 * tau)


def quadratic_maps(tau: float) -> tuple[IntervalMap, IntervalJac, float]:
    """Interval residual of the worked quadratic. ``Lip(H') = 2 tau``."""
    tau_iv = Interval.point(float(tau))
    one = Interval.point(1.0)
    two_tau = Interval.point(2.0 * float(tau))

    def func(xs: list[Interval]) -> list[Interval]:
        th = xs[0]
        return [th - one - tau_iv * th * th]

    def jacobian(xs: list[Interval]) -> list[list[Interval]]:
        return [[one - two_tau * xs[0]]]

    return func, jacobian, 2.0 * float(tau)


def semilinear_h(theta: float, tau: float) -> float:
    """Scalar ``-u + tau u^3`` toy of ``-u'' + tau u^3 = f`` with ``f=0.2``."""
    return theta - 0.2 - tau * theta * theta * theta


def semilinear_dh(theta: float, tau: float) -> float:
    return 1.0 - 3.0 * tau * theta * theta


def semilinear_maps(tau: float) -> tuple[IntervalMap, IntervalJac, float]:
    """``Lip(H') <= 12 tau`` on ``|u| <= 2``."""
    tau_iv = Interval.point(float(tau))
    shift = Interval.point(0.2)
    three_tau = Interval.point(3.0 * float(tau))

    def func(xs: list[Interval]) -> list[Interval]:
        th = xs[0]
        return [th - shift - tau_iv * th * th * th]

    def jacobian(xs: list[Interval]) -> list[list[Interval]]:
        th = xs[0]
        return [[Interval.point(1.0) - three_tau * th * th]]

    return func, jacobian, 12.0 * float(tau)


Residual2 = Callable[[float, float], float]


def _newton(theta: float, tau: float, h_fn: Residual2, dh_fn: Residual2, iters: int) -> float:
    val = float(theta)
    for _ in range(iters):
        deriv = dh_fn(val, tau)
        if abs(deriv) < 1e-18:
            return val
        val = val - h_fn(val, tau) / deriv
    return val


def _knots(n_knots: int) -> list[float]:
    if n_knots < 1:
        raise ValueError(f"n_knots must be >= 1, got {n_knots}")
    return [float(k) / float(n_knots) for k in range(1, n_knots + 1)]


def homotopy_step(
    theta: float,
    tau: float,
    *,
    config: HomotopyConfig | None = None,
    family: str = "quadratic",
) -> tuple[float, float, KantorovichAccept]:
    """Newton at one ``tau``, then the 08-04 accept/reject."""
    cfg = DEFAULT_CONFIG if config is None else config
    if family == "quadratic":
        h_fn, dh_fn, maps = quadratic_h, quadratic_dh, quadratic_maps
    elif family == "semilinear":
        h_fn, dh_fn, maps = semilinear_h, semilinear_dh, semilinear_maps
    else:
        raise ValueError(f"unknown family {family!r}")
    trial = _newton(theta, tau, h_fn, dh_fn, cfg.newton_iters)
    func, jac, lip = maps(tau)
    deriv = dh_fn(trial, tau)
    a_inv: list[list[float]] = [[1.0 / deriv]] if abs(deriv) >= 1e-18 else []
    decision = kantorovich_accept_step(
        func,
        jac,
        a_inv,
        [trial],
        lipschitz_df=lip,
        r_max=cfg.r_max,
    )
    return trial, abs(h_fn(trial, tau)), decision


def homotopy_train(
    h_fn: object | None = None,
    theta0: float = 1.0,
    *,
    config: HomotopyConfig | None = None,
    family: str = "quadratic",
) -> HomotopyReport:
    """Walk ``tau: 0 -> 1``. Halt if 08-04 rejects. Not a continuum PDE."""
    del h_fn
    cfg = DEFAULT_CONFIG if config is None else config
    theta = float(theta0)
    balls: list[bool] = []
    residual = 0.0
    for tau in _knots(cfg.n_knots):
        trial, residual, decision = homotopy_step(theta, tau, config=cfg, family=family)
        balls.append(decision.accepted)
        if cfg.require_ball and not decision.accepted:
            return HomotopyReport(
                theta=theta,
                tau_final=tau,
                halted=True,
                balls=tuple(balls),
                residual=residual,
                claimed_root=False,
            )
        theta = trial
    claimed = residual < 1e-6
    return HomotopyReport(
        theta=theta,
        tau_final=1.0,
        halted=False,
        balls=tuple(balls),
        residual=residual,
        claimed_root=claimed,
    )


def worked_example() -> dict[str, float | bool]:
    """G1: ``tau=0.1`` from ``theta=1``."""
    trial, residual, decision = homotopy_step(1.0, 0.1, config=HomotopyConfig())
    root = quadratic_root(0.1)
    return {
        "theta": trial,
        "root": root,
        "residual": residual,
        "accepted": decision.accepted,
        "reason": 1.0 if decision.reason == "ball" else 0.0,
        "abs_h": abs(quadratic_h(trial, 0.1)),
        "root_err": abs(trial - root),
    }


def homotopy_skill(*, seeds: int = 5) -> dict[str, object]:
    """G2: semilinear seeds, or an honest halt on the quadratic."""
    reached = 0
    halted_honest = True
    residuals: list[float] = []
    for seed in range(seeds):
        start = 0.2 + 0.01 * float(seed - 2)
        report = homotopy_train(theta0=start, family="semilinear")
        residuals.append(report.residual)
        if report.claimed_root and report.halted:
            halted_honest = False
        if report.tau_final == 1.0 and report.residual < 1e-6 and not report.halted:
            reached += 1
    quad = homotopy_train(theta0=1.0, family="quadratic")
    if quad.claimed_root and quad.halted:
        halted_honest = False
    return {
        "reached": reached,
        "residuals": residuals,
        "quad_halted": quad.halted,
        "quad_tau": quad.tau_final,
        "quad_claimed_root": quad.claimed_root,
        "honest": halted_honest,
        "g2_earned": halted_honest and (reached >= 3 or quad.halted),
    }


__all__ = [
    "DEFAULT_CONFIG",
    "DISCLAIMER",
    "HomotopyConfig",
    "HomotopyReport",
    "homotopy_skill",
    "homotopy_step",
    "homotopy_train",
    "honesty_payload",
    "quadratic_h",
    "quadratic_maps",
    "quadratic_root",
    "semilinear_h",
    "semilinear_maps",
    "worked_example",
]
