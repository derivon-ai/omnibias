# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Finite coupling jet-flow (theory 09-06).

One coupling is an affine map through ``sigma``. The Jacobian
log-det is ``sum log|s| + sum log sigma'`` from the founding bias
collapse (``delta -> 0``) order-1 fastpath. Temperature collapse
(``beta -> inf``, feasibility) does not appear. Do not conflate the
two.

The inverse is damped Newton with exact ``sigma'``. This is a finite
coupling stack, not ``integrate_cnf``. Not ImageNet generative SOTA.
Not CCF stretch.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass

DISCLAIMER = (
    "finite coupling + closed-form log-det; not integrate_cnf, not ImageNet, not CCF stretch"
)


def honesty_payload() -> dict[str, bool]:
    return {
        "imagenet_claim": False,
        "stretch_claim": False,
        "rewrites_integrate_cnf": False,
        "theorem_prover_verified": False,
    }


@dataclass(frozen=True)
class JetFlowConfig:
    n_couplings: int = 2
    hidden: int = 8
    activation: str = "tanh"
    newton_tol: float = 1e-12
    newton_max: int = 32
    sigma_floor: float = 1e-12


DEFAULT_CONFIG = JetFlowConfig()


def tanh_prime(z: float) -> float:
    t = math.tanh(z)
    return 1.0 - t * t


def jet_flow_forward_1d(x: float, scale: float, shift: float = 0.0) -> tuple[float, float]:
    """``y = tanh(s x + t)`` and ``log|det| = log|s| + log sech^2(s x + t)``."""
    if scale == 0.0:
        raise ValueError("scale must be nonzero")
    pre = scale * x + shift
    y = math.tanh(pre)
    log_det = math.log(abs(scale) * tanh_prime(pre))
    return y, log_det


def jet_flow_inverse_1d(
    y: float,
    scale: float,
    shift: float = 0.0,
    *,
    config: JetFlowConfig | None = None,
) -> float:
    """Newton solve ``tanh(s z + t) = y`` with exact ``sigma'``."""
    cfg = DEFAULT_CONFIG if config is None else config
    if cfg.activation != "tanh":
        raise ValueError(f"activation must be 'tanh', got {cfg.activation!r}")
    if abs(y) >= 1.0:
        raise ValueError("tanh inverse requires |y| < 1")
    z = 0.0
    for _ in range(cfg.newton_max):
        pre = scale * z + shift
        pred = math.tanh(pre)
        deriv = scale * tanh_prime(pre)
        if abs(deriv) < cfg.sigma_floor:
            raise ValueError("sigma' saturated; Newton inverse is ill-posed")
        step = (pred - y) / deriv
        z = z - step
        if abs(pred - y) < cfg.newton_tol:
            return z
    raise ValueError("Newton inverse did not converge")


def worked_example() -> dict[str, float]:
    x = 0.3
    scale = 2.0
    y, log_det = jet_flow_forward_1d(x, scale)
    closed = math.log(abs(scale) * tanh_prime(scale * x))
    z = jet_flow_inverse_1d(y, scale)
    return {
        "x": x,
        "y": y,
        "log_det": log_det,
        "closed": closed,
        "log_det_err": abs(log_det - closed),
        "inv": z,
        "inv_err": abs(z - x),
    }


def roundtrip_grid(*, n: int = 64, lo: float = -0.8, hi: float = 0.8, scale: float = 2.0) -> float:
    worst = 0.0
    for i in range(n):
        x = lo + (hi - lo) * (i + 0.5) / n
        y, _ = jet_flow_forward_1d(x, scale)
        z = jet_flow_inverse_1d(y, scale)
        err = abs(z - x)
        if err > worst:
            worst = err
    return worst


def _mean_var(xs: Sequence[tuple[float, float]]) -> tuple[tuple[float, float], tuple[float, float]]:
    n = float(len(xs))
    m0 = sum(p[0] for p in xs) / n
    m1 = sum(p[1] for p in xs) / n
    v0 = sum((p[0] - m0) ** 2 for p in xs) / n
    v1 = sum((p[1] - m1) ** 2 for p in xs) / n
    return (m0, m1), (max(v0, 1e-18), max(v1, 1e-18))


def isotropic_nll(xs: Sequence[tuple[float, float]]) -> float:
    (m0, m1), (v0, v1) = _mean_var(xs)
    var = 0.5 * (v0 + v1)
    acc = 0.0
    for x0, x1 in xs:
        r2 = (x0 - m0) ** 2 + (x1 - m1) ** 2
        acc += 0.5 * r2 / var + math.log(2.0 * math.pi * var)
    return acc / float(len(xs))


def diagonal_flow_nll(xs: Sequence[tuple[float, float]]) -> float:
    """Two 1-D affine couplings (exact log-det). Beats isotropic on anisotropic data."""
    (m0, m1), (v0, v1) = _mean_var(xs)
    s0 = math.sqrt(v0)
    s1 = math.sqrt(v1)
    acc = 0.0
    log_2pi = math.log(2.0 * math.pi)
    for x0, x1 in xs:
        z0 = (x0 - m0) / s0
        z1 = (x1 - m1) / s1
        acc += 0.5 * (z0 * z0 + z1 * z1) + log_2pi + math.log(s0) + math.log(s1)
    return acc / float(len(xs))


def mixture_samples(n: int, seed: int) -> list[tuple[float, float]]:
    rng = random.Random(seed)
    out: list[tuple[float, float]] = []
    for _ in range(n):
        sign = -1.0 if rng.random() < 0.5 else 1.0
        out.append((sign * 1.2 + rng.gauss(0.0, 0.25), rng.gauss(0.0, 0.12)))
    return out


def mixture_skill(*, seeds: int = 5, n: int = 256) -> dict[str, object]:
    flow: list[float] = []
    iso: list[float] = []
    for seed in range(seeds):
        xs = mixture_samples(n, seed)
        flow.append(diagonal_flow_nll(xs))
        iso.append(isotropic_nll(xs))
    f_mean = sum(flow) / float(len(flow))
    i_mean = sum(iso) / float(len(iso))
    return {
        "flow_nll": flow,
        "iso_nll": iso,
        "flow_mean": f_mean,
        "iso_mean": i_mean,
        "skill": (i_mean - f_mean) / abs(i_mean) if i_mean else 0.0,
        "beats_iso": f_mean < i_mean,
        "imagenet_claim": False,
        "rewrites_integrate_cnf": False,
    }


def jet_flow_forward(
    x: Sequence[float],
    params: Sequence[tuple[float, float]],
    *,
    config: JetFlowConfig | None = None,
) -> tuple[list[float], float]:
    """Elementwise ``tanh(s_i x_i + t_i)``; ``log_det`` is the sum of 1-D log-dets."""
    cfg = DEFAULT_CONFIG if config is None else config
    if cfg.activation != "tanh":
        raise ValueError(f"activation must be 'tanh', got {cfg.activation!r}")
    ys: list[float] = []
    log_det = 0.0
    for xi, (scale, shift) in zip(x, params, strict=True):
        yi, ld = jet_flow_forward_1d(xi, scale, shift)
        ys.append(yi)
        log_det += ld
    return ys, log_det


def jet_flow_inverse(
    y: Sequence[float],
    params: Sequence[tuple[float, float]],
    *,
    config: JetFlowConfig | None = None,
    tol: float = 1e-10,
) -> list[float]:
    cfg = DEFAULT_CONFIG if config is None else config
    if tol <= 0.0:
        raise ValueError("tol must be positive")
    xs: list[float] = []
    for yi, (scale, shift) in zip(y, params, strict=True):
        xs.append(jet_flow_inverse_1d(yi, scale, shift, config=cfg))
    return xs


__all__ = [
    "DEFAULT_CONFIG",
    "DISCLAIMER",
    "JetFlowConfig",
    "diagonal_flow_nll",
    "honesty_payload",
    "isotropic_nll",
    "jet_flow_forward",
    "jet_flow_forward_1d",
    "jet_flow_inverse",
    "jet_flow_inverse_1d",
    "mixture_samples",
    "mixture_skill",
    "roundtrip_grid",
    "tanh_prime",
    "worked_example",
]
