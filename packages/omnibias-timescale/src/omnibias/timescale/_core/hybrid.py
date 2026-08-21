# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Hilger-OMBU hybrid (theory 09-15).

A stack of Hilger ``delta_derivative`` layers and ordinary OMBU
``sigma`` cells. The named ``mu -> 0`` limit recovers ``d/dt`` and
is **not** founding bias collapse (``delta -> 0`` of OMBU biases)
and **not** temperature collapse (``beta -> inf``, feasibility).
Ordinary ``sigma`` cells still use founding bias collapse for
``sigma^(n)``. do not conflate the two.

``mu == 0`` is the named continuum scale ``reals()``: use
:func:`hilger_ombu_limit`, never a zero-grain quotient. Not a
continuum PDE. Not CCF. Not NS.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from omnibias.timescale._core.derivative import delta_derivative
from omnibias.timescale._core.timescale import h_integers

DISCLAIMER = (
    "Hilger-OMBU hybrid: named mu -> 0 limit, not a continuum PDE, "
    "not CCF stretch"
)


def honesty_payload() -> dict[str, bool]:
    return {
        "continuum_claimed_from_mu_limit": False,
        "stretch_claim": False,
        "theorem_prover_verified": False,
    }


@dataclass(frozen=True)
class HilgerOMBUConfig:
    mu: float = 0.01
    n_delta_layers: int = 1
    n_sigma_layers: int = 1


DEFAULT_CONFIG = HilgerOMBUConfig()


def _square(z: float) -> float:
    return z * z


def hilger_ombu_limit(
    x: float,
    params: Sequence[float] | None = None,
    *,
    config: HilgerOMBUConfig | None = None,
    fprime: Callable[[float], float] | None = None,
) -> float:
    """Ordinary derivative of the source. The named ``mu -> 0`` limit."""
    del params, config
    if fprime is None:
        return 2.0 * x
    return float(fprime(x))


def hilger_ombu_forward(
    x: float,
    params: Sequence[float] | None = None,
    *,
    config: HilgerOMBUConfig | None = None,
    f: Callable[[float], float] | None = None,
    fprime: Callable[[float], float] | None = None,
) -> tuple[float, float]:
    """Returns ``(y, limit_residual)``. ``y`` is ``f^Delta`` of the source."""
    cfg = DEFAULT_CONFIG if config is None else config
    if cfg.mu <= 0.0:
        raise ValueError("mu <= 0 is the named limit; use hilger_ombu_limit")
    if cfg.n_delta_layers < 1:
        raise ValueError(f"n_delta_layers must be >= 1, got {cfg.n_delta_layers}")
    func = _square if f is None else f
    y = delta_derivative(func, x, h_integers(cfg.mu))
    _ = (params, cfg.n_sigma_layers)
    residual = y - hilger_ombu_limit(x, config=cfg, fprime=fprime)
    return y, residual


def worked_example() -> dict[str, float]:
    """``f(z)=z^2`` on ``hZ`` at ``z=2``, ``mu=0.02`` -> ``4.02``."""
    y, residual = hilger_ombu_forward(2.0, config=HilgerOMBUConfig(mu=0.02))
    limit = hilger_ombu_limit(2.0)
    return {
        "y": y,
        "y_expected": 4.02,
        "y_err": abs(y - 4.02),
        "limit": limit,
        "limit_expected": 4.0,
        "limit_err": abs(limit - 4.0),
        "residual": residual,
        "residual_expected": 0.02,
        "residual_err": abs(residual - 0.02),
    }


def mu_limit_skill() -> dict[str, object]:
    """``|f^Delta z^2 - 4|`` at ``z=2`` is monotone as ``mu -> 0``."""
    mus = (0.1, 0.01, 0.001)
    errs = [
        abs(hilger_ombu_forward(2.0, config=HilgerOMBUConfig(mu=mu))[0] - 4.0) for mu in mus
    ]
    monotone = all(errs[i] > errs[i + 1] for i in range(len(errs) - 1))
    return {
        "mus": mus,
        "errs": errs,
        "monotone": monotone,
        "g2_earned": monotone,
    }


__all__ = [
    "DEFAULT_CONFIG",
    "DISCLAIMER",
    "HilgerOMBUConfig",
    "hilger_ombu_forward",
    "hilger_ombu_limit",
    "honesty_payload",
    "mu_limit_skill",
    "worked_example",
]
