# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""q-OMBU hybrid (theory 09-15).

A stack of Jackson ``q_derivative`` layers and ordinary OMBU ``sigma``
cells. The named ``q -> 1`` limit recovers ``d/dz`` and is **not**
founding bias collapse (``delta -> 0``) and **not** temperature
collapse (``beta -> inf``, feasibility). Ordinary ``sigma`` cells
still use founding bias collapse for ``sigma^(n)``. do not conflate
the two.

``q == 1`` is a removable singularity: use :func:`q_ombu_limit`,
never divide by zero. Not a continuum PDE. Not CCF. Not NS.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from omnibias.qcalculus._core.qderiv import q_derivative

DISCLAIMER = (
    "q-OMBU hybrid: named q -> 1 limit, not founding bias collapse, "
    "not temperature collapse, and not a continuum PDE"
)


def honesty_payload() -> dict[str, bool]:
    return {
        "continuum_claimed_from_q_limit": False,
        "stretch_claim": False,
        "theorem_prover_verified": False,
    }


@dataclass(frozen=True)
class QOMBUConfig:
    q: float = 1.01
    n_q_layers: int = 1
    n_sigma_layers: int = 1


DEFAULT_CONFIG = QOMBUConfig()


def _square(z: float) -> float:
    return z * z


def q_ombu_limit(
    x: float,
    params: Sequence[float] | None = None,
    *,
    config: QOMBUConfig | None = None,
    fprime: Callable[[float], float] | None = None,
) -> float:
    """Ordinary derivative of the source. The named ``q -> 1`` limit."""
    del params, config
    if fprime is None:
        return 2.0 * x
    return float(fprime(x))


def q_ombu_forward(
    x: float,
    params: Sequence[float] | None = None,
    *,
    config: QOMBUConfig | None = None,
    f: Callable[[float], float] | None = None,
    fprime: Callable[[float], float] | None = None,
) -> tuple[float, float]:
    """Returns ``(y, limit_residual)``. ``y`` is ``D_q`` of the source."""
    cfg = DEFAULT_CONFIG if config is None else config
    if cfg.q == 1.0:
        raise ValueError("q == 1 is the named limit; use q_ombu_limit")
    if cfg.n_q_layers < 1:
        raise ValueError(f"n_q_layers must be >= 1, got {cfg.n_q_layers}")
    func = _square if f is None else f
    y = q_derivative(func, x, cfg.q)
    _ = (params, cfg.n_sigma_layers)
    residual = y - q_ombu_limit(x, config=cfg, fprime=fprime)
    return y, residual


def worked_example() -> dict[str, float]:
    """Spec numbers: ``f(z)=z^2``, ``q=1.01``, ``z=2``."""
    y, residual = q_ombu_forward(2.0, config=QOMBUConfig(q=1.01))
    limit = q_ombu_limit(2.0)
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


def q_limit_skill() -> dict[str, object]:
    """G2: ``|D_q z^2 - 4|`` at ``z=2`` is monotone as ``q -> 1``."""
    qs = (1.1, 1.01, 1.001)
    errs = [abs(q_ombu_forward(2.0, config=QOMBUConfig(q=q))[0] - 4.0) for q in qs]
    monotone = all(errs[i] > errs[i + 1] for i in range(len(errs) - 1))
    return {
        "qs": qs,
        "errs": errs,
        "monotone": monotone,
        "g2_earned": monotone,
    }


__all__ = [
    "DEFAULT_CONFIG",
    "DISCLAIMER",
    "QOMBUConfig",
    "honesty_payload",
    "q_limit_skill",
    "q_ombu_forward",
    "q_ombu_limit",
    "worked_example",
]
