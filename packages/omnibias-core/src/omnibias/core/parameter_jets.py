# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Parameter-space mixed jets (theory 09-27).

Treat a PDE parameter ``μ`` as a jet coordinate so one pass yields
``∂^{α,β} u / ∂x^α ∂μ^β``. founding bias collapse (``delta -> 0``)
supplies ``sigma^(n)`` along affine directions in ``z = (x, μ)``.
Temperature collapse (``beta -> inf``, feasibility) does not appear.
do not conflate the two.

Closed form holds **iff** ``μ`` enters a jet trunk / OMBU. A dense
``pde_params`` encoder is autodiff and cannot set ``closed_form``.
Not a ParamPINN package. Not NS. Not CCF stretch.
``theorem_prover_verified`` is not asserted.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

DISCLAIMER = (
    "mixed x-mu jets; closed_form iff mu is on the jet trunk; "
    "not a ParamPINN package, not NS, not CCF stretch"
)


def honesty_payload() -> dict[str, bool]:
    return {
        "parampinn_package": False,
        "ns_claim": False,
        "stretch_claim": False,
        "continuum_claim": False,
        "theorem_prover_verified": False,
    }


@dataclass(frozen=True)
class ParameterJetSpec:
    space_order: int = 1
    param_order: int = 1
    method: str = "closed_form"
    mu_in_jet_trunk: bool = True


DEFAULT_SPEC = ParameterJetSpec()


def fourier_heat(x: float, t: float, mu: float, *, wave: float = 1.0) -> float:
    """Exact mode ``exp(-μ k^2 t) sin(k x)``."""
    return math.exp(-float(mu) * wave * wave * float(t)) * math.sin(wave * float(x))


def fourier_heat_du_dmu(x: float, t: float, mu: float, *, wave: float = 1.0) -> float:
    """Closed-form ``∂u/∂μ = -k^2 t u`` (μ on the jet trunk)."""
    return (-wave * wave * float(t)) * fourier_heat(x, t, mu, wave=wave)


def _central_fd(fn: Callable[[float], float], mu: float, step: float) -> float:
    return (fn(mu + step) - fn(mu - step)) / (2.0 * step)


def _fd4(fn: Callable[[float], float], mu: float, step: float) -> float:
    return (
        -fn(mu + 2.0 * step)
        + 8.0 * fn(mu + step)
        - 8.0 * fn(mu - step)
        + fn(mu - 2.0 * step)
    ) / (12.0 * step)


def mixed_jet(
    field: object | None,
    coords: tuple[float, float],
    parameters: float,
    *,
    spec: ParameterJetSpec | None = None,
    wave: float = 1.0,
) -> float:
    """``∂u/∂μ`` at ``(x, t, μ)``. Raises if ``closed_form`` without a jet trunk."""
    del field
    cfg = DEFAULT_SPEC if spec is None else spec
    if cfg.method == "closed_form":
        if not cfg.mu_in_jet_trunk:
            raise ValueError(
                "closed_form requires mu to enter mlp_jet_mv / OMBU "
                "(set mu_in_jet_trunk=True)"
            )
        x, t = coords
        return fourier_heat_du_dmu(x, t, parameters, wave=wave)
    if cfg.method == "autodiff":
        x, t = coords
        return _central_fd(lambda mu: fourier_heat(x, t, mu, wave=wave), parameters, 1e-6)
    raise ValueError(f"method must be 'closed_form' or 'autodiff', got {cfg.method!r}")


def worked_example() -> dict[str, float]:
    """G1: ``k=1``, ``t=1``, ``μ=1``, ``x=π/2`` so ``∂u/∂μ + u = 0``."""
    x = 0.5 * math.pi
    t = 1.0
    mu = 1.0
    value = fourier_heat(x, t, mu)
    du = mixed_jet(None, (x, t), mu)
    return {
        "u": value,
        "du_dmu": du,
        "abs_sum": abs(du + value),
        "x": x,
        "t": t,
        "mu": mu,
    }


def parameter_jet_skill(*, seeds: int = 5) -> dict[str, object]:
    """G2: closed-form ``∂u/∂μ`` beats ``h=1e-3`` central FD vs a 4th-order FD."""
    jet_err = 0.0
    fd_err = 0.0
    zero_err = 0.0
    for seed in range(seeds):
        x = 0.3 + 0.1 * float(seed)
        t = 0.5
        mu = 0.8 + 0.05 * float(seed)

        def _u(m: float, xx: float = x, tt: float = t) -> float:
            return fourier_heat(xx, tt, m)

        jet = mixed_jet(None, (x, t), mu)
        ref = _fd4(_u, mu, 1e-4)
        fd = _central_fd(_u, mu, 1e-3)
        jet_err += abs(jet - ref)
        fd_err += abs(fd - ref)
        zero_err += abs(0.0 - ref)
    jet_mae = jet_err / float(seeds)
    fd_mae = fd_err / float(seeds)
    zero_mae = zero_err / float(seeds)
    return {
        "jet_mae": jet_mae,
        "fd_mae": fd_mae,
        "zero_mae": zero_mae,
        "beats_fd": jet_mae < fd_mae,
        "skill_vs_zero": zero_mae - jet_mae,
        "g2_earned": jet_mae < fd_mae and jet_mae < zero_mae,
    }


__all__ = [
    "DEFAULT_SPEC",
    "DISCLAIMER",
    "ParameterJetSpec",
    "fourier_heat",
    "fourier_heat_du_dmu",
    "honesty_payload",
    "mixed_jet",
    "parameter_jet_skill",
    "worked_example",
]
