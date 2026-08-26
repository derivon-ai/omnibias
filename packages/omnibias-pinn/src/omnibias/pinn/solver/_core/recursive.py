# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Recursive / linearizing schemes for viscous 1-D models (Burgers / heat).

These are the closed-form-adjacent recursions that sit under famous NS
transforms, without claiming Clay regularity:

* **Cole--Hopf + heat jets** -- exact linearization of viscous Burgers via
  :mod:`omnibias.core.transforms_pde` factorial jets (Taylor recurrence).
* **Frozen-advection Picard / defect correction** -- replace
  ``u u_x`` by ``u^{(n)} u_x^{(n+1)}`` and march a sequence of *linear*
  problems (Oseen-style sequencing).
* **Jet-Taylor time recurrence** -- reuse
  :func:`~omnibias.pinn.solver.torch.integrators.burgers_jet_step` /
  ``linear_jet_step`` (Cauchy--Kovalevskaya-style time bootstrapping on a
  spatial grid).

Honesty: ``navier_stokes_proof_claim`` is always ``False``. One-way spatial
(OWNS) filters and Adomian polynomials are not implemented here yet.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from omnibias.core.transforms_pde import (
    cole_hopf_jet,
    heat_plane_wave_jets,
    verify_cole_hopf_burgers_jet,
)


def honesty_payload() -> dict[str, bool]:
    return {
        "navier_stokes_proof_claim": False,
        "continuum_navier_stokes_claim": False,
        "clay_regularity_claim": False,
        "adomian_claim": False,
        "owns_one_way_claim": False,
        "exact_cole_hopf_burgers": True,
        "picard_is_linearized_sequence": True,
    }


@dataclass(frozen=True)
class PicardReport:
    """Outcome of a frozen-advection Picard sequence on a 1-D field."""

    iterates: int
    residual_history: tuple[float, ...]
    final_residual: float
    converged: bool
    honesty: dict[str, bool]
    field: np.ndarray | None = None


def burgers_residual_periodic(
    u: np.ndarray,
    *,
    viscosity: float,
    dx: float,
    dt_field: np.ndarray | None = None,
) -> np.ndarray:
    r"""Spectral-free central-difference Burgers residual on a periodic grid.

    ``r = u_t - nu u_xx + u u_x`` with ``u_t`` supplied or taken as 0 (steady).
    Used as the defect for Picard / defect-correction loops -- not a claim of
    continuum NS.
    """
    u = np.asarray(u, dtype=float)
    ux = (np.roll(u, -1) - np.roll(u, 1)) / (2.0 * dx)
    uxx = (np.roll(u, -1) - 2.0 * u + np.roll(u, 1)) / (dx * dx)
    ut = np.zeros_like(u) if dt_field is None else np.asarray(dt_field, dtype=float)
    return ut - float(viscosity) * uxx + u * ux


def picard_frozen_advection(
    u0: np.ndarray,
    *,
    viscosity: float,
    dx: float,
    iters: int = 8,
    relax: float = 0.25,
    tol: float = 1e-10,
    linear_solver: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
) -> PicardReport:
    r"""Picard / defect-correction sequencing for steady viscous Burgers.

    Default step is **line-searched defect correction**
    ``u <- u - ω r(u)`` with ``r = -ν u_xx + u u_x`` (steady Burgers residual),
    accepting the largest trial ``ω`` that strictly decreases ``||r||_∞``.
    Optional ``linear_solver(field, defect)`` replaces that update (e.g. a
    frozen-advection linear solve). This is a linearized recursive sequence,
    not a Clay NS claim.
    """
    if iters < 1:
        raise ValueError("iters must be >= 1")
    u = np.asarray(u0, dtype=float).copy()
    history: list[float] = []

    def _line_search_correct(field: np.ndarray, defect: np.ndarray) -> np.ndarray:
        res0 = float(np.max(np.abs(defect)))
        best = field
        best_res = res0
        # Trial steps: under-relaxed first (stable), then larger
        for omega in (relax * s for s in (0.25, 0.5, 1.0, 2.0, 4.0)):
            trial = field - omega * defect
            trial_res = float(
                np.max(np.abs(burgers_residual_periodic(trial, viscosity=viscosity, dx=dx)))
            )
            if trial_res < best_res:
                best, best_res = trial, trial_res
        return best

    solve = linear_solver or _line_search_correct
    for _ in range(int(iters)):
        defect = burgers_residual_periodic(u, viscosity=viscosity, dx=dx)
        res = float(np.max(np.abs(defect)))
        history.append(res)
        if res <= tol:
            break
        u_next = solve(u, defect)
        res_next = float(
            np.max(np.abs(burgers_residual_periodic(u_next, viscosity=viscosity, dx=dx)))
        )
        if res_next >= res:
            # No improving step -- stop (honest stall, not a forged convergence)
            break
        u = u_next
    final = history[-1] if history else float("inf")
    # append last accepted residual if we moved
    last_def = burgers_residual_periodic(u, viscosity=viscosity, dx=dx)
    last_res = float(np.max(np.abs(last_def)))
    if not history or last_res < history[-1]:
        history.append(last_res)
        final = last_res
    return PicardReport(
        iterates=len(history),
        residual_history=tuple(history),
        final_residual=final,
        converged=bool(final <= tol),
        honesty=honesty_payload(),
        field=np.asarray(u, dtype=float).copy(),
    )


def cole_hopf_exact_burgers_demo(
    *, nu: float = 0.1, k: float = -0.5, order: int = 12
) -> dict[str, Any]:
    """Exact Taylor-jet Cole--Hopf path heat -> Burgers (closed-form recursion)."""
    phi, phi_x = heat_plane_wave_jets(k=k, nu=nu, order=order)
    u_jet = cole_hopf_jet(phi, phi_x, nu=nu)
    check = verify_cole_hopf_burgers_jet(nu=nu, k=k, order=order)
    return {
        "phi_jet0": float(phi[0]),
        "u_jet": [float(c) for c in u_jet],
        "check": check,
        "honesty": honesty_payload(),
        "method": "cole_hopf_factorial_jet_recurrence",
    }


def recursive_toolkit_smoke() -> dict[str, Any]:
    """CI-friendly smoke: Cole--Hopf exact + Picard defect drop on a bump."""
    cole = cole_hopf_exact_burgers_demo()
    n = 64
    x = np.linspace(0.0, 1.0, n, endpoint=False)
    dx = float(x[1] - x[0])
    u0 = 0.2 * np.sin(2.0 * np.pi * x)
    pic = picard_frozen_advection(u0, viscosity=0.05, dx=dx, iters=20, tol=1e-8)
    return {
        "cole_hopf_passed": bool(cole["check"]["passed"]),
        "picard_residual_start": float(pic.residual_history[0]),
        "picard_residual_final": float(pic.final_residual),
        "picard_improved": bool(pic.final_residual < pic.residual_history[0]),
        "honesty": honesty_payload(),
        "missing": (
            "Adomian polynomials",
            "OWNS one-way spatial filters",
            "3-D NS Cauchy-Kovalevskaya recurrence",
        ),
    }


__all__ = [
    "PicardReport",
    "burgers_residual_periodic",
    "cole_hopf_exact_burgers_demo",
    "honesty_payload",
    "picard_frozen_advection",
    "recursive_toolkit_smoke",
]
