# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Viscous perturbation enclosure — deferred NS-window track.

Takes a self-similar Euler/Boussinesq-style profile and forms a compact
self-similar time-window residual bound for a Navier-Stokes viscous
perturbation. ``PROVED`` language applies only to the **compact window**
finite obligation — never global NS regularity.

For the CCF fractional-dissipation threshold addressing Kiselev Open Problem 1
(``alpha_crit >= 1/(1+lambda_hi)``), use
:mod:`omnibias.pinn.certified.dissipation_threshold` instead. This module stays
as the algebraic NS-window track and is intentionally not the CCF dissipation
certificate.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def viscous_perturbation_enclosure(
    *,
    inviscid_residual_sup: float,
    viscosity: float,
    enstrophy_bound: float,
    window_length: float,
    tol: float = 1e-2,
) -> dict[str, Any]:
    """Crude a-priori viscous error ball on a compact window.

    Model: ``||R_NS|| <= ||R_Euler|| + nu * C * enstrophy`` on ``[0, T]``.
    Closes when the upper bound is ``<= tol``.
    """
    nu = float(viscosity)
    if nu < 0.0:
        raise ValueError("viscosity must be non-negative")
    if window_length <= 0.0:
        raise ValueError("window_length must be positive")
    euler = float(abs(inviscid_residual_sup))
    viscous_term = nu * float(abs(enstrophy_bound)) * float(window_length)
    upper = euler + viscous_term
    closed = bool(upper <= float(tol))
    return {
        "schema_version": "viscous-perturbation-enclosure-1",
        "inviscid_residual_sup": euler,
        "viscosity": nu,
        "enstrophy_bound": float(enstrophy_bound),
        "window_length": float(window_length),
        "viscous_term_bound": viscous_term,
        "ns_residual_upper_bound": upper,
        "tol": float(tol),
        "enclosure_closed": closed,
        "honesty": {
            "unproven_claim": False,
            "navier_stokes_proof_claim": False,
            "continuum_navier_stokes_claim": False,
            "scope": "compact_self_similar_time_window_only",
            "notes": (
                "Finite compact-window a-priori bound; not a global regularity "
                "or blow-up theorem for 3D Navier-Stokes."
            ),
        },
    }


def verify_viscous_perturbation_enclosure(cert: dict[str, Any]) -> dict[str, Any]:
    """Independent recomputation of the upper bound."""
    upper = (
        float(cert["inviscid_residual_sup"])
        + float(cert["viscosity"])
        * float(cert["enstrophy_bound"])
        * float(cert["window_length"])
    )
    match = abs(upper - float(cert["ns_residual_upper_bound"])) <= 1e-12
    return {
        "recomputed_upper": upper,
        "replay_match": bool(match),
        "enclosure_closed": bool(upper <= float(cert["tol"])),
        "navier_stokes_proof_claim": False,
    }


__all__ = [
    "verify_viscous_perturbation_enclosure",
    "viscous_perturbation_enclosure",
]
