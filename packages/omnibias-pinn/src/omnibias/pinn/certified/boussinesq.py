# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Boussinesq certified-evidence helpers (honesty-first, streamfunction form)."""

from __future__ import annotations

from typing import Any

import numpy as np
from omnibias.core.verified.interval import Interval

BOUSSINESQ_REMAINDER_LEFTOVER = {
    "leftover_id": 54,
    "leftover_recorded": True,
    "named_closing_object": "smoke_grid_residual_hull",
    "note": (
        "full Boussinesq streamfunction-Poisson remainder stays external; "
        "lambda_n is an empirical hypothesis, not a theorem"
    ),
}


def _boussinesq_residual_numpy(
    y1: float,
    y2: float,
    omega: float,
    omega_y1: float,
    omega_y2: float,
    theta: float,
    theta_y1: float,
    theta_y2: float,
    psi_lap: float,
    psi_y1: float,
    psi_y2: float,
    lam: float,
) -> tuple[float, float, float]:
    u1 = psi_y2
    u2 = -psi_y1
    r_omega = (
        u1 * omega_y1
        + u2 * omega_y2
        - (1.0 + lam) * (y1 * omega_y1 + y2 * omega_y2)
        + omega
        - theta_y1
    )
    r_theta = (
        u1 * theta_y1
        + u2 * theta_y2
        - (1.0 + lam) * (y1 * theta_y1 + y2 * theta_y2)
        + (1.0 - lam) * theta
    )
    r_psi = psi_lap - omega
    return r_omega, r_theta, r_psi


def enclose_boussinesq_grid_residual(discovery: dict[str, Any]) -> dict[str, Any]:
    """Named interval hull of the smoke-grid residual plus a truth sample."""
    packed = np.concatenate(
        [
            np.asarray(discovery["residual_omega"], dtype=np.float64).ravel(),
            np.asarray(discovery["residual_theta"], dtype=np.float64).ravel(),
            np.asarray(discovery["residual_psi"], dtype=np.float64).ravel(),
        ]
    )
    hull = Interval.hull(*(float(x) for x in packed))
    vin = discovery.get("validation_inputs") or {}
    truth_ok = False
    if vin:
        ro, rt, rp = _boussinesq_residual_numpy(
            float(np.asarray(vin["y1"]).ravel()[0]),
            float(np.asarray(vin["y2"]).ravel()[0]),
            float(np.asarray(vin["omega"]).ravel()[0]),
            float(np.asarray(vin["omega_y1"]).ravel()[0]),
            float(np.asarray(vin["omega_y2"]).ravel()[0]),
            float(np.asarray(vin["theta"]).ravel()[0]),
            float(np.asarray(vin["theta_y1"]).ravel()[0]),
            float(np.asarray(vin["theta_y2"]).ravel()[0]),
            float(np.asarray(vin["psi_lap"]).ravel()[0]),
            float(np.asarray(vin["psi_y1"]).ravel()[0]),
            float(np.asarray(vin["psi_y2"]).ravel()[0]),
            float(vin["lambda"]),
        )
        truth_ok = hull.contains(ro) and hull.contains(rt) and hull.contains(rp)
    return {
        "lo": hull.lo,
        "hi": hull.hi,
        "contains_truth_sample": bool(truth_ok),
        "full_boussinesq_proved": False,
        "leftover": dict(BOUSSINESQ_REMAINDER_LEFTOVER),
        "honesty": {
            "navier_stokes_proof_claim": False,
            "lambda_n_hypothesis_is_theorem": False,
            "full_boussinesq_proved": False,
        },
    }


def build_boussinesq_cap_bundle(discovery: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "boussinesq-cap-2",
        "lambda": float(discovery["lam"]),
        "lam_inferred": discovery.get("lam_inferred"),
        "domain": {"type": "halfplane_with_boundary_smoke"},
        "residual_omega": np.asarray(discovery["residual_omega"]).tolist(),
        "residual_theta": np.asarray(discovery["residual_theta"]).tolist(),
        "residual_psi": np.asarray(discovery["residual_psi"]).tolist(),
        "validation_inputs": discovery.get("validation_inputs", {}),
        "honesty": {
            "unproven_claim": False,
            "navier_stokes_proof_claim": False,
            "lambda_n_hypothesis_is_theorem": False,
            "formulation": "streamfunction_poisson_residual",
        },
        "lambda_n_hypothesis": discovery.get("lambda_n_hypothesis"),
        "full_boussinesq_proved": False,
        "remainder": enclose_boussinesq_grid_residual(discovery),
    }


__all__ = [
    "BOUSSINESQ_REMAINDER_LEFTOVER",
    "build_boussinesq_cap_bundle",
    "enclose_boussinesq_grid_residual",
]
