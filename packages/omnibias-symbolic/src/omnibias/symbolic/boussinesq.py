# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Independent numpy Boussinesq residual twin (streamfunction formulation)."""

from __future__ import annotations

from typing import Any

import numpy as np


def infer_lambda_from_streamfunction_u1_y1(u1_y1_at_origin: float) -> float:
    return -3.0 - 2.0 * float(u1_y1_at_origin)


def boussinesq_selfsimilar_residual(
    y1: np.ndarray,
    y2: np.ndarray,
    omega: np.ndarray,
    omega_y1: np.ndarray,
    omega_y2: np.ndarray,
    theta: np.ndarray,
    theta_y1: np.ndarray,
    theta_y2: np.ndarray,
    psi: np.ndarray,
    psi_y1: np.ndarray,
    psi_y2: np.ndarray,
    psi_lap: np.ndarray,
    lam: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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


def verify_boussinesq_bundle(bundle: dict[str, Any], *, atol: float = 1e-8) -> dict[str, Any]:
    vin = bundle["validation_inputs"]
    ro, rt, rp = boussinesq_selfsimilar_residual(
        np.asarray(vin["y1"]),
        np.asarray(vin["y2"]),
        np.asarray(vin["omega"]),
        np.asarray(vin["omega_y1"]),
        np.asarray(vin["omega_y2"]),
        np.asarray(vin["theta"]),
        np.asarray(vin["theta_y1"]),
        np.asarray(vin["theta_y2"]),
        np.asarray(vin["psi"]),
        np.asarray(vin["psi_y1"]),
        np.asarray(vin["psi_y2"]),
        np.asarray(vin["psi_lap"]),
        float(vin["lambda"]),
    )
    d_o = float(np.max(np.abs(ro - np.asarray(bundle["residual_omega"]))))
    d_t = float(np.max(np.abs(rt - np.asarray(bundle["residual_theta"]))))
    d_p = float(np.max(np.abs(rp - np.asarray(bundle["residual_psi"]))))
    return {
        "agreement_max_abs_diff_omega": d_o,
        "agreement_max_abs_diff_theta": d_t,
        "agreement_max_abs_diff_psi": d_p,
        "residual_samples_match": bool(d_o <= atol and d_t <= atol and d_p <= atol),
        "navier_stokes_proof_claim": False,
    }


__all__ = [
    "boussinesq_selfsimilar_residual",
    "infer_lambda_from_streamfunction_u1_y1",
    "verify_boussinesq_bundle",
]
