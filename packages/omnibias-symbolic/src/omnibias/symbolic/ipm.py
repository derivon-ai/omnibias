# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Independent numpy IPM residual twin (streamfunction formulation)."""

from __future__ import annotations

from typing import Any

import numpy as np


def ipm_selfsimilar_residual(
    y1: np.ndarray,
    y2: np.ndarray,
    theta: np.ndarray,
    theta_y1: np.ndarray,
    theta_y2: np.ndarray,
    psi: np.ndarray,
    psi_y1: np.ndarray,
    psi_y2: np.ndarray,
    psi_lap: np.ndarray,
    lam: float,
) -> tuple[np.ndarray, np.ndarray]:
    omega = -theta_y1
    u1 = psi_y2
    u2 = -psi_y1
    adv = u1 * theta_y1 + u2 * theta_y2
    r_theta = (1.0 + lam) * (y1 * theta_y1 + y2 * theta_y2) - lam * theta + adv
    r_psi = psi_lap - omega
    return r_theta, r_psi


def verify_ipm_bundle(bundle: dict[str, Any], *, atol: float = 1e-8) -> dict[str, Any]:
    vin = bundle["validation_inputs"]
    r_th, r_psi = ipm_selfsimilar_residual(
        np.asarray(vin["y1"]),
        np.asarray(vin["y2"]),
        np.asarray(vin["theta"]),
        np.asarray(vin["theta_y1"]),
        np.asarray(vin["theta_y2"]),
        np.asarray(vin["psi"]),
        np.asarray(vin["psi_y1"]),
        np.asarray(vin["psi_y2"]),
        np.asarray(vin["psi_lap"]),
        float(vin["lambda"]),
    )
    d_th = float(np.max(np.abs(r_th - np.asarray(bundle["residual_theta"]))))
    d_psi = float(np.max(np.abs(r_psi - np.asarray(bundle["residual_psi"]))))
    return {
        "agreement_max_abs_diff_theta": d_th,
        "agreement_max_abs_diff_psi": d_psi,
        "residual_samples_match": bool(d_th <= atol and d_psi <= atol),
        "navier_stokes_proof_claim": False,
    }


__all__ = ["ipm_selfsimilar_residual", "verify_ipm_bundle"]
