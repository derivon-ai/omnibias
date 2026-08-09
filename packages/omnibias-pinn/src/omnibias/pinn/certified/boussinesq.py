# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Boussinesq certified-evidence helpers (honesty-first, streamfunction form)."""

from __future__ import annotations

from typing import Any

import numpy as np


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
    }


__all__ = ["build_boussinesq_cap_bundle"]
