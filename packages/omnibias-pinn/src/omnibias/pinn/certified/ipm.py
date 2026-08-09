# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""IPM certified-evidence helpers (honesty-first, streamfunction form)."""

from __future__ import annotations

from typing import Any

import numpy as np


def build_ipm_cap_bundle(discovery: dict[str, Any]) -> dict[str, Any]:
    vin = discovery.get("validation_inputs")
    if vin is None:
        raise ValueError("discovery must carry validation_inputs for streamfunction CAP")
    return {
        "schema_version": "ipm-cap-2",
        "lambda": float(discovery["lam"]),
        "domain": {"type": "compactified_halfplane_smoke"},
        "residual_theta": np.asarray(discovery["residual_theta"]).tolist(),
        "residual_psi": np.asarray(discovery["residual_psi"]).tolist(),
        "validation_inputs": vin,
        "honesty": {
            "unproven_claim": False,
            "navier_stokes_proof_claim": False,
            "exact_solution_claim": False,
            "formulation": "streamfunction_poisson_residual",
        },
    }


__all__ = ["build_ipm_cap_bundle"]
