# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""IPM residual radii construction + banded toy CAP."""

from __future__ import annotations

from omnibias.pinn.certified.ipm import (
    build_ipm_cap_bundle,
    build_ipm_radii_construction,
    export_ipm_toy_cap_replay,
    ipm_banded_toy_radii,
)


def test_legacy_json_bundle_still_packs_residuals() -> None:
    bundle = build_ipm_cap_bundle(
        {
            "lam": 1.25,
            "residual_theta": [0.1, -0.2],
            "residual_psi": [0.0, 0.05],
            "validation_inputs": {"grid": "smoke"},
        }
    )
    assert bundle["honesty"]["navier_stokes_proof_claim"] is False
    assert bundle["schema_version"] == "ipm-cap-2"


def test_packed_residual_radii_refuses_large_residual() -> None:
    out = build_ipm_radii_construction(
        {
            "lam": 1.0,
            "residual_theta": [0.5],
            "residual_psi": [0.5],
            "validation_inputs": {"grid": "smoke"},
        },
        tail_bound=0.1,
        inverse_bound=2.0,
        z2=1.0,
    )
    assert out["full_ipm_proved"] is False
    assert out["honesty"]["navier_stokes_proof_claim"] is False


def test_banded_toy_cap_proves_and_exports_replay() -> None:
    result = ipm_banded_toy_radii()
    assert result["named_subcase"] == "banded_quadratic_selfsimilar_toy"
    assert result["full_ipm_proved"] is False
    assert result["honesty"]["navier_stokes_proof_claim"] is False
    assert result["proved"]
    trace = export_ipm_toy_cap_replay(result)
    assert len(trace.steps) >= 2
    assert trace.conclusions == (0, 1)
