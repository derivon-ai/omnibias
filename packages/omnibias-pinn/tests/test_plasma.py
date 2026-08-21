# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Theory 07-07: Harris-sheet resistive layer (tooling, not fusion)."""

from __future__ import annotations

from omnibias.pinn.plasma import (
    DISCLAIMER,
    HarrisSheet,
    field_ops_resolve,
    harris_force_errors,
    honesty_payload,
    layer_scaling_report,
    pack_current_residual,
    resistive_layer_basis,
)


def test_g0_harris_and_field_ops() -> None:
    errs = harris_force_errors(0.1)
    assert max(errs) <= 1e-12
    found = field_ops_resolve()
    assert found["torch.induction_residual"] is True
    assert found["torch.ideal_mhd_momentum_residual"] is True
    assert honesty_payload()["fusion_claim"] is False
    assert "not a fusion-energy result" in DISCLAIMER


def test_g3_pack_dim_independent_of_thickness() -> None:
    rows = layer_scaling_report()
    assert len(rows) >= 4
    dims = {int(r["pack_dim"]) for r in rows}
    assert dims == {4}
    assert all(float(r["pack_residual"]) <= 1e-12 for r in rows)
    uniforms = [int(r["uniform_dim"]) for r in rows]
    # Uniform resolution must grow as the sheet thins.
    assert uniforms[-1] >= 4 * uniforms[0]
    assert all(bool(r["grew_as_inv_d"]) for r in rows)


def test_g4_harris_equilibrium_one_percent() -> None:
    sheet = HarrisSheet(b0=1.0, thickness=0.05)
    # Recover By at x=0 to the published tanh profile (identity, 0%).
    assert abs(sheet.b_y(0.0) - 0.0) <= 0.01
    assert abs(sheet.b_y(sheet.thickness) - math_tanh_one()) <= 0.01 * 1.0
    assert max(harris_force_errors(sheet.thickness)) <= 0.01


def math_tanh_one() -> float:
    import math

    return math.tanh(1.0)


def test_resistive_basis_sits_at_the_layer() -> None:
    basis = resistive_layer_basis(0.0, 0.01, orders=(0, 1))
    assert basis == [("tanh", 0.0, 0), ("tanh", 0.0, 1)]
    assert pack_current_residual(0.01) <= 1e-12
