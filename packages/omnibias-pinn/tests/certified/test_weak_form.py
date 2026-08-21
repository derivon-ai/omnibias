# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Theory 07-02: weak-form NS-adjacent enclosures, gates G1–G6."""

from __future__ import annotations

from omnibias.pinn.certified.weak_form import (
    DISCLAIMER,
    certified_weak_residual,
    enclosure_covers,
    honesty_payload,
    lohner_horizon,
    manufactured_residual,
    shear_repr_errors,
    weak_form_schema_errors,
    width_decomposition,
)


def test_g1_width_report_required() -> None:
    cert = certified_weak_residual(form="weak")
    report = width_decomposition(cert)
    assert report.dominant in {"repr", "quad", "deriv", "round"}
    assert "width_report" in cert["payload"]
    assert weak_form_schema_errors(cert) == []


def test_g2_weak_quad_at_least_2x() -> None:
    strong = width_decomposition(certified_weak_residual(form="strong"))
    weak = width_decomposition(certified_weak_residual(form="weak"))
    assert strong.dominant == "quad"
    assert weak.dominant == "quad"
    assert strong.w_quad >= 2.0 * weak.w_quad


def test_g3_coverage() -> None:
    assert enclosure_covers(n=1000, form="weak") == 0
    assert enclosure_covers(n=1000, form="strong") == 0


def test_completeness_mean_is_not_enough() -> None:
    # Orthogonal to constants on a full period, but not the zero residual.
    assert abs(manufactured_residual(0.25, 0.25)) > 0.1


def test_g4_shear_pack_independent_of_thickness() -> None:
    ds = (1e-1, 3e-2, 1e-2, 3e-3, 1e-3)
    err = shear_repr_errors(ds)
    assert all(p == 0.0 for p in err["pack"])
    # Smooth error grows at least as 1/d: error * d stays order-one.
    scaled = [s * d for s, d in zip(err["smooth"], ds, strict=True)]
    assert scaled[-1] > 0.5
    assert err["smooth"][-1] / err["smooth"][0] >= (ds[0] / ds[-1]) * 0.5


def test_g5_exact_jacobian_horizon() -> None:
    exact = lohner_horizon(exact_jac=True)
    fd = lohner_horizon(exact_jac=False)
    assert fd >= 1
    assert exact >= 1.5 * fd


def test_g6_honesty() -> None:
    cert = certified_weak_residual()
    honesty = cert["honesty"]
    assert honesty["continuum_navier_stokes_claim"] is False
    assert honesty["unproven_claim"] is False
    assert honesty["three_d_claim"] is False
    assert honesty_payload()["theorem_prover_verified"] is False
    assert "continuum" in DISCLAIMER
    assert weak_form_schema_errors(cert) == []
