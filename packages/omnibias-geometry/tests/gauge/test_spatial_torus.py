# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Finite 2x2 spatial-torus transfer: gap, RP, honesty."""

from __future__ import annotations

import numpy as np
from omnibias.core.proof import Conjecture
from omnibias.geometry.gauge.proofmachine import build_gauge_machine
from omnibias.geometry.gauge.transfer.certificates import (
    TORUS_RP_KIND,
    TRANSFER_GAP_KIND,
    replay_strip_rp,
    replay_transfer_matrix_gap,
    seal_strip_rp_certificate,
    seal_transfer_gap_certificate,
    strip_rp_schema_errors,
    transfer_gap_schema_errors,
)
from omnibias.geometry.gauge.transfer.gap import certified_transfer_matrix_gap
from omnibias.geometry.gauge.transfer.matrices import rebuild
from omnibias.geometry.gauge.transfer.strip import (
    STRIP_COUPLING_LOCK,
    certified_strip_cluster_tail,
    certified_strip_reflection_positivity,
    su2_spatial_torus_transfer,
)


def test_torus_is_16_dimensional_and_symmetric() -> None:
    transfer = su2_spatial_torus_transfer(STRIP_COUPLING_LOCK, n_x=2, n_y=2, n_angles=2)
    assert transfer.dimension == 16
    assert transfer.model == "su2_spatial_torus"
    assert transfer.exact_eigenvalues is None
    mid = np.array([[0.5 * (c.lo + c.hi) for c in row] for row in transfer.entries])
    np.testing.assert_allclose(mid, mid.T, atol=1e-12)


def test_torus_gap_is_certified() -> None:
    transfer = su2_spatial_torus_transfer(STRIP_COUPLING_LOCK, n_x=2, n_y=2, n_angles=2)
    result = certified_transfer_matrix_gap(transfer)
    assert result.certified is True
    assert result.spectral_gap_lower > 0.0
    values = np.sort(
        np.linalg.eigvalsh(
            np.array([[0.5 * (c.lo + c.hi) for c in row] for row in transfer.entries])
        )
    )[::-1]
    numerical = float(-np.log(abs(values[1]) / values[0]))
    assert result.spectral_gap_lower <= numerical + 1e-9


def test_torus_reflection_positivity_forms_are_non_negative() -> None:
    transfer = su2_spatial_torus_transfer(STRIP_COUPLING_LOCK, n_x=2, n_y=2, n_angles=2)
    result = certified_strip_reflection_positivity(transfer)
    assert result.certified is True
    assert all(form.lo >= 0.0 for form in result.forms)


def test_torus_cluster_tail_contains_a_numerical_sample() -> None:
    transfer = su2_spatial_torus_transfer(STRIP_COUPLING_LOCK, n_x=2, n_y=2, n_angles=2)
    result = certified_strip_cluster_tail(transfer, n_keep=2)
    assert result.certified is True
    assert result.tail.contains(result.sample)


def test_torus_rp_seal_replay_and_honesty() -> None:
    transfer = su2_spatial_torus_transfer(STRIP_COUPLING_LOCK, n_x=2, n_y=2, n_angles=2)
    result = certified_strip_reflection_positivity(transfer)
    sealed = dict(seal_strip_rp_certificate(result, transfer))
    assert sealed["observable"] == TORUS_RP_KIND
    assert sealed["continuum_claim"] is False
    assert sealed["honesty"]["yang_mills_claim"] is False
    assert strip_rp_schema_errors(sealed) == []
    assert replay_strip_rp(sealed) is True
    forged = dict(sealed)
    forged["continuum_claim"] = True
    forged["honesty"] = dict(sealed["honesty"], yang_mills_claim=True)
    errors = strip_rp_schema_errors(forged)
    assert any("continuum_claim" in err for err in errors)
    assert any("yang_mills_claim" in err for err in errors)


def test_torus_rebuild_and_machine() -> None:
    transfer = su2_spatial_torus_transfer(STRIP_COUPLING_LOCK, n_x=2, n_y=2, n_angles=2)
    again = rebuild(transfer.parameters)
    assert again.dimension == transfer.dimension
    result = certified_transfer_matrix_gap(transfer)
    cert = seal_transfer_gap_certificate(result, transfer)
    assert transfer_gap_schema_errors(cert) == []
    assert replay_transfer_matrix_gap(cert) is True
    machine = build_gauge_machine()
    verdict = machine.evaluate(
        Conjecture(
            "torus gap",
            TRANSFER_GAP_KIND,
            {"parameters": dict(transfer.parameters)},
        )
    )
    assert verdict.status == "PROVED"
    rp = machine.evaluate(
        Conjecture(
            "torus rp",
            TORUS_RP_KIND,
            {"coupling": STRIP_COUPLING_LOCK, "n_x": 2, "n_y": 2, "n_angles": 2},
        )
    )
    assert rp.status == "PROVED"
    assert rp.honesty_ok is True
