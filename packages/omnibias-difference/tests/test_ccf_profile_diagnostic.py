# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Jet-Padé CCF profile diagnostic: poles only, never Dirichlet, never stretch."""

from __future__ import annotations

from pathlib import Path

from omnibias.difference._core.ccf_profile import (
    diagnose_ccf_profile,
    geometric_profile_jet,
    honesty_payload,
)


def test_geometric_snapshot_finds_the_pole() -> None:
    coeffs = geometric_profile_jet(radius=2.0, order=12)
    report = diagnose_ccf_profile(coeffs, snapshot="geometric_radius_2")
    assert report.estimate.failed is False
    assert report.estimate.location is not None
    assert abs(report.estimate.location.real - 2.0) < 0.25
    assert report.honesty["navier_stokes_proof_claim"] is False
    assert report.honesty["residual_substitute"] is False
    assert report.honesty["dirichlet_consumed"] is False


def test_module_does_not_import_dirichlet() -> None:
    path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "omnibias"
        / "difference"
        / "_core"
        / "ccf_profile.py"
    )
    text = path.read_text(encoding="utf-8")
    assert "omnibias.core.verified.dirichlet" not in text
    assert "zeta_euler_maclaurin" not in text
    assert "from omnibias.core.verified.dirichlet" not in text


def test_honesty_keys() -> None:
    honesty = honesty_payload()
    assert honesty["whole_line_certified"] is False
    assert honesty["stretch_cleared"] is False
    assert honesty["continuum_claim"] is False
