# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Theory 07-14: NS-core axis-regular profile search."""

from __future__ import annotations

from fractions import Fraction

import omnibias.pinn.certified.machine  # noqa: F401
from omnibias.core.proof.catalog import catalog_entry
from omnibias.core.proof.discovery import run_discovery
from omnibias.pinn.certified.anisotropic import (
    NS_CORE_OPPOSITE,
    NS_CORE_ORIGIN,
    NS_CORE_SECOND_WITNESS,
    NSCoreProfileFamily,
    check_ns_core_candidate,
    honesty_payload,
    locked_axis_regular_profile,
    ns_core_statement,
    profile_from_coeffs,
    profile_similarity_residual,
)
from omnibias.pinn.jax.discovery.ns_core import run_ns_core_discovery, run_ns_core_search
from omnibias.pinn.jax.discovery.pipeline import NSCoreAdapter, run_singularity_pipeline


def test_g1_locked_origin_still_discharges() -> None:
    origin = check_ns_core_candidate(NS_CORE_ORIGIN)
    assert origin is not None and origin.ok
    locked = locked_axis_regular_profile()
    assert profile_similarity_residual(locked) == 0
    result = run_discovery(ns_core_statement(), NSCoreProfileFamily(), budget=8)
    assert result.status == "PROVED"
    assert result.candidate == NS_CORE_ORIGIN
    assert result.search_incomplete is False


def test_g2_cone_opposite_is_named_blocked() -> None:
    checked = check_ns_core_candidate(NS_CORE_OPPOSITE)
    assert checked is not None
    assert checked.ok is False
    assert checked.payload["cone_status"] == "BLOCKED"
    assert checked.payload["cone_reason"] == "opposite_cone"
    assert checked.payload["profile_residual"] == "0"


def test_g3_budget_zero_is_search_incomplete() -> None:
    result = run_ns_core_discovery(budget=0)
    assert result.status == "BLOCKED"
    assert result.search_incomplete is True
    assert result.detail == "search_incomplete"


def test_g4_non_origin_witness() -> None:
    second = check_ns_core_candidate(NS_CORE_SECOND_WITNESS)
    assert second is not None and second.ok
    assert second.payload["is_origin"] is False
    collected = run_ns_core_discovery(budget=27, collect=True)
    assert NS_CORE_SECOND_WITNESS in collected.solutions
    payload = run_ns_core_search(budget=27)
    assert payload["origin_only"] is False
    assert ["1", "3", "-1"] in payload["non_origin_witnesses"]


def test_g5_pipeline_adapter_and_honesty() -> None:
    flags = honesty_payload()
    assert flags["navier_stokes_proof_claim"] is False
    assert flags["forced_blowup_reproof_claim"] is False
    out = run_singularity_pipeline(NSCoreAdapter(budget=8))
    assert out.adapter == "ns_core"
    assert out.discovery["status"] == "PROVED"
    assert out.certificate["honesty"]["navier_stokes_proof_claim"] is False
    assert out.certificate["honesty"]["forced_blowup_reproof_claim"] is False
    entry = catalog_entry("ns_core_profile_search")
    assert entry is not None
    assert entry.mode == "exact_search"
    assert entry.parent_status == "already_true"


def test_quadratic_pi_matches_f_squared() -> None:
    profile = profile_from_coeffs(1, 3, -1)
    X = Fraction(1)
    assert profile.Pi_X(X) == profile.F(X) * profile.F(X)
    assert profile.V0_poly(X, Fraction(1, 2)) + X * profile.U_poly_X(
        X, Fraction(1, 2)
    ) == 0


def test_family_cardinality() -> None:
    family = NSCoreProfileFamily()
    assert family.cardinality() == 27
    assert family.origin() == NS_CORE_ORIGIN
