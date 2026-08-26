# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Theory 09-25: world-model-as-jet."""

from __future__ import annotations

from omnibias.core.jet_world import (
    DISCLAIMER,
    honesty_payload,
    jet_world_skill,
    source_imports_no_backend,
    worked_example,
)


def test_g1_taylor() -> None:
    ex = worked_example()
    assert ex["abs_err"] < 1e-12
    assert ex["pred"] == ex["named"]


def test_g2_skill() -> None:
    report = jet_world_skill()
    assert report["g2_earned"] is True
    assert report["contains"] is True
    assert report["refused_wide"] is True


def test_g3_honesty() -> None:
    payload = honesty_payload()
    assert payload["navier_stokes_proof_claim"] is False
    assert payload["continuum_claim"] is False
    assert payload["theorem_prover_verified"] is False
    assert "not NS" in DISCLAIMER


def test_g4_no_torch_jax() -> None:
    assert source_imports_no_backend() is True
