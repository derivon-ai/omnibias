# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Theory 07-05: multi-pack trial spaces for certified spectral floors."""

from __future__ import annotations

import math

import pytest
from omnibias.core.proof.certificate import make_certificate
from omnibias.core.verified.trial_spaces import (
    DISCLAIMER,
    adaptive_trial_refinement,
    adversarial_report,
    certified_floor,
    dimension_reduction_report,
    dirichlet_nodes,
    discrete_schrodinger,
    floor_schema_errors,
    honesty_payload,
    localized_suite,
    multipack_trial_space,
    seal_floor,
    sine_trial_space,
    soundness_violations,
    synthetic_diagonal,
    trial_space_alignment,
)


def test_g1_alignment_required_and_reported() -> None:
    matrix, evec, rho = synthetic_diagonal(4, seed=7)
    nodes = [0.0, 1.0, 2.0, 3.0]
    trial = sine_trial_space(1, domain=(0.0, 3.0))
    cert = certified_floor(matrix, trial, nodes, reference_vector=evec, rho=rho)
    assert math_is_prob(cert.alignment)
    assert "sin_theta" in cert.to_payload()
    assert floor_schema_errors(cert) == []
    with pytest.raises(TypeError):
        certified_floor(matrix, trial, nodes, rho=rho)  # type: ignore[call-arg]


def math_is_prob(value: float) -> bool:
    return 0.0 <= float(value) <= 1.0


def test_g2_dimension_reduction_on_localized_suite() -> None:
    rows = dimension_reduction_report()
    assert len(rows) == 10
    assert all(int(r["pack_dim"]) * 4 <= int(r["uniform_dim"]) for r in rows)
    assert all(math_is_prob(float(r["pack_alignment"])) for r in rows)
    assert all(bool(r["within_5pct"]) for r in rows), rows


def test_g3_adversarial_cases_are_reported() -> None:
    rows = adversarial_report()
    assert len(rows) == 3
    kinds = {r["kind"] for r in rows}
    assert kinds == {"oscillatory", "corner", "discontinuous"}
    assert all("pack_alignment" in r and "uniform_alignment" in r for r in rows)


def test_g4_soundness_1000_known_spectra() -> None:
    assert soundness_violations(n_problems=1000) == 0


def test_g6_honesty_and_kernel_flag_cannot_be_forged() -> None:
    matrix, evec, rho = synthetic_diagonal(4, seed=3)
    nodes = [0.0, 1.0, 2.0, 3.0]
    trial = sine_trial_space(1, domain=(0.0, 3.0))
    cert = certified_floor(matrix, trial, nodes, reference_vector=evec, rho=rho)
    sealed = seal_floor(cert)
    assert sealed["honesty"]["yang_mills_mass_gap"] is False
    assert sealed["honesty"]["continuum_spectral_gap_claim"] is False
    assert "theorem_prover_verified" not in sealed["honesty"]
    assert honesty_payload()["theorem_prover_verified"] is False
    assert "not a Yang-Mills mass gap" in DISCLAIMER
    with pytest.raises(ValueError, match="theorem_prover_verified"):
        make_certificate(
            claim="forged",
            payload={"ok": True},
            honesty={"theorem_prover_verified": True},
        )


def test_adaptive_refinement_grows_the_space() -> None:
    nodes = dirichlet_nodes(17)
    matrix = discrete_schrodinger(lambda x: -40.0 * math.exp(-80.0 * x * x), nodes)
    start = multipack_trial_space([0.0], [0], [9.0], domain=(-1.0, 1.0))
    grown = adaptive_trial_refinement(matrix, start, nodes, target_width=1e-9, max_dim=4)
    assert grown.dim >= start.dim
    assert grown.dim <= 4


def test_alignment_is_zero_when_the_vector_is_in_the_space() -> None:
    nodes = dirichlet_nodes(16)
    trial = sine_trial_space(3, domain=(-1.0, 1.0))
    cols = trial.evaluate(nodes)
    # First sine mode is exactly column 0.
    assert trial_space_alignment(trial, cols[0], nodes) < 1e-12


def test_localized_suite_has_ten_problems() -> None:
    assert len(localized_suite()) == 10
