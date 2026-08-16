# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Three-plaquette KS Hamiltonian: census, gap, kind, seal, replay."""

from __future__ import annotations

import numpy as np
import pytest
from omnibias.core.proof import Conjecture
from omnibias.geometry.gauge.proofmachine import build_gauge_machine
from omnibias.geometry.gauge.transfer.certificates import (
    THREE_PLAQUETTE_GAP_KIND,
    hamiltonian_gap_schema_errors,
    replay_hamiltonian_gap,
    seal_hamiltonian_gap_certificate,
)
from omnibias.geometry.gauge.transfer.hamiltonian import (
    COUPLING_LOCK,
    LEHMANN_HOLONOMY_METHOD,
    LEHMANN_STANDARD_METHOD,
    THREE_PLAQUETTE_ELECTRIC,
    candidate_gap,
    certified_hamiltonian_gap,
    legal_chain,
    plaquette_holonomy_trial_space,
    su2_three_plaquette_hamiltonian,
    three_plaquette_basis,
)


def _midpoint(hamiltonian: object) -> np.ndarray:
    return np.array(
        [[0.5 * (c.lo + c.hi) for c in row] for row in hamiltonian.entries]  # type: ignore[attr-defined]
    )


def test_edge_census_is_locked() -> None:
    assert THREE_PLAQUETTE_ELECTRIC == (3, 2, 3, 1, 1)
    assert sum(THREE_PLAQUETTE_ELECTRIC) == 10


def test_every_basis_vector_obeys_both_triangles() -> None:
    hamiltonian = su2_three_plaquette_hamiltonian(COUPLING_LOCK, j_max=1)
    assert hamiltonian.dimension == 41
    assert hamiltonian.basis == three_plaquette_basis(1)
    for t1, t2, t3, s12, s23 in hamiltonian.basis:
        assert legal_chain(t1, t2, t3, s12, s23, two_j_max=2)


def test_g1_holonomy_lehmann_is_at_least_standard() -> None:
    hamiltonian = su2_three_plaquette_hamiltonian(COUPLING_LOCK, j_max=1)
    generic_official = certified_hamiltonian_gap(hamiltonian)
    trial = plaquette_holonomy_trial_space(hamiltonian)
    holonomy = certified_hamiltonian_gap(hamiltonian, trial=trial)
    assert holonomy.spectral_gap_lower > 0.0
    assert (
        holonomy.spectral_gap_lower + 1e-12 >= generic_official.spectral_gap_lower
    )
    generic = candidate_gap(holonomy, LEHMANN_STANDARD_METHOD)
    holonomy_lehmann = candidate_gap(holonomy, LEHMANN_HOLONOMY_METHOD)
    assert holonomy_lehmann + 1e-12 >= generic


def test_locked_coupling_certifies_a_positive_gap() -> None:
    hamiltonian = su2_three_plaquette_hamiltonian(COUPLING_LOCK, j_max=1)
    result = certified_hamiltonian_gap(hamiltonian)
    assert result.certified is True
    assert result.spectral_gap_lower > 0.0
    assert 0.0 <= result.subdominant_ratio_upper < 1.0


def test_g2_certified_gap_never_exceeds_midpoint_eigengap() -> None:
    hamiltonian = su2_three_plaquette_hamiltonian(COUPLING_LOCK, j_max=1)
    values = np.sort(np.linalg.eigvalsh(_midpoint(hamiltonian)))
    numerical = float(values[1] - values[0])
    result = certified_hamiltonian_gap(hamiltonian)
    assert result.spectral_gap_lower <= numerical + 1e-9


def test_matrix_is_symmetric() -> None:
    hamiltonian = su2_three_plaquette_hamiltonian(COUPLING_LOCK, j_max=1)
    mid = _midpoint(hamiltonian)
    np.testing.assert_allclose(mid, mid.T, atol=1e-12)


def test_g5_honesty_and_three_plaquette_kind() -> None:
    hamiltonian = su2_three_plaquette_hamiltonian(COUPLING_LOCK, j_max=1)
    result = certified_hamiltonian_gap(hamiltonian)
    sealed = dict(seal_hamiltonian_gap_certificate(result, hamiltonian))
    assert sealed["observable"] == THREE_PLAQUETTE_GAP_KIND
    assert sealed["continuum_claim"] is False
    assert sealed["honesty"]["yang_mills_claim"] is False
    assert hamiltonian_gap_schema_errors(sealed) == []
    forged = dict(sealed)
    forged["continuum_claim"] = True
    forged["honesty"] = dict(sealed["honesty"], yang_mills_claim=True)
    errors = hamiltonian_gap_schema_errors(forged)
    assert any("continuum_claim" in err for err in errors)
    assert any("yang_mills_claim" in err for err in errors)


def test_replay_and_machine_prove_the_locked_coupling() -> None:
    hamiltonian = su2_three_plaquette_hamiltonian(COUPLING_LOCK, j_max=1)
    result = certified_hamiltonian_gap(hamiltonian)
    cert = seal_hamiltonian_gap_certificate(result, hamiltonian)
    assert replay_hamiltonian_gap(cert) is True
    machine = build_gauge_machine()
    verdict = machine.evaluate(
        Conjecture(
            "three-plaq gap",
            THREE_PLAQUETTE_GAP_KIND,
            {"coupling": COUPLING_LOCK, "j_max": 1},
        )
    )
    assert verdict.status == "PROVED"
    assert verdict.schema_ok is True
    assert verdict.replay_ok is True
    assert verdict.honesty_ok is True


def test_unknown_magnetic_is_rejected() -> None:
    with pytest.raises(ValueError, match="magnetic"):
        su2_three_plaquette_hamiltonian(COUPLING_LOCK, magnetic="racah")  # type: ignore[arg-type]


def test_j_max_2_basis_is_strictly_larger_and_legal() -> None:
    small = three_plaquette_basis(1)
    large = three_plaquette_basis(2)
    assert len(large) > len(small)
    for t1, t2, t3, s12, s23 in large:
        assert legal_chain(t1, t2, t3, s12, s23, two_j_max=4)
