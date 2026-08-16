# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Two-plaquette KS Hamiltonian: soundness, Gauss law, honesty, kernel flag."""

from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest
from omnibias.core.proof import Conjecture
from omnibias.core.proof.lean_check import check_certificate, generate_obligation
from omnibias.geometry.gauge.band._core import su2_transverse_constant
from omnibias.geometry.gauge.proofmachine import build_gauge_machine
from omnibias.geometry.gauge.transfer.certificates import (
    HAMILTONIAN_GAP_KIND,
    hamiltonian_gap_schema_errors,
    replay_hamiltonian_gap,
    seal_hamiltonian_gap_certificate,
)
from omnibias.geometry.gauge.transfer.hamiltonian import (
    COUPLING_LOCK,
    LEHMANN_HOLONOMY_METHOD,
    LEHMANN_STANDARD_METHOD,
    candidate_gap,
    certified_hamiltonian_gap,
    legal_triple,
    physical_basis,
    plaquette_holonomy_trial_space,
    su2_two_plaquette_hamiltonian,
)
from omnibias.geometry.gauge.transfer.trial import su2_holonomy_trace

G1_COUPLINGS = (Fraction(1, 2), Fraction(1), Fraction(2), Fraction(4))


def _midpoint(hamiltonian: object) -> np.ndarray:
    return np.array(
        [[0.5 * (c.lo + c.hi) for c in row] for row in hamiltonian.entries]  # type: ignore[attr-defined]
    )


def test_every_basis_vector_is_a_legal_triple() -> None:
    hamiltonian = su2_two_plaquette_hamiltonian(COUPLING_LOCK, j_max=1)
    two_j_max = 2
    assert hamiltonian.dimension >= 8
    for t1, t2, ts in hamiltonian.basis:
        assert legal_triple(t1, t2, ts, two_j_max=two_j_max)
    assert physical_basis(1) == hamiltonian.basis


def test_locked_coupling_certifies_a_positive_gap() -> None:
    hamiltonian = su2_two_plaquette_hamiltonian(COUPLING_LOCK, j_max=1)
    result = certified_hamiltonian_gap(hamiltonian)
    assert result.certified is True
    assert result.spectral_gap_lower > 0.0
    assert 0.0 <= result.subdominant_ratio_upper < 1.0


def test_g2_certified_gap_never_exceeds_midpoint_eigengap() -> None:
    for coupling in G1_COUPLINGS:
        hamiltonian = su2_two_plaquette_hamiltonian(coupling, j_max=1)
        values = np.sort(np.linalg.eigvalsh(_midpoint(hamiltonian)))
        numerical = float(values[1] - values[0])
        result = certified_hamiltonian_gap(hamiltonian)
        assert result.spectral_gap_lower <= numerical + 1e-9


def test_g3_random_gauge_leaves_plaquette_trace_invariant() -> None:
    components = (0.3, -0.2, 0.5)
    bare = su2_holonomy_trace(components, length=1.0, coupling=1.0)
    u00, u01, u10, u11 = su2_transverse_constant(
        components, length=1.0, coupling=1.0
    )
    g00, g01, g10, g11 = su2_transverse_constant(
        (0.1, 0.7, -0.4), length=1.2, coupling=0.8
    )
    gd00, gd01, gd10, gd11 = (
        g00.conjugate(),
        g10.conjugate(),
        g01.conjugate(),
        g11.conjugate(),
    )
    w00 = gd00 * u00 + gd01 * u10
    w01 = gd00 * u01 + gd01 * u11
    w10 = gd10 * u00 + gd11 * u10
    w11 = gd10 * u01 + gd11 * u11
    t00 = w00 * g00 + w01 * g10
    t11 = w10 * g01 + w11 * g11
    transformed = float((t00 + t11).real)
    assert abs(transformed - bare) / max(abs(bare), 1e-15) < 1e-14


def test_g1_holonomy_is_at_least_generic_standard_basis() -> None:
    factors: list[float] = []
    for coupling in G1_COUPLINGS:
        hamiltonian = su2_two_plaquette_hamiltonian(coupling, j_max=1)
        generic_official = certified_hamiltonian_gap(hamiltonian)
        trial = plaquette_holonomy_trial_space(hamiltonian)
        holonomy_official = certified_hamiltonian_gap(hamiltonian, trial=trial)
        assert (
            holonomy_official.spectral_gap_lower + 1e-12
            >= generic_official.spectral_gap_lower
        )
        generic = candidate_gap(holonomy_official, LEHMANN_STANDARD_METHOD)
        holonomy = candidate_gap(holonomy_official, LEHMANN_HOLONOMY_METHOD)
        assert holonomy + 1e-12 >= generic
        if generic > 0.0:
            factors.append(holonomy / generic)
        else:
            factors.append(1.0)
    assert factors
    assert min(factors) >= 1.0 - 1e-12


def test_g4_gram_condition_is_sealed_and_flagged() -> None:
    hamiltonian = su2_two_plaquette_hamiltonian(COUPLING_LOCK, j_max=1)
    trial = plaquette_holonomy_trial_space(hamiltonian)
    result = certified_hamiltonian_gap(hamiltonian, trial=trial)
    assert result.trial_gram_condition is not None
    assert result.trial_flagged is trial.flagged
    sealed = seal_hamiltonian_gap_certificate(result, hamiltonian)
    assert sealed["trial_gram_condition"] == pytest.approx(trial.gram_condition)
    assert sealed["trial_flagged"] is trial.flagged


def test_g5_honesty_flags_reject_continuum_and_yang_mills() -> None:
    hamiltonian = su2_two_plaquette_hamiltonian(COUPLING_LOCK, j_max=1)
    result = certified_hamiltonian_gap(hamiltonian)
    sealed = dict(seal_hamiltonian_gap_certificate(result, hamiltonian))
    assert sealed["continuum_claim"] is False
    assert sealed["honesty"]["yang_mills_claim"] is False
    assert sealed["honesty"]["continuum_claim"] is False
    assert hamiltonian_gap_schema_errors(sealed) == []
    forged = dict(sealed)
    forged["continuum_claim"] = True
    forged["honesty"] = dict(sealed["honesty"], yang_mills_claim=True)
    errors = hamiltonian_gap_schema_errors(forged)
    assert any("continuum_claim" in err for err in errors)
    assert any("yang_mills_claim" in err for err in errors)


def test_g6_kernel_flag_cannot_be_forged() -> None:
    hamiltonian = su2_two_plaquette_hamiltonian(COUPLING_LOCK, j_max=1)
    result = certified_hamiltonian_gap(hamiltonian)
    cert = seal_hamiltonian_gap_certificate(result, hamiltonian)
    source = generate_obligation(cert)
    assert source is not None
    assert "spectral_gap_pos" in source
    checked = check_certificate(cert)
    if checked.available:
        assert checked.verified is True
    else:
        assert checked.verified is False
    machine = build_gauge_machine()
    verdict = machine.evaluate(
        Conjecture(
            "forged formal",
            HAMILTONIAN_GAP_KIND,
            {"coupling": COUPLING_LOCK, "j_max": 1},
            claims={"theorem_prover_verified": True},
        ),
        lean_check=True,
    )
    if not checked.available:
        assert verdict.status == "BLOCKED"
        assert verdict.theorem_prover_verified is False
    else:
        assert verdict.theorem_prover_verified is True


def test_replay_and_machine_prove_the_locked_coupling() -> None:
    hamiltonian = su2_two_plaquette_hamiltonian(COUPLING_LOCK, j_max=1)
    result = certified_hamiltonian_gap(hamiltonian)
    cert = seal_hamiltonian_gap_certificate(result, hamiltonian)
    assert replay_hamiltonian_gap(cert) is True
    machine = build_gauge_machine()
    verdict = machine.evaluate(
        Conjecture(
            "two-plaq gap",
            HAMILTONIAN_GAP_KIND,
            {"coupling": COUPLING_LOCK, "j_max": 1},
        )
    )
    assert verdict.status == "PROVED"
    assert verdict.schema_ok is True
    assert verdict.replay_ok is True
    assert verdict.honesty_ok is True


def test_matrix_is_symmetric() -> None:
    hamiltonian = su2_two_plaquette_hamiltonian(COUPLING_LOCK, j_max=1)
    mid = _midpoint(hamiltonian)
    np.testing.assert_allclose(mid, mid.T, atol=1e-14)


def test_j_max_2_certifies_a_positive_gap() -> None:
    hamiltonian = su2_two_plaquette_hamiltonian(COUPLING_LOCK, j_max=2)
    assert hamiltonian.dimension > len(physical_basis(1))
    result = certified_hamiltonian_gap(hamiltonian)
    assert result.certified is True
    assert result.spectral_gap_lower > 0.0
    values = np.sort(np.linalg.eigvalsh(_midpoint(hamiltonian)))
    numerical = float(values[1] - values[0])
    assert result.spectral_gap_lower <= numerical + 1e-9
