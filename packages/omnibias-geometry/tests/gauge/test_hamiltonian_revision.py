# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Recoupling revisions and certificate scope cannot be changed by rehashing."""
from __future__ import annotations

import copy
from fractions import Fraction

import pytest
from omnibias.core.proof.certificate import seal_certificate
from omnibias.geometry.gauge.transfer.certificates import (
    replay_hamiltonian_gap,
    seal_hamiltonian_gap_certificate,
)
from omnibias.geometry.gauge.transfer.hamiltonian import (
    certified_hamiltonian_gap,
    su2_four_plaquette_hamiltonian,
    su2_three_plaquette_hamiltonian,
    su2_two_plaquette_hamiltonian,
)
from omnibias.geometry.gauge.transfer.matrices import decode_scalar, encode_scalar


@pytest.fixture(scope="module")
def certificate():
    operator = su2_two_plaquette_hamiltonian(1, j_max=1)
    return seal_hamiltonian_gap_certificate(certified_hamiltonian_gap(operator), operator)


def test_corrected_operator_replays_and_legacy_revision_is_refused(certificate):
    assert replay_hamiltonian_gap(certificate) is True
    changed = copy.deepcopy(certificate)
    changed["parameters"].pop("operator_revision")
    assert replay_hamiltonian_gap(seal_certificate(changed)) is False


@pytest.mark.parametrize("field,value", [
    ("continuum_claim", True), ("theorem_prover_verified", True), ("dimension", 999),
    ("lambda0_upper", -1e6), ("lambda1_lower", 1e6),
])
def test_rehashed_promotions_fail(certificate, field, value):
    changed = copy.deepcopy(certificate)
    changed[field] = value
    assert replay_hamiltonian_gap(seal_certificate(changed)) is False


def test_three_and_four_graph_identification_is_explicitly_unverified():
    for builder in (su2_three_plaquette_hamiltonian, su2_four_plaquette_hamiltonian):
        operator = builder(1, j_max=1)
        assert operator.parameters["operator_identification"] == "unverified_multi_plaquette_graph"


@pytest.mark.parametrize("value", [1, -1, 2**80+1, Fraction(7, 3), Fraction(2), .1, 1.0])
def test_scalar_replay_preserves_exact_integers_and_rationals(value):
    restored = decode_scalar(encode_scalar(value))
    assert restored == value
    if isinstance(value, int) or isinstance(value, Fraction):
        assert not isinstance(restored, float)
    else:
        assert isinstance(restored, float)
