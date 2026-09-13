# SPDX-License-Identifier: Apache-2.0
"""Exact conditional tail arithmetic and refusal to infer unprovided graph premises."""

from __future__ import annotations

from copy import deepcopy
from fractions import Fraction as Q
from typing import Any

import pytest
from omnibias.core.proof.certificate import seal_certificate, verify_certificate_digest
from omnibias.geometry.gauge.transfer.finite_graph_resolvent import (
    replay_su2_finite_graph_linearized_tail_certificate as replay,
)
from omnibias.geometry.gauge.transfer.finite_graph_resolvent import (
    su2_finite_graph_linearized_tail as certify,
)


def _theta() -> dict[str, Any]:
    return certify(5, plaquette_count=2, plaquettes_per_edge=2, energy_root=8)


def test_exact_theta_tail_in_original_energy_normalization() -> None:
    result = _theta()
    a = result["witness"]["arithmetic"]
    # E>=8^2 in the full seven-link Casimir, not a three-unit-link cutoff.
    assert a["g"] == "4/25"
    assert a["energy_cutoff"] == "64"
    assert Q(a["anchored_derivative_tail_upper"]) == Q(16, 25)
    assert Q(a["shifted_anchor_tail_upper"]) == Q(3, 50)
    assert result["operator_tail_upper"] == "7/10"
    assert result["status"] == "CONDITIONAL_BOUND"
    assert result["conditional_tail_arithmetic_verified"] is True
    assert result["numeric_tail_below_one"] is True
    assert replay(result["certificate"])


@pytest.mark.parametrize("root", [Q(1, 3), Q(1), Q(3, 2), Q(8), Q(100)])
def test_rational_cutoff_and_distinct_decay_terms(root: Q) -> None:
    coupling, base, count, cap = Q(17, 3), Q(7, 5), 19, 4
    result = certify(coupling, plaquette_count=count, plaquettes_per_edge=cap,
                     energy_root=root, decay_base=base)
    g = 4 / coupling**2
    expected = g * base**2 * (16 * count / root + 12 * cap / root**2)
    assert Q(result["operator_tail_upper"]) == expected
    assert Q(result["witness"]["arithmetic"]["energy_cutoff"]) == root**2
    assert result["numeric_tail_below_one"] is (expected < 1)
    assert replay(result["certificate"])


def test_tail_decreases_but_volume_count_does_not_disappear() -> None:
    tails = [
        Q(certify(5, plaquette_count=2, plaquettes_per_edge=2, energy_root=t)
          ["operator_tail_upper"]) for t in (1, 2, 4, 8, 16)
    ]
    assert all(x > y for x, y in zip(tails, tails[1:], strict=False))
    larger = certify(5, plaquette_count=200, plaquettes_per_edge=2, energy_root=8)
    assert Q(larger["operator_tail_upper"]) > 1
    assert larger["numeric_tail_below_one"] is False
    assert larger["conditional_tail_arithmetic_verified"] is True
    assert larger["volume_uniform_inverse_verified"] is False
    assert replay(larger["certificate"])


def test_exact_unit_tail_is_not_a_strict_contraction() -> None:
    # P=1,q=4 is a loose incidence cap. The bracket is64 and g=1/64.
    result = certify(16, plaquette_count=1, plaquettes_per_edge=4, energy_root=1)
    assert result["operator_tail_upper"] == "1"
    assert result["numeric_tail_below_one"] is False
    assert result["finite_gate_verified"] is True
    assert result["finite_inverse_verified"] is False


def test_all_required_premises_remain_external_even_for_small_tail() -> None:
    witness = _theta()["witness"]
    assert "input electric energies" in witness["projection"]
    assert "not a square Galerkin truncation" in witness["approximation"]
    premises = " ".join(witness["external_premises"])
    for phrase in ("canonical positive-axis", "every vertex", "four distinct original",
                   "nuclear norm eight", "unit original-edge electric",
                   "ambient original edge", "energy_root squared",
                   "original anchored Fourier Banach space"):
        assert phrase in premises


@pytest.mark.parametrize("field", [
    "actual_graph_verified", "actual_reference_verified", "finite_inverse_verified",
    "fourier_nuclear_inverse_verified", "actual_vacuum_verified",
    "target_hamiltonian_gap_verified", "volume_uniform_inverse_verified",
    "infinite_volume_claim", "uniform_in_a_claim", "continuum_claim",
    "yang_mills_claim", "yang_mills_mass_gap_claim",
])
def test_rehash_cannot_promote_unverified_premises_or_claims(field: str) -> None:
    result = _theta()
    assert result[field] is False
    certificate = deepcopy(result["certificate"])
    certificate["honesty"][field] = True
    assert not replay(seal_certificate(certificate))


@pytest.mark.parametrize("field", [
    "g", "energy_cutoff", "square_nuclear_norm_premise",
    "anchored_derivative_tail_upper", "shifted_anchor_tail_upper",
    "operator_tail_upper", "operator_tail_formula", "tail_strictly_below_one",
])
def test_rehashed_altered_arithmetic_rejected(field: str) -> None:
    certificate = deepcopy(_theta()["certificate"])
    certificate["payload"]["witness"]["arithmetic"][field] = "unearned"
    certificate = seal_certificate(certificate)
    assert verify_certificate_digest(certificate)
    assert not replay(certificate)


@pytest.mark.parametrize("field", [
    "operator", "norm", "projection", "approximation", "units",
    "external_premises", "conditional_conclusion",
])
def test_rehashed_assumption_or_norm_rewrites_rejected(field: str) -> None:
    certificate = deepcopy(_theta()["certificate"])
    certificate["payload"]["witness"][field] = "unearned replacement"
    assert not replay(seal_certificate(certificate))


@pytest.mark.parametrize("field,value", [
    ("kappa", "6"), ("plaquette_count", 3), ("plaquettes_per_edge", 4),
    ("energy_root", "9"), ("decay_base", "2"),
])
def test_rehashed_changed_inputs_recomputed(field: str, value: Any) -> None:
    certificate = deepcopy(_theta()["certificate"])
    certificate["payload"]["witness"]["inputs"][field] = value
    assert not replay(seal_certificate(certificate))


@pytest.mark.parametrize("field,value", [
    ("kappa", 0), ("kappa", -1), ("kappa", True), ("kappa", 5.0),
    ("plaquette_count", 0), ("plaquette_count", True), ("plaquette_count", Q(2)),
    ("plaquettes_per_edge", 0), ("plaquettes_per_edge", False), ("plaquettes_per_edge", 2.0),
    ("energy_root", 0), ("energy_root", -1), ("energy_root", True), ("energy_root", 8.0),
    ("decay_base", Q(1, 2)), ("decay_base", True), ("decay_base", 1.0),
])
def test_exact_input_guards(field: str, value: Any) -> None:
    inputs: dict[str, Any] = {
        "kappa": 5, "plaquette_count": 2, "plaquettes_per_edge": 2, "energy_root": 8,
    }
    inputs[field] = value
    with pytest.raises((ValueError, TypeError)):
        certify(**inputs)


@pytest.mark.parametrize("value", [None, False, [], {}, {"payload": {}}, {"digest": "bad"}])
def test_malformed_replay_refuses(value: Any) -> None:
    assert replay(value) is False


def test_formal_tiers_stay_unearned() -> None:
    result = _theta()
    assert result["theorem_prover_verified"] is False
    assert result["mathlib_verified"] is False
