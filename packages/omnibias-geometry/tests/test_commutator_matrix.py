# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Independent polynomial, angular-channel and source-replay regressions."""

from copy import deepcopy
from fractions import Fraction as Q
from itertools import product
from math import factorial
from typing import Any

import pytest
import sympy as sp
from omnibias.core.proof.certificate import seal_certificate, verify_certificate_digest
from omnibias.geometry.gauge.transfer import commutator_matrix as matrix


@pytest.fixture(scope="module")
def source() -> dict[str, Any]:
    return matrix.su2_commutator_matrix_gap()


def _gaussian_moment(powers: tuple[int, ...], width: Q) -> Q:
    answer = Q(1)
    for n in powers:
        if n % 2:
            return Q(0)
        answer *= Q(factorial(n), factorial(n // 2)) / (4 * width) ** (n // 2)
    return answer


def test_quartic_gaussian_expectation_from_nine_variable_wick_expansion(
    source: dict[str, Any],
) -> None:
    # Expand every cross-product square into monomials, then integrate each
    # original coordinate using its Gaussian moment recurrence.
    width = Q(5, 4)
    total = Q(0)
    for i, j in ((0, 1), (0, 2), (1, 2)):
        for b, c in ((0, 1), (1, 2), (2, 0)):
            terms = [(Q(1), (3 * i + b, 3 * j + c)), (Q(-1), (3 * i + c, 3 * j + b))]
            for (v, indices), (w, others) in product(terms, repeat=2):
                powers = tuple((indices + others).count(k) for k in range(9))
                total += v * w * _gaussian_moment(powers, width)
    kinetic = sum(
        (
            width**2 * _gaussian_moment(tuple(2 if k == i else 0 for k in range(9)), width)
            for i in range(9)
        ),
        Q(0),
    )
    assert total == Q(72, 25)
    assert kinetic == Q(45, 8)
    assert total + kinetic == Q(source["arithmetic"]["ground_energy_upper"]) == Q(1701, 200)


def test_barta_expression_by_independent_symbolic_differentiation(source: dict[str, Any]) -> None:
    r = sp.Symbol("r", positive=True)
    u = r**2 * sp.exp(-(r ** sp.Rational(3, 2)) / 2)
    quotient = sp.simplify((-sp.diff(u, r, 2) + (r + 2 / r**2) * u) / u)
    assert sp.simplify(quotient - 7 * r / 16 - 27 / (8 * sp.sqrt(r))) == 0
    minimum_cube = Q(27, 4) * Q(7, 16) * Q(27, 8) ** 2
    assert minimum_cube == Q(source["arithmetic"]["barta_minimum_cubed_lower"])
    assert minimum_cube > Q(16, 5) ** 3


def test_all_low_angular_singlet_channels_enumerated_independently(source: dict[str, Any]) -> None:
    energies = []
    for angular in product(range(5), repeat=3):
        a, b, c = angular
        # SO(3) Clebsch--Gordan contains a scalar iff the third angular
        # momentum occurs in the product of the first two.
        if not abs(a - b) <= c <= a + b:
            continue
        for radial in product(range(2), repeat=3):
            if angular == (0, 0, 0) and radial == (0, 0, 0):
                continue
            energies.append(
                sum(
                    (
                        Q(16, 5) if ell else Q(4) if n else Q(23, 10)
                        for ell, n in zip(angular, radial, strict=True)
                    ),
                    Q(0),
                )
            )
    assert (
        min(energies) == Q(source["arithmetic"]["first_singlet_excited_energy_lower"]) == Q(43, 5)
    )
    assert Q(source["arithmetic"]["singlet_gap_lower"]) == Q(19, 200)


def test_transverse_oscillator_all_space_allocation_is_exact() -> None:
    # Six ordered pairs allocate T_j/4+V_ij/2; the remaining kinetic is T/2.
    kinetic = [Q(1, 2)] * 3
    potential = {(i, j): Q(0) for i in range(3) for j in range(i + 1, 3)}
    for i, j in product(range(3), repeat=2):
        if i != j:
            kinetic[j] += Q(1, 4)
            potential[min(i, j), max(i, j)] += Q(1, 2)
    assert kinetic == [1, 1, 1]
    assert list(potential.values()) == [1, 1, 1]
    assert 4 * Q(1, 4) * Q(1, 2) == Q(1, 2)  # squared two-dimensional oscillator floor


def test_homogeneous_dilation_does_not_claim_volume_uniformity(source: dict[str, Any]) -> None:
    kappa, n = sp.symbols("kappa n", positive=True)
    length = kappa ** sp.Rational(1, 3) / n
    kinetic = kappa / (2 * n**3) / length**2
    quartic = n**3 / (2 * kappa) * length**4
    scale = kappa ** sp.Rational(1, 3) / (2 * n)
    assert sp.simplify(kinetic - scale) == sp.simplify(quartic - scale) == 0
    assert not source["spatial_lattice_invariant_subspace_verified"]
    assert not source["uniform_spatial_volume_gap_verified"]


def test_free_control_has_arbitrarily_small_gaussian_energy() -> None:
    # Deleting the quartic invalidates the confining comparison. This free
    # singlet trial family has energy tending to zero instead of19/200.
    assert 9 * Q(1, 1024) / 2 < Q(19, 200)


def test_complete_dynamic_replay_and_proof_scope(source: dict[str, Any]) -> None:
    assert source["status"] == "PASS"
    assert source["scalar_source_replay_verified"]
    assert source["actual_matrix_singlet_gap_verified_in_written_analysis"]
    assert matrix.replay_su2_commutator_matrix_gap_certificate(source["certificate"])
    for flag in (
        "finite_spin_truncation_used",
        "quadratic_mass_added",
        "compact_lattice_gap_verified",
        "continuum_claim",
        "yang_mills_mass_gap_claim",
        "analytic_proof_formally_verified",
        "theorem_prover_verified",
        "mathlib_verified",
    ):
        assert source[flag] is False


@pytest.mark.parametrize(
    "case", ["upper", "excited", "gap", "sector", "source_cell", "source_flag", "scope"]
)
def test_resealed_full_source_attacks(source: dict[str, Any], case: str) -> None:
    forged = deepcopy(source["certificate"])
    p = forged["payload"]
    if case in {"upper", "excited", "gap"}:
        key = {
            "upper": "ground_energy_upper",
            "excited": "first_singlet_excited_energy_lower",
            "gap": "singlet_gap_lower",
        }[case]
        p["arithmetic"][key] = "1"
    elif case == "sector":
        p["physical_sector"] = "all scalar states"
    elif case == "source_cell":
        sc = p["scalar_source"]["certificate"]
        sc["payload"]["shooting"][0]["cells"][0]["C"] = ["0", "0"]
        sc.pop("digest", None)
        p["scalar_source"]["certificate"] = seal_certificate(sc)
    elif case == "source_flag":
        p["scalar_source"]["status"] = "INCONCLUSIVE"
    else:
        p["compact_lattice_gap_verified"] = True
    forged.pop("digest", None)
    forged = seal_certificate(forged)
    assert verify_certificate_digest(forged)
    assert not matrix.replay_su2_commutator_matrix_gap_certificate(forged)


def test_failure_of_actual_scalar_replay_prevents_promotion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(matrix, "replay_airy_sturm_certificate", lambda _: False)
    failed = matrix.su2_commutator_matrix_gap()
    assert failed["status"] == "INCONCLUSIVE"
    assert not failed["actual_matrix_singlet_gap_verified_in_written_analysis"]
    assert not failed["all_space_comparison_verified_in_written_analysis"]


def test_detached_scalar_report_mutation_prevents_promotion(
    source: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    bad = deepcopy(source["scalar_source"])
    bad["shooting"][0]["cells"][0]["q"] = "0"
    monkeypatch.setattr(matrix, "airy_half_line_lower_bounds", lambda: bad)
    failed = matrix.su2_commutator_matrix_gap()
    assert not failed["scalar_source_replay_verified"]
    assert failed["status"] == "INCONCLUSIVE"


@pytest.mark.parametrize("bad", [None, [], (), "certificate", 0, True, {}])
def test_malformed_replayer_inputs(bad: Any) -> None:
    assert not matrix.replay_su2_commutator_matrix_gap_certificate(bad)
