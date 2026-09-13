# SPDX-License-Identifier: Apache-2.0
"""Exact physics, norm-conversion, complete-tail and replay regressions."""

from __future__ import annotations

import random
from copy import deepcopy
from fractions import Fraction as Q
from itertools import product
from typing import Any

import pytest
import sympy as sp  # type: ignore[import-untyped]
from omnibias.core.proof.certificate import seal_certificate
from omnibias.geometry.gauge.transfer import adjacent_resolvent as module
from omnibias.geometry.gauge.transfer.adjacent_resolvent import (
    replay_su2_adjacent_linearized_inverse_certificate as replay,
)
from omnibias.geometry.gauge.transfer.adjacent_resolvent import (
    su2_adjacent_linearized_inverse as certify,
)
from sympy.physics.wigner import wigner_3j, wigner_6j  # type: ignore[import-untyped]


@pytest.fixture(scope="module")
def witness() -> dict[str, Any]:
    return certify(5, cutoff=3)


def test_exact_default_retained_boundary_and_all_spin_gate(witness: dict[str, Any]) -> None:
    a = witness["witness"]["arithmetic"]
    assert a["retained_dimension"] == 22
    assert a["retained_defect_upper"] == "59/540"
    assert Q(a["boundary_defect_upper"]) < Q(1, 10)
    assert a["far_tail_upper"] == a["all_spin_defect_upper"] == "14336/25625"
    assert a["old_strip_linear_majorant"] == "224/75"
    assert Q(witness["original_N_inverse_upper"]) < 6
    assert Q(witness["original_N_inverse_upper"]) == Q(3, 2) * Q(witness["scalar_M_inverse_upper"])
    assert witness["status"] == "PASS" and witness["digest_verified"]
    assert replay(witness["certificate"])


def test_an_actual_original_norm_obstruction_is_passed_by_preconditioning() -> None:
    result = certify(Q(14, 5), cutoff=7)
    a = result["witness"]["arithmetic"]
    assert a["retained_dimension"] == 153
    assert a["original_N_single_column_norm_lower"] == "325/294"
    assert Q(a["original_N_single_column_norm_lower"]) > 1
    assert a["all_spin_defect_upper"] == "704000/726327"
    assert result["full_spin_fourier_reference_inverse_verified"] is True
    assert Q(result["original_N_inverse_upper"]) < 162
    assert replay(result["certificate"])


def test_insufficient_cutoff_is_a_faithfully_replayable_refusal() -> None:
    small = certify(4, cutoff=1)
    larger = certify(4, cutoff=3)
    assert small["finite_reference_block_inverse_verified"] is True
    assert small["status"] == "INCONCLUSIVE"
    assert small["witness"]["arithmetic"]["far_tail_upper"] == "1280/837"
    assert small["original_N_inverse_upper"] is None
    assert small["full_spin_fourier_reference_inverse_verified"] is False
    assert larger["status"] == "PASS"
    assert replay(small["certificate"]) and replay(larger["certificate"])


def test_independent_low_character_products_force_the_norm_obstruction() -> None:
    # Haar identities: chi_p^2=1+chi_adj,p and chi_p chi_q=b110+3b112.
    # Since input b101=chi_p/2, these are all nonconstant products W*b101.
    products = {(2, 0, 2): Q(3, 2), (1, 1, 0): Q(1, 2), (1, 1, 2): Q(3, 2)}
    energies = {(2, 0, 2): Q(8), (1, 1, 0): Q(9, 2), (1, 1, 2): Q(13, 2)}
    # Four/six/seven original edges give these energies. No module energy helper.
    coefficients = {s: (6 - energies[s]) * m / energies[s] for s, m in products.items()}
    assert coefficients == {(2, 0, 2): Q(-3, 8), (1, 1, 0): Q(1, 6), (1, 1, 2): Q(-3, 26)}
    assert module._kernel_column((1, 0, 1), Q(1)) == coefficients
    anchor_costs = [
        sum(
            (
                Q(s[e], 2) * energies[s] * (s[0] + 1) ** 2 * (s[1] + 1) ** 2 * abs(c)
                for s, c in coefficients.items()
            ),
            Q(0),
        )
        for e in range(3)
    ]
    assert anchor_costs == [Q(39), Q(12), Q(39)]
    input_anchors = [Q(6), Q(0), Q(6)]
    assert max(anchor_costs) / max(input_anchors) == Q(13, 2)
    assert sum(anchor_costs) / sum(input_anchors) == Q(15, 2)


def test_triangle_constraints_sharpen_the_scalar_norm_conversion_and_both_ends_are_sharp() -> None:
    rng = random.Random(1791)
    states = [state for state in module._basis(4) if state != module.VACUUM]
    for _ in range(10):
        coefficients = {state: Q(rng.randrange(-50, 51), rng.randrange(1, 12)) for state in states}
        anchors = [
            sum(
                (
                    Q(state[e], 2) * module._energy(state) * module._nuclear(state) * abs(c)
                    for state, c in coefficients.items()
                ),
                Q(0),
            )
            for e in range(3)
        ]
        assert 2 * max(anchors) <= sum(anchors) <= 3 * max(anchors)
    # Single admissible modes attain each endpoint, so neither constant can
    # improve uniformly without more information about the input or output.
    assert sum((1, 0, 1)) == 2 * max((1, 0, 1))
    assert sum((2, 2, 2)) == 3 * max((2, 2, 2))


@pytest.mark.parametrize(
    "labels", [(0, 3, 3), (1, 2, 1), (1, 2, 3), (2, 3, 3), (2, 2, 0), (3, 3, 4)]
)
def test_general_theta_endpoint_tensor_normalization_not_just_low_spin(
    labels: tuple[int, int, int],
) -> None:
    spins = [sp.Rational(label, 2) for label in labels]
    dims = [label + 1 for label in labels]
    for isolated in (0, 1):
        others = [j for j in range(3) if j != isolated]
        reduced = sp.zeros(dims[isolated], dims[others[0]] * dims[others[1]])
        for indices in product(*(range(d) for d in dims)):
            ms = [j - m for j, m in zip(spins, indices, strict=True)]
            reduced[
                indices[isolated], indices[others[0]] * dims[others[1]] + indices[others[1]]
            ] = wigner_3j(*spins, *ms)
        assert sp.simplify(reduced * reduced.H) == sp.eye(dims[isolated]) / dims[isolated]
    # The two rectangular endpoint singular factors cancel the two all-in/out
    # bivalent factors. Mixed bivalent ranks multiply the endpoint ranks.
    rank = dims[0] ** 2 * dims[1] ** 2
    singular_square = sp.Rational(1, dims[0] * dims[1]) * dims[0] * dims[1]
    assert singular_square == 1
    assert module._nuclear(labels) == rank
    ambient = dims[0] ** 3 * dims[1] ** 3 * dims[2]
    assert Q(rank, ambient) == Q(1, dims[0] * dims[1] * dims[2])


def test_magnetic_coefficients_match_independent_wigner_values_and_complete_mass() -> None:
    for state in module._basis(4):
        outputs = dict(module._magnetic_column(state))
        assert sum(outputs.values()) == 4
        for active in (0, 1):
            spectator = 1 - active
            component = {s: v for s, v in outputs.items() if s[spectator] == state[spectator]}
            assert sum(component.values()) == 2
            for out, value in component.items():
                exact = (
                    wigner_6j(
                        sp.Rational(out[active], 2),
                        sp.Rational(out[2], 2),
                        sp.Rational(state[spectator], 2),
                        sp.Rational(state[2], 2),
                        sp.Rational(state[active], 2),
                        sp.Rational(1, 2),
                    )
                    ** 2
                    * (out[active] + 1)
                    * (out[2] + 1)
                )
                assert value == Q(exact) >= 0


def test_complete_tail_on_exact_grid_and_seeded_random_high_labels() -> None:
    # Finite diagnostics guard the universal polynomial/tensor proof in the docs.
    rng = random.Random(77104)
    states = list(module._basis(6))[1:]
    for _ in range(80):
        a, b = rng.randrange(50), rng.randrange(50)
        s = rng.randrange(abs(a - b), a + b + 1, 2)
        if (a, b, s) != (0, 0, 0):
            states.append((a, b, s))
    for state in states:
        j = max(state)
        assert sum(state) >= 2 * j
        assert module._energy(state) >= Q(j * (5 * j + 16), 8)
        norm = sum(
            (module._weight(s) * abs(c) for s, c in module._kernel_column(state, Q(1)).items()),
            Q(0),
        ) / module._weight(state)
        assert norm <= module._tail(Q(1), j)
    for j in [*range(1, 20), 2**54 + 1]:
        difference = Q(
            (j + 2) * (5 * j**3 + 30 * j**2 + 68 * j + 21),
            j**2 * (j + 1) ** 2 * (5 * j + 16) * (5 * j + 21),
        )
        assert module._tail(Q(1), j) - module._tail(Q(1), j + 1) == 256 * difference > 0


def test_exact_fraction_free_inverse_handles_row_swaps_and_singular_refusals() -> None:
    matrices = [
        [[Q(0), Q(1)], [Q(2, 3), Q(1)]],
        [[Q(1, 3), Q(2), Q(-1)], [Q(3), Q(-1, 2), Q(5)], [Q(2), Q(1), Q(1)]],
        [[Q(1), Q(2)], [Q(2), Q(4)]],
    ]
    for matrix in matrices:
        exact = sp.Matrix(matrix)
        inverse, _, verified = module._exact_inverse(matrix)
        if exact.det():
            assert verified and inverse is not None
            assert sp.Matrix(inverse) == exact.inv()
        else:
            assert inverse is None and not verified


def test_larger_finite_matrix_cannot_escape_the_complete_tail_bound(
    witness: dict[str, Any],
) -> None:
    basis = [s for s in module._basis(5) if s != module.VACUUM]
    index = {s: i for i, s in enumerate(basis)}
    matrix = sp.eye(len(basis))
    for j, state in enumerate(basis):
        for out, value in module._kernel_column(state, Q(4, 75)).items():
            if out in index:
                matrix[index[out], j] -= sp.Rational(value)
    inverse = matrix.inv(method="DM")
    norm = max(
        sum((module._weight(s) * abs(Q(inverse[i, j])) for i, s in enumerate(basis)), Q(0))
        / module._weight(source)
        for j, source in enumerate(basis)
    )
    assert norm <= Q(witness["scalar_M_inverse_upper"])


@pytest.mark.parametrize("value", [True, False, 5.0, "5", None, 0, -1])
def test_nonexact_or_nonpositive_couplings_are_refused(value: Any) -> None:
    with pytest.raises((TypeError, ValueError)):
        certify(value)


@pytest.mark.parametrize("value", [True, False, 3.0, "3", Q(3), None, 0, -1])
def test_cutoff_requires_an_exact_positive_integer(value: Any) -> None:
    with pytest.raises((TypeError, ValueError)):
        certify(5, cutoff=value)


@pytest.mark.parametrize(
    "field",
    [
        "retained_defect_columns",
        "boundary_defect_columns",
        "far_tail_first_label",
        "far_tail_upper",
        "norm_conversion_factor",
        "original_N_inverse_upper",
        "retained_inverse_exact_pivots",
        "finite_inverse_residual_verified",
    ],
)
def test_resealed_boundary_tail_or_conversion_tampering_fails(
    witness: dict[str, Any], field: str
) -> None:
    certificate = deepcopy(witness["certificate"])
    certificate["payload"]["witness"]["arithmetic"][field] = "0"
    assert not replay(seal_certificate(certificate))


@pytest.mark.parametrize(
    "field",
    [
        "actual_vacuum_verified",
        "actual_vacuum_claim",
        "physical_mass_gap_claim",
        "uniform_in_volume_claim",
        "infinite_volume_claim",
        "uniform_in_a_claim",
        "all_scale_refinement_claim",
        "spatial_locality_claim",
        "continuum_claim",
        "yang_mills_claim",
        "yang_mills_mass_gap_claim",
        "theorem_prover_verified",
        "mathlib_verified",
    ],
)
def test_reference_inverse_cannot_be_promoted_by_resealing(
    witness: dict[str, Any], field: str
) -> None:
    assert witness[field] is False
    certificate = deepcopy(witness["certificate"])
    certificate["honesty"][field] = True
    assert not replay(seal_certificate(certificate))


def test_basis_orientation_raw_input_and_unexpected_premises_are_canonical(
    witness: dict[str, Any],
) -> None:
    for field in ("basis", "boundary_basis", "normalization", "norm_conversion", "invented_gap"):
        certificate = deepcopy(witness["certificate"])
        certificate["payload"]["witness"][field] = "changed"
        assert not replay(seal_certificate(certificate))
    certificate = deepcopy(witness["certificate"])
    certificate["payload"]["witness"]["inputs"]["kappa"] = "10/2"
    assert not replay(seal_certificate(certificate))


@pytest.mark.parametrize("value", [None, [], (), "certificate", 1, True, {}])
def test_malformed_certificates_return_false(value: Any) -> None:
    assert replay(value) is False
