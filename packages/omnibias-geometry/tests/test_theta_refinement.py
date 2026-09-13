# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Independent exact regressions for the geometry theta refinement API.

The finite spin enumerations exercise the written all-spin argument; they do
not constitute a spin-tail proof. The optional SymPy oracle computes Wigner
6j contractions independently of the production rational Jacobi formula.
"""

from __future__ import annotations

import copy
from fractions import Fraction as Q
from typing import Any

import pytest
from omnibias.core.proof.certificate import seal_certificate, verify_certificate_digest
from omnibias.geometry.gauge.transfer.theta_refinement import (
    replay_su2_theta_refinement_certificate,
    su2_theta_refinement,
)


def _matrix(result: dict[str, Any]) -> list[list[Q]]:
    return [[Q(entry) for entry in row] for row in result["witness"]["second_order_matrix"]]


def _sixj_matrix(
    alpha: Q, paths: tuple[Q, Q, Q], left: Q, right: Q, z: Q, cutoff: int
) -> list[list[Q]]:
    """Independent normalized spin-network oracle, with exact radicals."""
    sympy = pytest.importorskip("sympy")
    wigner = pytest.importorskip("sympy.physics.wigner")
    half = sympy.Rational(1, 2)
    # Enumerate all admissible intermediate triples with the new spin 1/2.
    # Fundamental multiplication cannot reach another new-path spin from P.
    states = [
        (a, b, 1)
        for a in range(cutoff + 2)
        for b in range(cutoff + 2)
        if abs(a - b) <= 1 <= a + b and (a + b + 1) % 2 == 0
    ]

    def amplitude(n: int, state: tuple[int, int, int]) -> Any:
        value = sympy.S.Zero
        a, b, s = state
        for active, strength in ((0, left), (1, right)):
            spectator = b if active == 0 else a
            shifted = a if active == 0 else b
            if spectator != n or abs(shifted - n) != 1:
                continue
            coefficient = wigner.wigner_6j(
                n * half, 0, n * half, s * half, shifted * half, half
            ) ** 2
            # Two normalized endpoint intertwiners and Peter-Weyl norms.
            normalization = (n + 1) * sympy.sqrt((a + 1) * (b + 1) * (s + 1))
            value += sympy.Rational(str(strength)) * coefficient * normalization / (n + 1)
        return value

    columns = [[amplitude(n, state) for state in states] for n in range(cutoff + 1)]
    denominators = [
        alpha * sum(
            (weight * Q(label * (label + 2), 4)
             for weight, label in zip(paths, state, strict=True)), Q(0)
        ) - z
        for state in states
    ]
    return [
        [Q(str(sympy.simplify(sum(
            (left_column[k] * right_column[k] / sympy.Rational(str(denominators[k]))
             for k in range(len(states))), sympy.S.Zero
        )))) for right_column in columns]
        for left_column in columns
    ]


def test_default_exact_constants_and_scope() -> None:
    result = su2_theta_refinement()
    witness = result["witness"]
    assert result["status"] == "ENCLOSED"
    expected = {
        "coarse_electric_gap": "9",
        "complement_electric_floor": "6",
        "new_minus_coarse_electric_gap": "-3",
        "electric_cross_block_norm": "0",
        "magnetic_cross_block_norm": "1",
        "complement_H_lower": "4",
        "neumann_ratio_upper": "1/3",
        "self_energy_remainder_lower": "0",
        "self_energy_remainder_upper": "1/48",
        "wilson_nonnegative_convention_shift": "2",
    }
    for key, value in expected.items():
        assert witness[key] == value
    assert witness["cross_block_square"] == {
        "identity_coefficient": "1/2",
        "fundamental_outer_character_coefficient": "1/4",
    }
    assert _matrix(result)[:3] == [
        [Q(1, 12), Q(1, 24), Q(0), Q(0), Q(0)],
        [Q(1, 24), Q(1, 24), Q(1, 72), Q(0), Q(0)],
        [Q(0), Q(1, 72), Q(1, 54), Q(1, 144), Q(0)],
    ]
    assert result["all_complement_spins_covered"] is True
    assert result["digest_verified"] is True
    for key in (
        "vacuum_preserving_embedding_claim", "all_scale_refinement_claim",
        "spectral_gap_claim", "infinite_volume_claim", "uniform_in_a_claim",
        "continuum_claim", "yang_mills_claim", "yang_mills_mass_gap_claim",
        "theorem_prover_verified", "mathlib_verified",
    ):
        assert result[key] is False
    assert result["certificate"]["meta"]["transcend_backend"] == "not_used"


@pytest.mark.parametrize("paths", [(1, 1, 1), (3, 3, 1), (1, 3, 4), (5, 2, 3)])
def test_complete_electric_floor_has_exact_admissible_witness(
    paths: tuple[int, int, int],
) -> None:
    result = su2_theta_refinement(path_weights=paths, magnetic_left=0, magnetic_right=0)
    witness = result["witness"]
    admissible = [
        (a, b, s)
        for a in range(9) for b in range(9) for s in range(1, 9)
        if abs(a - b) <= s <= a + b and (a + b + s) % 2 == 0
    ]

    def energy(state: tuple[int, int, int]) -> Q:
        return 2 * sum(
            (Q(w * n * (n + 2), 4) for w, n in zip(paths, state, strict=True)), Q(0)
        )

    attaining = tuple(witness["complement_attaining_two_j"])
    assert attaining in admissible
    assert energy(attaining) == Q(witness["complement_electric_floor"])
    assert min(map(energy, admissible)) == energy(attaining)
    assert Q(witness["new_minus_coarse_electric_gap"]) == Q(3, 2) * (
        paths[2] - max(paths[:2])
    )


@pytest.mark.parametrize("offset, enclosed", [(Q(-1, 10**30), True), (Q(0), False),
                                               (Q(1, 10**30), False)])
def test_neumann_boundary_is_strict_and_exact(offset: Q, enclosed: bool) -> None:
    result = su2_theta_refinement(spectral_parameter=4 + offset)
    witness = result["witness"]
    assert result["self_energy_enclosure_verified"] is enclosed
    assert witness["neumann_domain_verified"] is enclosed
    assert result["status"] == ("ENCLOSED" if enclosed else "INCONCLUSIVE_DOMAIN")
    if not enclosed:
        assert witness["second_order_matrix"] is None
        assert witness["self_energy_remainder_lower"] is None
        assert witness["self_energy_remainder_upper"] is None
        assert result["exact_refinement_identities_verified"] is True
    assert replay_su2_theta_refinement_certificate(result["certificate"])


@pytest.mark.parametrize("left,right", [(Q(0), Q(0)), (Q(0), Q(2, 3)), (Q(2, 3), Q(0))])
def test_zero_or_single_plaquette_limits(left: Q, right: Q) -> None:
    result = su2_theta_refinement(magnetic_left=left, magnetic_right=right)
    witness = result["witness"]
    matrix = _matrix(result)
    assert Q(witness["magnetic_cross_block_norm"]) == left + right
    assert witness["cross_block_square"]["fundamental_outer_character_coefficient"] == "0"
    assert all(value == 0 for i, row in enumerate(matrix) for j, value in enumerate(row) if i != j)
    if left == right == 0:
        assert all(value == 0 for row in matrix for value in row)
        assert witness["self_energy_remainder_upper"] == "0"


@pytest.mark.parametrize("alpha,paths,left,right,z", [
    (Q(2), (Q(3), Q(3), Q(1)), Q(1, 2), Q(1, 2), Q(0)),
    (Q(5, 3), (Q(2), Q(7, 2), Q(3)), Q(2, 5), Q(1, 7), Q(-1, 3)),
    (Q(4), (Q(1), Q(5), Q(2)), Q(0), Q(3, 5), Q(1, 2)),
    (Q(3), (Q(4), Q(1), Q(7, 3)), Q(4, 7), Q(0), Q(-2)),
])
def test_rational_entries_match_independent_sixj_contractions(
    alpha: Q, paths: tuple[Q, Q, Q], left: Q, right: Q, z: Q,
) -> None:
    cutoff = 5
    result = su2_theta_refinement(
        electric=alpha, path_weights=paths, magnetic_left=left,
        magnetic_right=right, spectral_parameter=z, coarse_two_j_max=cutoff,
    )
    assert result["status"] == "ENCLOSED"
    actual = _matrix(result)
    assert actual == _sixj_matrix(alpha, paths, left, right, z, cutoff)
    assert all(actual[i][j] == actual[j][i] for i in range(cutoff + 1)
               for j in range(cutoff + 1))
    assert all(actual[i][j] == 0 for i in range(cutoff + 1)
               for j in range(cutoff + 1) if abs(i - j) > 1)


def test_pack_size_does_not_change_full_complement_bound() -> None:
    small = su2_theta_refinement(coarse_two_j_max=0)
    large = su2_theta_refinement(coarse_two_j_max=12)
    for key in ("complement_electric_floor", "magnetic_cross_block_norm",
                "self_energy_remainder_upper", "neumann_ratio_upper"):
        assert small["witness"][key] == large["witness"][key]
    assert _matrix(small)[0][0] == _matrix(large)[0][0]


def test_default_coarse_tail_and_display_errors_are_exact() -> None:
    witness = su2_theta_refinement()["witness"]
    assert witness["second_order_coarse_tail_norm_upper"] == "1/90"
    assert witness["second_order_coarse_boundary_entry"] == "1/360"
    assert witness["second_order_display_error_upper"] == "1/72"
    assert witness["full_self_energy_display_error_upper"] == "5/144"
    # The boundary coefficient is an actual entry, just outside the display.
    extended = _matrix(su2_theta_refinement(coarse_two_j_max=5))
    assert Q(witness["second_order_coarse_boundary_entry"]) == extended[4][5]


def test_coarse_tail_at_zero_cutoff_and_decay_to_interaction_remainder() -> None:
    zero = su2_theta_refinement(coarse_two_j_max=0)["witness"]
    assert zero["second_order_coarse_tail_norm_upper"] == "1/6"
    assert zero["second_order_coarse_boundary_entry"] == "1/24"
    assert zero["second_order_display_error_upper"] == "5/24"
    assert zero["full_self_energy_display_error_upper"] == "11/48"
    previous = Q(zero["full_self_energy_display_error_upper"])
    for cutoff in (1, 4, 16, 64):
        witness = su2_theta_refinement(coarse_two_j_max=cutoff)["witness"]
        # Independent closed simplification for equal outer path weights.
        scale = (cutoff + 1) * (cutoff + 2)
        assert Q(witness["second_order_coarse_tail_norm_upper"]) == Q(1, 3 * scale)
        assert Q(witness["second_order_coarse_boundary_entry"]) == Q(1, 12 * scale)
        assert Q(witness["second_order_display_error_upper"]) == Q(5, 12 * scale)
        current = Q(witness["full_self_energy_display_error_upper"])
        assert current == Q(1, 48) + Q(5, 12 * scale)
        assert Q(1, 48) < current < previous
        previous = current


@pytest.mark.parametrize("z", [4, 6, 100])
def test_display_error_is_not_reported_outside_resolvent_domain(z: int) -> None:
    result = su2_theta_refinement(spectral_parameter=z)
    assert result["status"] == "INCONCLUSIVE_DOMAIN"
    for key in (
        "second_order_coarse_tail_norm_upper", "second_order_coarse_boundary_entry",
        "second_order_display_error_upper", "full_self_energy_display_error_upper",
    ):
        assert result["witness"][key] is None


@pytest.mark.parametrize("cutoff", [0, 4, 16])
def test_zero_magnetic_has_zero_tail_and_display_error(cutoff: int) -> None:
    witness = su2_theta_refinement(
        magnetic_left=0, magnetic_right=0, coarse_two_j_max=cutoff
    )["witness"]
    for key in (
        "second_order_coarse_tail_norm_upper", "second_order_coarse_boundary_entry",
        "second_order_display_error_upper", "full_self_energy_display_error_upper",
    ):
        assert witness[key] == "0"


def test_parity_remainder_is_quartic_and_better_than_unsigned_neumann() -> None:
    values = []
    for strength in (Q(1, 2), Q(1, 4)):
        witness = su2_theta_refinement(magnetic_left=strength, magnetic_right=strength)["witness"]
        cross = Q(witness["magnetic_cross_block_norm"])
        denominator = Q(witness["complement_electric_floor"])
        ratio = Q(witness["neumann_ratio_upper"])
        remainder = Q(witness["self_energy_remainder_upper"])
        assert remainder == cross**2 / denominator * ratio**2 / (1 - ratio**2)
        assert remainder < cross**2 / denominator * ratio / (1 - ratio)
        values.append(remainder)
    assert values[0] / values[1] > 16


@pytest.mark.parametrize("kwargs, error", [
    ({"electric": 0}, ValueError), ({"electric": -1}, ValueError),
    ({"electric": 2.0}, TypeError), ({"electric": True}, TypeError),
    ({"path_weights": (1, 2)}, ValueError), ({"path_weights": "123"}, ValueError),
    ({"path_weights": (1, 0, 2)}, ValueError), ({"path_weights": (1, -1, 2)}, ValueError),
    ({"path_weights": (1, 2.0, 3)}, TypeError), ({"path_weights": (1, True, 3)}, TypeError),
    ({"magnetic_left": -1}, ValueError), ({"magnetic_right": Q(-1, 3)}, ValueError),
    ({"magnetic_left": 0.5}, TypeError), ({"magnetic_right": True}, TypeError),
    ({"spectral_parameter": 0.0}, TypeError), ({"spectral_parameter": False}, TypeError),
    ({"coarse_two_j_max": -1}, ValueError), ({"coarse_two_j_max": 2.0}, ValueError),
    ({"coarse_two_j_max": True}, ValueError),
])
def test_invalid_inputs_are_rejected(kwargs: dict[str, Any], error: type[Exception]) -> None:
    with pytest.raises(error):
        su2_theta_refinement(**kwargs)


@pytest.mark.parametrize("z", [0, 4, 100])
def test_canonical_replay_accepts_both_outcomes(z: int) -> None:
    certificate = su2_theta_refinement(spectral_parameter=z)["certificate"]
    assert verify_certificate_digest(certificate)
    assert replay_su2_theta_refinement_certificate(certificate)


@pytest.mark.parametrize("key,value", [
    ("complement_electric_floor", "7"),
    ("magnetic_cross_block_norm", "1/2"),
    ("self_energy_remainder_upper", "0"),
    ("second_order_coarse_tail_norm_upper", "0"),
    ("second_order_coarse_boundary_entry", "0"),
    ("second_order_display_error_upper", "0"),
    ("full_self_energy_display_error_upper", "0"),
    ("display_error_scope", "finite sampled vectors only"),
    ("all_complement_spins_covered", False),
    ("vacuum_preserving_embedding_claim", True),
])
def test_rehashed_witness_tampering_fails_replay(key: str, value: Any) -> None:
    original = su2_theta_refinement()["certificate"]
    forged = copy.deepcopy(original)
    forged["payload"]["witness"][key] = value
    assert not verify_certificate_digest(forged)
    forged = seal_certificate(forged)
    assert verify_certificate_digest(forged)
    assert not replay_su2_theta_refinement_certificate(forged)


def test_rehashed_matrix_honesty_and_metadata_tampering_fail_replay() -> None:
    original = su2_theta_refinement()["certificate"]
    for section in ("matrix", "honesty", "meta", "claim", "input"):
        forged = copy.deepcopy(original)
        if section == "matrix":
            forged["payload"]["witness"]["second_order_matrix"][0][0] = "0"
        elif section == "honesty":
            forged["honesty"]["continuum_claim"] = True
        elif section == "meta":
            forged["meta"]["analytic_implication"] = "unrelated theorem"
        elif section == "claim":
            forged["claim"] = "continuum theorem"
        else:
            forged["payload"]["witness"]["inputs"]["electric"] = "3"
        assert not replay_su2_theta_refinement_certificate(seal_certificate(forged))


@pytest.mark.parametrize("bad", [None, [], 1, {}, {"digest": "sha256:invalid"}])
def test_malformed_replay_is_false(bad: Any) -> None:
    assert not replay_su2_theta_refinement_certificate(bad)
