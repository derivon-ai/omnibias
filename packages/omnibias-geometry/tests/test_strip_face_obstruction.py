# SPDX-License-Identifier: Apache-2.0
"""Exact original covers, boundary charges and independent Haar/Schmidt checks."""

from copy import deepcopy
from fractions import Fraction as Q
from itertools import product
from typing import Any

import pytest
from omnibias.core.proof.certificate import seal_certificate
from omnibias.geometry.gauge.transfer import strip_face_obstruction as module

Quat = tuple[int, int, int, int]
GROUP: tuple[Quat, ...] = tuple(
    (sign * int(i == 0), sign * int(i == 1), sign * int(i == 2), sign * int(i == 3))
    for i in range(4)
    for sign in (1, -1)
)


def mul(a: Quat, b: Quat) -> Quat:
    return (
        a[0] * b[0] - sum(a[i] * b[i] for i in range(1, 4)),
        a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
        a[0] * b[2] + a[2] * b[0] + a[3] * b[1] - a[1] * b[3],
        a[0] * b[3] + a[3] * b[0] + a[1] * b[2] - a[2] * b[1],
    )


def inv(a: Quat) -> Quat:
    return a[0], -a[1], -a[2], -a[3]


@pytest.fixture(scope="module")
def result() -> dict[str, Any]:
    return module.su2_strip_face_obstruction(Q(1, 64), 3)


def test_actual_three_face_scopes_and_exact_weights(result: dict[str, Any]) -> None:
    assert result["status"] == "PASS"
    assert result["arithmetic"]["n_original_links"] == 10
    assert result["local_boundary_flux_gap_upper"] == "3/1024"
    assert [f["total_electric_weight"] for f in result["local_faces"]] == ["7/2", "3", "7/2"]
    assert [f["internal_vertices_relative_to_full_strip"] for f in result["local_faces"]] == [
        [0, 4],
        [],
        [3, 7],
    ]
    assert [p["shared_link_ids"] for p in result["adjacent_pairs"]] == [[8], [9]]
    assert result["arithmetic"]["accumulated_original_electric_weights"] == ["1"] * 10
    assert result["certificate"]["payload"] == {
        k: v for k, v in result.items() if k != "certificate"
    }
    assert module.replay_su2_strip_face_obstruction_certificate(result["certificate"])
    for flag in (
        "exact_original_operator_decomposition_verified_in_written_analysis",
        "actual_local_ground_identification_verified_in_written_analysis",
        "boundary_flux_gap_ceiling_verified_in_written_analysis",
        "adjacent_local_ground_intersection_trivial_verified_in_written_analysis",
        "strict_local_energy_frustration_verified_in_written_analysis",
    ):
        assert result[flag] is True
    for flag in (
        "global_physical_gap_upper_verified",
        "actual_ambient_conditional_gap_verified",
        "isolated_neutral_gap_transferred_to_boundary_sector",
        "knabe_hypotheses_verified",
        "actual_global_groundstate_transform_locality_verified",
        "uniform_physical_gap_verified",
        "yang_mills_claim",
        "continuum_claim",
        "analytic_proof_formally_verified",
        "mathlib_verified",
        "theorem_prover_verified",
    ):
        assert result[flag] is False


@pytest.mark.parametrize("n", (2, 3, 4, 7, 12))
def test_independent_original_star_and_local_cover(n: int) -> None:
    row = module.su2_strip_face_obstruction(Q(7, 5), n)
    edges = [(x, x + 1) for x in range(n)]
    edges += [(x + n + 1, x + n + 2) for x in range(n)]
    edges += [(x, x + n + 1) for x in range(n + 1)]
    count: dict[int, Q] = {}
    for i, face in enumerate(row["local_faces"]):
        ids = {i + 1, n + i + 1, 2 * n + i + 1, 2 * n + i + 2}
        assert set(face["original_link_ids"]) == ids
        for e in ids:
            incidence_count = 2 if 2 * n + 2 <= e <= 3 * n else 1
            weight = Q(face["electric_weights"][str(e)])
            assert weight == Q(1, incidence_count)
            count[e] = count.get(e, Q(0)) + weight
        internal = [
            v
            for v in range(2 * n + 2)
            if {e + 1 for e, edge in enumerate(edges) if v in edge} <= ids
        ]
        assert face["internal_vertices_relative_to_full_strip"] == internal
        trial = face["charged_trial_link_id"]
        assert Q(face["electric_weights"][str(trial)]) == Q(1, 2)
        assert set(edges[trial - 1]).isdisjoint(internal)
    assert len(count) == 3 * n + 1 and set(count.values()) == {Q(1)}


@pytest.mark.parametrize("kappa", (Q(1, 2**4096), Q(1, 64), Q(2, 7), Q(1), Q(8), Q(10**100)))
def test_all_positive_couplings_and_sharp_asymptotic_enclosure(kappa: Q) -> None:
    row = module.su2_strip_face_obstruction(kappa)
    upper = Q(row["local_boundary_flux_gap_upper"])
    lower = Q(row["arithmetic"]["boundary_flux_gap_lower_from_electric_and_haar"])
    assert row["status"] == "PASS"
    assert upper == 3 * kappa / 16
    assert 0 <= lower <= upper and upper - lower <= 4 / kappa
    assert module.replay_su2_strip_face_obstruction_certificate(row["certificate"])


@pytest.mark.parametrize("bad", (True, False, 0, -1, 0.1, "1/64", None, []))
def test_coupling_guards(bad: Any) -> None:
    with pytest.raises((TypeError, ValueError)):
        module.su2_strip_face_obstruction(bad)


@pytest.mark.parametrize("bad", (True, False, 0, 1, -1, 3.0, Q(3), "3", None))
def test_size_guards(bad: Any) -> None:
    with pytest.raises((TypeError, ValueError)):
        module.su2_strip_face_obstruction(1, bad)


@pytest.mark.parametrize("bad", (True, 1, "certificate", None, [], ()))
def test_malformed_replay(bad: Any) -> None:
    assert not module.replay_su2_strip_face_obstruction_certificate(bad)


@pytest.mark.parametrize(
    "attack", ("weight", "internal", "endpoint", "norm", "upper", "frustration", "global", "formal")
)
def test_resealed_claim_and_geometry_attacks(result: dict[str, Any], attack: str) -> None:
    cert = deepcopy(result["certificate"])
    payload = cert["payload"]
    if attack == "weight":
        payload["local_faces"][0]["electric_weights"]["8"] = "1"
    elif attack == "internal":
        payload["local_faces"][0]["internal_vertices_relative_to_full_strip"].append(1)
    elif attack == "endpoint":
        payload["local_faces"][0]["charged_trial_endpoint_vertices"] = [0, 4]
    elif attack == "norm":
        payload["local_faces"][0]["actual_local_trial_norm_squared"] = "2"
    elif attack == "upper":
        payload["local_boundary_flux_gap_upper"] = "0"
    elif attack == "frustration":
        payload["adjacent_pairs"][0]["common_local_ground_space"] = "one-dimensional"
    elif attack == "global":
        payload["global_physical_gap_upper_verified"] = True
    else:
        payload["analytic_proof_formally_verified"] = True
    assert not module.replay_su2_strip_face_obstruction_certificate(seal_certificate(cert))


def test_exact_group_haar_marginal_and_charged_trial_identity() -> None:
    # Q8 is a genuine SU2 subgroup and a Haar quadrature for the degree-two
    # single-link character/gradient moments. Private-link averaging is exact
    # for ANY weight on the product; here a positive nonconstant central trial.
    mass = {s: Q(0) for s in GROUP}
    for b, s, top, left in product(GROUP, repeat=4):
        loop = mul(mul(mul(b, s), inv(top)), inv(left))
        phi = 1 + Q(loop[0], 2)
        assert phi > 0
        mass[s] += phi * phi
    total = sum(mass.values(), Q(0))
    assert set(mass.values()) == {total / 8}
    mean = sum((mass[s] * 2 * s[0] for s in GROUP), Q(0)) / total
    norm = sum((mass[s] * (2 * s[0]) ** 2 for s in GROUP), Q(0)) / total
    energy = sum((mass[s] * (1 - s[0] ** 2) for s in GROUP), Q(0)) / total
    assert mean == 0 and norm == 1 and energy == Q(3, 4)
    assert Q(1, 2) * Q(1, 2) * energy == Q(3, 16)
    # Boundary center charge, despite invariance at every other vertex.
    assert all(2 * mul((-1, 0, 0, 0), s)[0] == -2 * s[0] for s in GROUP)


def test_exact_schmidt_mixedness_without_probability_marginal_confusion() -> None:
    # For phi(U)=1+chi(U)/4, its normalized shared-rung reduced state
    # has eigenvalues16/17 and1/68 (four times). Both probability marginals
    # are uniform; the quantum state is nevertheless mixed.
    values = [[1 + Q(mul(s, r)[0], 2) for r in GROUP] for s in GROUP]
    z = Q(17, 16)
    gram = [
        [sum((values[i][r] * values[j][r] for r in range(8)), Q(0)) / (64 * z) for j in range(8)]
        for i in range(8)
    ]
    assert all(gram[i][i] == Q(1, 8) for i in range(8))
    assert sum((gram[i][i] for i in range(8)), Q(0)) == 1
    purity = sum((entry * entry for row in gram for entry in row), Q(0))
    assert purity == Q(1025, 1156) < 1
    # Constant vector eigenvalue is the largest exact Schmidt probability.
    assert all(sum(row, Q(0)) == Q(16, 17) for row in gram)
    assert Q(16, 17) ** 2 + 4 * Q(1, 68) ** 2 == purity
