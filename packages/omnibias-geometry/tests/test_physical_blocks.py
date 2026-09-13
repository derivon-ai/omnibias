# SPDX-License-Identifier: Apache-2.0
"""Independent exact controls and public replay tests for physical blocks."""

from __future__ import annotations

from copy import deepcopy
from fractions import Fraction as Q
from random import Random
from typing import Any

import pytest
from omnibias.core.proof.certificate import seal_certificate
from omnibias.geometry.gauge.transfer.adjacent_cone_vacuum import su2_adjacent_cone_vacuum
from omnibias.geometry.gauge.transfer.physical_blocks import (
    conditional_overlap_budget,
    conditional_overlap_transfer_budget,
    physical_block_controls,
    replay_conditional_overlap_certificate,
    replay_conditional_overlap_transfer_certificate,
    replay_physical_block_control_certificate,
    replay_su2_theta_physical_block_certificate,
    su2_theta_physical_block_gap,
)


@pytest.fixture(scope="module")
def source() -> dict[str, Any]:
    return su2_adjacent_cone_vacuum(6, correction_radius=Q(6, 25))


@pytest.fixture(scope="module")
def theta(source: dict[str, Any]) -> dict[str, Any]:
    return su2_theta_physical_block_gap(source["certificate"])


def test_actual_theta_constants_and_negative_source_curvature(
    source: dict[str, Any],
    theta: dict[str, Any],
) -> None:
    assert source["actual_vacuum_verified"] is True
    assert source["curvature_gap_verified"] is False
    a = theta["arithmetic"]
    gamma = Q(3, 4) * Q(2383, 2700) ** 4
    cross = Q(5, 12) * Q(1, 9) + Q(2, 3) * Q(6, 25)
    assert Q(a["conditional_A_gap_lower"]) == gamma
    assert Q(a["conditional_B_gap_lower"]) == 2 * gamma
    assert Q(a["mixed_log_vacuum_hessian_upper"]) == cross == Q(557, 2700)
    determinant = (gamma - Q(1, 5)) * (2 * gamma - Q(1, 5)) - 4 * cross**2
    assert Q(a["shifted_determinant"]) == determinant > 0
    assert theta["status"] == "PASS"
    assert theta["physical_gap_lower"] == "3/5"
    assert theta["source_gap_used_as_premise"] is False
    assert theta["new_family_gap_improvement_claim"] is False
    assert replay_su2_theta_physical_block_certificate(theta["certificate"])
    assert su2_theta_physical_block_gap()["certificate"] == theta["certificate"]


def test_geometry_really_contains_complete_internal_stars(theta: dict[str, Any]) -> None:
    geometry = theta["geometry"]
    edges = geometry["edges"]
    for letter in ("A", "B"):
        block = set(geometry[f"block_{letter}_edges"])
        for vertex in geometry[f"internal_{letter}_vertices"]:
            assert {i + 1 for i, edge in enumerate(edges) if vertex in edge} <= block
    assert set(geometry["block_A_edges"]) | set(geometry["block_B_edges"]) == set(range(1, 8))
    assert set(geometry["block_A_edges"]).isdisjoint(geometry["block_B_edges"])
    assert geometry["boundary_vertices"] == [0, 2, 4]
    assert theta["all_boundary_flux_sectors_retained"] is True


def test_original_seed_incidence_operator_bound_is_independent() -> None:
    # M M^T has eigenvalues 0,2,6. Check the dominating quadratic form
    # by an independent rational grid and random vectors, not float SVD.
    matrix = [[1, 1, 0, 0], [0, 0, 1, 1], [1, 1, 1, 1]]
    rng = Random(127)
    vectors = [
        [Q(i), Q(j), Q(k), Q(last)]
        for i in (-1, 0, 1)
        for j in (-1, 0, 1)
        for k in (-1, 0, 1)
        for last in (-1, 0, 1)
    ]
    vectors += [[Q(rng.randint(-9, 9), rng.randint(1, 9)) for _ in range(4)] for _ in range(100)]
    for vector in vectors:
        output = [sum((Q(m) * v for m, v in zip(row, vector, strict=True)), Q(0)) for row in matrix]
        assert sum(v * v for v in output) <= 6 * sum(v * v for v in vector)
    assert Q(6) < Q(5, 2) ** 2


def test_actual_overlap_uses_global_density_not_quantum_gap(theta: dict[str, Any]) -> None:
    overlap = theta["actual_overlap"]
    minorant = Q(2183, 2700) ** 4
    assert Q(overlap["joint_density_lower"]) == minorant
    assert Q(overlap["maximal_correlation_upper"]) == 1 - minorant
    assert overlap["marginals_are_actual_product_haar"] is True
    assert overlap["heat_bath_gap_is_quantum_hamiltonian_gap"] is False
    assert theta["certificate"]["honesty"]["actual_joint_overlap_verified"] is True
    assert theta["uniform_in_volume_claim"] is False


def test_failed_matrix_floor_does_not_erase_earned_overlap(source: dict[str, Any]) -> None:
    result = su2_theta_physical_block_gap(source["certificate"], comparison_floor=1, sweeps=3)
    assert result["status"] == "INCONCLUSIVE"
    assert result["physical_gap_lower"] is None
    assert result["actual_vacuum_verified"] is True
    delta = Q(result["actual_overlap"]["maximal_correlation_upper"])
    assert Q(result["actual_overlap"]["alternating_projection_norm_upper"]) == delta**5
    assert replay_su2_theta_physical_block_certificate(result["certificate"])


def test_failed_actual_source_never_supplies_conditional_or_overlap_premises() -> None:
    bad = su2_adjacent_cone_vacuum(3, correction_radius=Q(6, 25))
    result = su2_theta_physical_block_gap(bad["certificate"])
    assert result["status"] == "INCONCLUSIVE"
    for key in (
        "actual_vacuum_verified",
        "physical_gap_verified",
        "actual_joint_overlap_verified",
        "conditional_internal_gauss_gaps_verified",
    ):
        assert result[key] is False
    assert result["actual_overlap"]["joint_density_lower"] is None
    assert replay_su2_theta_physical_block_certificate(result["certificate"])


@pytest.mark.parametrize(
    "where",
    [
        "gap",
        "geometry",
        "overlap",
        "source",
        "comparison",
        "parent",
        "honesty",
        "meta",
        "claim",
        "input",
    ],
)
def test_theta_rehashed_tampering_rejected(theta: dict[str, Any], where: str) -> None:
    cert = deepcopy(theta["certificate"])
    p = cert["payload"]
    if where == "gap":
        p["physical_gap_lower"] = "100"
    elif where == "geometry":
        p["geometry"]["boundary_vertices"] = []
    elif where == "overlap":
        p["actual_overlap"]["maximal_correlation_upper"] = "0"
    elif where in {"source", "comparison"}:
        key = "source_certificate" if where == "source" else "coefficient_comparison_certificate"
        p[key]["honesty"]["actual_vacuum_verified"] = False
        p[key] = seal_certificate(p[key])
    elif where == "parent":
        p["continuum_claim"] = True
    elif where == "honesty":
        cert["honesty"]["made_up_verified"] = True
    elif where == "meta":
        cert["meta"]["transcend_backend"] = "forged"
    elif where == "claim":
        cert["claim"] = "Clay solved"
    else:
        p["inputs"]["comparison_floor"] = "2/10"
    assert not replay_su2_theta_physical_block_certificate(seal_certificate(cert))


def test_public_mutation_cannot_poison_future_certificate(source: dict[str, Any]) -> None:
    first = su2_theta_physical_block_gap(source["certificate"])
    expected = deepcopy(first["certificate"])
    first["actual_overlap"]["joint_density_lower"] = "9"
    first["source_certificate"]["honesty"]["actual_vacuum_verified"] = False
    assert first["certificate"] == expected
    first["certificate"]["payload"]["arithmetic"]["mixed_log_vacuum_hessian_upper"] = "0"
    assert su2_theta_physical_block_gap(source["certificate"])["certificate"] == expected


@pytest.mark.parametrize(
    "error,mx,my,expected",
    [
        (Q(0), Q(1), Q(1), Q(0)),
        (Q(1, 8), Q(1, 4), Q(1), Q(1, 4)),
        (Q(1), Q(1), Q(1), Q(1)),
        (Q(2), Q(1), Q(1), Q(1)),
    ],
)
def test_overlap_exact_square_roots(error: Q, mx: Q, my: Q, expected: Q) -> None:
    result = conditional_overlap_budget(error, mx, my)
    assert Q(result["arithmetic"]["maximal_correlation_upper"]) == expected
    assert (result["status"] == "PASS") == (expected < 1)
    assert result["input_analytic_premises_verified"] is False
    assert result["actual_joint_overlap_verified"] is False
    assert replay_conditional_overlap_certificate(result["certificate"])


def test_overlap_outward_rounding_can_refuse_near_one() -> None:
    result = conditional_overlap_budget(Q(999, 1000), 1, 1, sqrt_bits=1)
    assert result["status"] == "INCONCLUSIVE"
    assert result["arithmetic"]["maximal_correlation_upper"] == "1"
    for bits in (8, 16, 32):
        result = conditional_overlap_budget(Q(1, 3), Q(1, 2), Q(1), sqrt_bits=bits)
        upper = Q(result["arithmetic"]["maximal_correlation_upper"])
        assert upper**2 >= Q(2, 9)
        assert (upper - Q(1, 2**bits)) ** 2 < Q(2, 9)


def test_transfer_counts_multiple_decompositions_as_an_explicit_premise() -> None:
    result = conditional_overlap_transfer_budget(Q(3, 4), Q(1, 3), decompositions=4)
    assert result["arithmetic"]["larger_block_gap_lower"] == "2/5"
    assert result["arithmetic"]["averaged_energy_multiplicity_upper"] == "5/4"
    assert result["actual_vacuum_verified"] is False
    assert result["iteration_at_all_scales_verified"] is False
    assert replay_conditional_overlap_transfer_certificate(result["certificate"])
    assert conditional_overlap_transfer_budget(1, 1)["status"] == "INCONCLUSIVE"


def _matmul(a: list[list[Q]], b: list[list[Q]]) -> list[list[Q]]:
    return [
        [sum((a[i][k] * b[k][j] for k in range(len(b))), Q(0)) for j in range(len(b[0]))]
        for i in range(len(a))
    ]


def test_two_projection_power_exponent_from_independent_rational_matrices() -> None:
    # v=(3/5,4/5), P projects onto e1, Q projects onto v. On this
    # common-centered plane the product has rank one and known Frobenius norm.
    p = [[Q(1), Q(0)], [Q(0), Q(0)]]
    q = [[Q(9, 25), Q(12, 25)], [Q(12, 25), Q(16, 25)]]
    product = _matmul(p, q)
    power = product
    for n in range(1, 9):
        norm_squared = sum(value**2 for row in power for value in row)
        assert norm_squared == Q(3, 5) ** (4 * n - 2)
        power = _matmul(power, product)


def test_doeblin_actual_density_bound_on_grid_and_random_centered_functions() -> None:
    # Independent finite doubly-stochastic kernels exercise the proof of (9).
    # Entries are densities relative to uniform probability bases, not masses.
    rng = Random(5027)
    for size in (2, 3, 5):
        weights = [Q(i + 1, size * (size + 1) // 2) for i in range(size)]
        kernel = [[size * weights[(j - i) % size] for j in range(size)] for i in range(size)]
        for minorant in (Q(1, 8), Q(1, 2), Q(7, 8)):
            density = [[minorant + (1 - minorant) * value for value in row] for row in kernel]
            assert all(sum(row) == size for row in density)
            assert all(sum(density[i][j] for i in range(size)) == size for j in range(size))
            for _ in range(24):
                f = [Q(rng.randint(-9, 9), rng.randint(1, 7)) for _ in range(size)]
                g = [Q(rng.randint(-9, 9), rng.randint(1, 7)) for _ in range(size)]
                fm, gm = sum(f, Q(0)) / size, sum(g, Q(0)) / size
                f, g = [v - fm for v in f], [v - gm for v in g]
                covariance = (
                    sum(f[i] * g[j] * density[i][j] for i in range(size) for j in range(size))
                    / size**2
                )
                assert (
                    covariance**2
                    <= (1 - minorant) ** 2 * sum(v * v for v in f) * sum(v * v for v in g) / size**2
                )


def test_generic_density_error_normalization_is_probability_base() -> None:
    # A positive 2x2 joint law with nonuniform own marginals. Enumerating
    # centered directions checks the exact Hilbert-Schmidt normalization.
    joint = [[Q(2, 5), Q(1, 10)], [Q(1, 5), Q(3, 10)]]
    mx = [sum(row) for row in joint]
    my = [sum(joint[i][j] for i in range(2)) for j in range(2)]
    residual = [[4 * (joint[i][j] - mx[i] * my[j]) for j in range(2)] for i in range(2)]
    epsilon = max(abs(v) for row in residual for v in row)
    result = conditional_overlap_budget(epsilon, 2 * min(mx), 2 * min(my))
    covariance = joint[1][1] - mx[1] * my[1]
    correlation_squared = covariance**2 / (mx[0] * mx[1] * my[0] * my[1])
    assert correlation_squared <= Q(result["arithmetic"]["maximal_correlation_upper"]) ** 2 < 1
    assert result["actual_joint_overlap_verified"] is False


@pytest.mark.parametrize("p", [Q(1, 2), Q(1, 10), Q(1, 1000), Q(1, 1000000)])
def test_positive_rare_bernoulli_controls_are_direct_probability_arithmetic(p: Q) -> None:
    result = physical_block_controls(rare_probability=p)
    b = result["rare_bernoulli"]
    joint = [[Q(value) for value in row] for row in b["joint_probabilities"]]
    assert min(value for row in joint for value in row) > 0
    marg = [sum(row) for row in joint]
    residual = [[joint[i][j] - marg[i] * marg[j] for j in range(2)] for i in range(2)]
    assert (
        Q(b["total_variation_to_own_product"]) == sum(abs(x) for row in residual for x in row) / 2
    )
    covariance = joint[1][1] - marg[1] ** 2
    assert Q(b["maximal_correlation"]) == covariance / (marg[0] * marg[1])
    assert Q(b["density_residual_Linf_on_uniform_base"]) == 4 * max(
        abs(x) for row in residual for x in row
    )
    if p <= Q(1, 10):
        assert Q(b["maximal_correlation"]) > Q(9, 10)
        assert Q(b["total_variation_to_own_product"]) < 2 * p


@pytest.mark.parametrize("length,width", [(8, 2), (8, 4), (16, 4), (32, 2), (64, 16)])
def test_pinned_gaussian_overlap_from_independent_green_functions(length: int, width: int) -> None:
    result = physical_block_controls(chain_length=length, overlap_width=width)[
        "pinned_gaussian_overlap"
    ]
    left, right = (length - width) // 2, (length + width) // 2

    def massless(i: int, j: int) -> Q:
        return Q(min(i, j)) - Q(i * j, length)

    correlation_squared = massless(left, right) ** 2 / (
        massless(left, left) * massless(right, right)
    )
    assert correlation_squared == Q(result["massless_maximal_correlation"]) ** 2
    # Exact inverse tridiagonal Green function for diagonal 3, off-diagonal -1.
    determinants = [1, 3]
    for _ in range(2, length):
        determinants.append(3 * determinants[-1] - determinants[-2])

    def massive(i: int, j: int) -> Q:
        lo, hi = min(i, j), max(i, j)
        return Q(determinants[lo - 1] * determinants[length - hi - 1], determinants[length - 1])

    massive_squared = massive(left, right) ** 2 / (massive(left, left) * massive(right, right))
    assert massive_squared <= Q(result["massive_maximal_correlation_upper"]) ** 2


def test_fixed_width_massless_overlap_deteriorates_but_massive_bound_does_not() -> None:
    values = [
        physical_block_controls(chain_length=n, overlap_width=2)["pinned_gaussian_overlap"]
        for n in (8, 32, 128, 1024)
    ]
    correlations = [Q(v["massless_maximal_correlation"]) for v in values]
    assert correlations == sorted(correlations)
    assert correlations[-1] > Q(99, 100)
    assert len({v["massive_maximal_correlation_upper"] for v in values}) == 1


@pytest.mark.parametrize("size", range(3, 13))
def test_gaussian_cycle_grid_and_random_exact_quadratic_form(size: int) -> None:
    rng = Random(911 + size)
    for mass in (Q(0), Q(1, size**2), Q(2, 3)):
        result = physical_block_controls(side=size, mass_squared=mass)
        assert Q(result["gaussian_cycle"]["smallest_eigenvalue"]) == mass
        assert result["gaussian_cycle"]["probability_defined_on_full_space"] == (mass > 0)
        for vector in [[Q(1)] * size, [Q(i % 3 - 1) for i in range(size)]] + [
            [Q(rng.randint(-9, 9), rng.randint(1, 7)) for _ in range(size)] for _ in range(10)
        ]:
            norm = sum(x * x for x in vector)
            quadratic = (mass + 2) * norm - 2 * sum(
                vector[i] * vector[(i + 1) % size] for i in range(size)
            )
            differences = sum((vector[i] - vector[(i + 1) % size]) ** 2 for i in range(size))
            assert quadratic == mass * norm + differences >= mass * norm
        assert replay_physical_block_control_certificate(result["certificate"])


def test_gaussian_and_star_controls_do_not_claim_nonlinear_no_go() -> None:
    result = physical_block_controls()
    assert Q(result["original_link_gaussian"]["unscaled_row_margin_upper"]) == -Q(125, 128)
    assert result["original_link_gaussian"]["actual_nonlinear_vacuum_no_go_proved"] is False
    star = result["internal_star_cap"]
    assert Q(star["dirichlet_energy"]) / Q(star["variance"]) == Q(3, 4)
    assert star["physical_internal_gauss_gap_capped_by_this_test"] is False
    assert result["actual_vacuum_verified"] is False


@pytest.mark.parametrize("value", [True, 0.2, "1/5", None, [], {}])
def test_exact_rational_inputs_refuse_non_exact_types(value: Any) -> None:
    with pytest.raises((TypeError, ValueError)):
        conditional_overlap_budget(value, 1, 1)
    with pytest.raises((TypeError, ValueError)):
        conditional_overlap_transfer_budget(value, 0)
    with pytest.raises((TypeError, ValueError)):
        physical_block_controls(mass_squared=value)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"epsilon": -1},
        {"marginal_x_lower": 0},
        {"marginal_y_lower": 2},
        {"sqrt_bits": True},
        {"sweeps": 0},
    ],
)
def test_overlap_guard_values(kwargs: dict[str, Any]) -> None:
    args: dict[str, Any] = {"epsilon": Q(1, 4), "marginal_x_lower": 1, "marginal_y_lower": 1}
    args.update(kwargs)
    with pytest.raises((TypeError, ValueError)):
        conditional_overlap_budget(**args)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"side": 2},
        {"side": True},
        {"rare_probability": 0},
        {"regulator": 1},
        {"chain_length": 7},
        {"overlap_width": 3},
        {"overlap_width": 16},
        {"chain_length": 1024, "overlap_width": 514},
    ],
)
def test_control_guards(kwargs: dict[str, Any]) -> None:
    with pytest.raises((TypeError, ValueError)):
        physical_block_controls(**kwargs)


def test_all_replayers_refuse_malformed_top_levels() -> None:
    malformed: list[Any] = [None, True, [], "", {}, {"payload": []}]
    for replay in (
        replay_su2_theta_physical_block_certificate,
        replay_conditional_overlap_certificate,
        replay_conditional_overlap_transfer_certificate,
        replay_physical_block_control_certificate,
    ):
        for value in malformed:
            assert not replay(value)


@pytest.mark.parametrize("kind", ["residual", "transfer", "controls"])
def test_generic_canonical_replay_rejects_forged_analytic_flags(kind: str) -> None:
    if kind == "residual":
        result, replay = (
            conditional_overlap_budget(Q(1, 4), 1, 1),
            replay_conditional_overlap_certificate,
        )
    elif kind == "transfer":
        result, replay = (
            conditional_overlap_transfer_budget(1, Q(1, 4)),
            replay_conditional_overlap_transfer_certificate,
        )
    else:
        result, replay = physical_block_controls(), replay_physical_block_control_certificate
    for location in ("payload", "honesty"):
        cert = deepcopy(result["certificate"])
        cert[location]["actual_vacuum_verified"] = True
        assert not replay(seal_certificate(cert))
