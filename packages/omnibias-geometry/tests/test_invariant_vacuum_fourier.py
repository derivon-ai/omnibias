# SPDX-License-Identifier: Apache-2.0
"""Independent exact gates, graph structure and hostile certificate replay.

These regressions check finite arithmetic and API scope. They do not turn
sampling or certificate hashing into verification of the analytic theorem.
"""

from copy import deepcopy
from fractions import Fraction as Q
from itertools import combinations, permutations
from random import Random

import pytest
from omnibias.core.proof.certificate import seal_certificate, verify_certificate_digest
from omnibias.geometry.gauge.transfer.invariant_vacuum_fourier import (
    _simple_graph_girth,
)
from omnibias.geometry.gauge.transfer.invariant_vacuum_fourier import (
    invariant_vacuum_fourier_bounds as bounds,
)
from omnibias.geometry.gauge.transfer.invariant_vacuum_fourier import (
    invariant_vacuum_fourier_family as family,
)
from omnibias.geometry.gauge.transfer.invariant_vacuum_fourier import (
    replay_invariant_vacuum_fourier_certificate as replay,
)

SQUARE = [(0, 1), (1, 2), (2, 3), (3, 0)]
LOOP = [[1, 2, 3, 4]]


def _square(**kwargs):
    parameters = dict(group="su2", kappa=27, correction_radius=Q(1, 9), plaquettes=LOOP)
    parameters.update(kwargs)
    return bounds(4, SQUARE, **parameters)


@pytest.mark.parametrize(
    ("group", "kappa", "radius", "expected"),
    [
        (
            "su2",
            27,
            Q(1, 9),
            {
                "minimum_nonconstant_invariant_energy": Q(3),
                "bilinear_constant": Q(4, 3),
                "seed_norm_upper": Q(128, 729),
                "seed_hessian_row_upper": Q(32, 2187),
                "self_map_slack": Q(2423, 1594323),
                "contraction_upper": Q(1672, 2187),
                "actual_log_vacuum_hessian_row_upper": Q(194, 2187),
                "curvature_candidate_lower": Q(1411, 4374),
                "neutral_gap_lower": Q(1411, 324),
                "vacuum_resolvent_contraction_upper": Q(324, 1735),
                "charged_linear_upper_coefficient": Q(81, 8),
                "actual_tv_influence_row_upper": Q(496, 729),
                "actual_factorization_constant_upper": Q(729, 233),
            },
        ),
        (
            "su3",
            48,
            Q(1, 4),
            {
                "minimum_nonconstant_invariant_energy": Q(16, 3),
                "bilinear_constant": Q(3, 8),
                "seed_norm_upper": Q(9, 16),
                "seed_hessian_row_upper": Q(1, 384),
                "residual_norm_upper": Q(243, 2048),
                "self_map_slack": Q(5, 2048),
                "contraction_upper": Q(39, 64),
                "actual_log_vacuum_hessian_row_upper": Q(97, 384),
                "curvature_candidate_lower": Q(47, 192),
                "neutral_gap_lower": Q(47, 8),
                "vacuum_resolvent_contraction_upper": Q(8, 55),
                "charged_linear_upper_coefficient": Q(32),
                "actual_tv_influence_row_upper": Q(51, 128),
                "actual_factorization_constant_upper": Q(128, 77),
            },
        ),
    ],
)
def test_exact_cubic_family_constants_and_scope(group, kappa, radius, expected):
    report = family(group, kappa, correction_radius=radius)
    assert report["status"] == "PASS"
    assert report["finite_gate_verified"]
    assert report["volume_uniform_finite_graph_family_verified"]
    assert report["volume_uniform_charged_linear_bound_verified"]
    assert report["volume_uniform_factorization_bound_verified"]
    assert not report["explicit_graph_neutral_gap_verified"]
    assert replay(report["certificate"])
    witness = report["witness"]
    assert witness["energy_units"] == "dimensionless aH"
    assert "at every vertex" in witness["gauss_constraint"]
    assert "entire scalar product-group" in witness["gap_scope"]
    assert "no dynamical fundamental matter" in witness["charged_scope"]
    for key, value in expected.items():
        assert Q(witness["arithmetic"][key]) == value, key
    assert witness["arithmetic"]["charged_linear_lower_coefficient"] == str(
        expected["neutral_gap_lower"]
    )
    for flag in (
        "infinite_volume_claim",
        "uniform_in_a_claim",
        "continuum_claim",
        "yang_mills_claim",
        "yang_mills_mass_gap_claim",
        "static_confinement_claim",
        "string_tension_claim",
        "theorem_prover_verified",
        "mathlib_verified",
        "analytic_implication_formally_verified",
    ):
        assert report[flag] is False


@pytest.mark.parametrize(("group", "kappa", "radius"), [("su2", 20, Q(1, 9)), ("su3", 32, Q(1, 4))])
def test_failed_fixed_point_earns_no_actual_vacuum_bound(group, kappa, radius):
    report = family(group, kappa, correction_radius=radius)
    assert report["status"] == "INCONCLUSIVE"
    assert not report["finite_gate_verified"]
    assert not report["volume_uniform_finite_graph_family_verified"]
    a = report["witness"]["arithmetic"]
    assert not a["fixed_point_verified"]
    assert a["actual_log_vacuum_hessian_row_upper"] is None
    assert a["actual_tv_influence_row_upper"] is None
    assert a["neutral_gap_lower"] == "0"
    assert a["charged_linear_lower_coefficient"] == "0"
    assert a["vacuum_resolvent_contraction_upper"] is None
    assert verify_certificate_digest(report["certificate"])
    assert not replay(report["certificate"])


def test_fixed_point_and_curvature_are_separate_obligations():
    report = family("su3", 1000, correction_radius=Q(3, 8))
    a = report["witness"]["arithmetic"]
    assert a["fixed_point_verified"]
    assert a["actual_log_vacuum_hessian_row_upper"] is not None
    assert Q(a["curvature_candidate_lower"]) < 0
    assert not a["neutral_gap_verified"]
    assert a["neutral_gap_lower"] == "0"
    assert report["status"] == "INCONCLUSIVE"
    assert not replay(report["certificate"])


def test_contraction_equality_is_refused():
    report = family("su2", 100, correction_radius=Q(3, 8), weighted_incidence_cap=0)
    a = report["witness"]["arithmetic"]
    assert a["contraction_upper"] == "1"
    assert a["self_map_verified"]
    assert not a["strict_contraction_verified"]
    assert not a["fixed_point_verified"]


def test_a_checked_graph_can_pass_when_its_coarse_family_does_not():
    report = _square(kappa=20)
    assert report["status"] == "PASS"
    assert report["explicit_graph_neutral_gap_verified"]
    assert not report["volume_uniform_finite_graph_family_verified"]
    assert not report["witness"]["family"]["arithmetic"]["neutral_gap_verified"]
    assert report["witness"]["arithmetic"]["neutral_gap_verified"]
    assert replay(report["certificate"])


def test_unused_triangle_chord_invalidates_the_four_girth_family():
    report = bounds(
        4,
        SQUARE + [(0, 2)],
        group="su2",
        kappa=27,
        correction_radius=Q(1, 9),
        plaquettes=LOOP,
    )
    w = report["witness"]
    assert w["actual_electric_graph_girth"] == 3
    assert w["weighted_incidence"][-1] == "0"
    assert not w["structural_cap_checks"]["entire_electric_graph_girth"]
    assert report["status"] == "INCONCLUSIVE"
    assert not report["explicit_graph_neutral_gap_verified"]
    assert not replay(report["certificate"])
    accepted = bounds(
        4,
        SQUARE + [(0, 2)],
        group="su2",
        kappa=64,
        correction_radius=Q(1, 9),
        plaquettes=LOOP,
        minimum_girth=3,
    )
    assert accepted["finite_gate_verified"]
    assert Q(accepted["witness"]["arithmetic"]["bilinear_constant"]) == Q(16, 9)


@pytest.mark.parametrize("copies", [1, 3, 9])
def test_disconnected_components_and_isolated_vertices_have_local_cost(copies):
    edges = [(4 * i + u, 4 * i + v) for i in range(copies) for u, v in SQUARE]
    loops = [[4 * i + t for t in LOOP[0]] for i in range(copies)]
    report = bounds(
        4 * copies + 2,
        edges,
        group="su2",
        kappa=27,
        correction_radius=Q(1, 9),
        plaquettes=loops,
    )
    assert report["finite_gate_verified"]
    assert report["witness"]["arithmetic"] == _square()["witness"]["arithmetic"]
    assert report["witness"]["actual_electric_graph_girth"] == 4
    assert replay(report["certificate"])


def test_forest_with_unused_vertices_has_zero_forcing():
    report = bounds(
        8,
        [(0, 1), (1, 2), (4, 5)],
        group="su2",
        kappa=1,
        correction_radius=Q(1, 100),
        weighted_incidence_cap=0,
    )
    assert report["finite_gate_verified"]
    assert report["witness"]["forest"]
    assert report["witness"]["actual_electric_graph_girth"] is None
    assert report["witness"]["forcing_by_edge"] == ["0"] * 3
    assert report["witness"]["arithmetic"]["seed_norm_upper"] == "0"
    assert replay(report["certificate"])


def test_magnetic_weights_orientation_and_exact_cap_boundary():
    plain = _square(magnetic_weights=[Q(1, 2)], weighted_incidence_cap=Q(1, 2))
    reverse = _square(plaquettes=[[-4, -3, -2, -1]], magnetic_weights=[Q(1, 2)])
    assert plain["finite_gate_verified"]
    assert plain["witness"]["arithmetic"] == reverse["witness"]["arithmetic"]
    assert plain["witness"]["weighted_incidence"] == ["1/2"] * 4
    outside = _square(weighted_incidence_cap=1 - Q(1, 10**60))
    assert not outside["witness"]["structural_cap_checks"]["weighted_incidence"]
    assert not outside["finite_gate_verified"]


def test_cycle_length_and_ambient_diameter_are_checked():
    hexagon = [(i, (i + 1) % 6) for i in range(6)]
    report = bounds(
        6,
        hexagon,
        group="su2",
        kappa=100,
        correction_radius=Q(1, 9),
        plaquettes=[[1, 2, 3, 4, 5, 6]],
    )
    checks = report["witness"]["structural_cap_checks"]
    assert checks["entire_electric_graph_girth"]
    assert not checks["cycle_length"]
    assert not checks["cycle_diameter"]
    assert report["witness"]["cycle_ambient_line_graph_diameters"] == [3]
    assert not report["finite_gate_verified"]


def _brute_girth(n, edges):
    adjacency = {frozenset(edge) for edge in edges}
    for length in range(3, n + 1):
        for cycle in permutations(range(n), length):
            if all(
                frozenset((cycle[i], cycle[(i + 1) % length])) in adjacency for i in range(length)
            ):
                return length
    return None


def test_girth_matches_independent_cycle_enumeration():
    complete4 = list(combinations(range(4), 2))
    for mask in range(1 << len(complete4)):
        edges = tuple(edge for i, edge in enumerate(complete4) if mask & (1 << i))
        assert _simple_graph_girth(4, edges) == _brute_girth(4, edges)
    rng = Random(19081)
    complete6 = list(combinations(range(6), 2))
    for _ in range(40):
        edges = tuple(edge for edge in complete6 if rng.randrange(3) == 0)
        assert _simple_graph_girth(6, edges) == _brute_girth(6, edges)


@pytest.mark.parametrize("edges", [[], [(0, 0)], [(0, 1), (0, 1)], [(0, 1), (1, 0)], [(0, 4)]])
def test_empty_self_loop_parallel_and_out_of_range_graphs_are_refused(edges):
    with pytest.raises(ValueError):
        bounds(4, edges, group="su2", kappa=27, correction_radius=Q(1, 9))


@pytest.mark.parametrize(
    "change",
    [
        {"kappa": True},
        {"kappa": 27.0},
        {"kappa": "27"},
        {"correction_radius": False},
        {"correction_radius": 0.1},
        {"decay_base": 1.0},
        {"weighted_incidence_cap": True},
        {"minimum_girth": Q(4)},
        {"minimum_girth": 4.0},
        {"max_cycle_length": True},
        {"max_cycle_diameter": 2.0},
    ],
)
def test_nonexact_scalar_inputs_are_refused(change):
    parameters = dict(group="su2", kappa=27, correction_radius=Q(1, 9))
    parameters.update(change)
    with pytest.raises(TypeError):
        family(**parameters)


@pytest.mark.parametrize(
    "change",
    [
        {"group": "SU(2)"},
        {"group": "su4"},
        {"kappa": 0},
        {"correction_radius": 0},
        {"decay_base": Q(1, 2)},
        {"weighted_incidence_cap": -1},
        {"minimum_girth": 2},
        {"max_cycle_length": 2},
        {"max_cycle_diameter": -1},
    ],
)
def test_invalid_scalar_ranges_are_refused(change):
    parameters = dict(group="su2", kappa=27, correction_radius=Q(1, 9))
    parameters.update(change)
    with pytest.raises(ValueError):
        family(**parameters)


@pytest.mark.parametrize("n", [True, 4.0, Q(4)])
def test_noninteger_vertex_count_is_refused(n):
    with pytest.raises(TypeError):
        bounds(n, SQUARE, group="su2", kappa=27, correction_radius=Q(1, 9))


@pytest.mark.parametrize("edge", [(0, True), (0, 1.0), (0, Q(1))])
def test_noninteger_edge_endpoint_is_refused(edge):
    with pytest.raises(TypeError):
        bounds(4, [edge], group="su2", kappa=27, correction_radius=Q(1, 9))


@pytest.mark.parametrize(
    "changes",
    [{"magnetic_weights": [True]}, {"magnetic_weights": [0.5]}, {"plaquettes": [[True, 2, 3, 4]]}],
)
def test_nonexact_magnetic_input_is_refused(changes):
    with pytest.raises(TypeError):
        _square(**changes)


@pytest.mark.parametrize(
    "changes",
    [
        {"magnetic_weights": []},
        {"magnetic_weights": [-1]},
        {"plaquettes": [[1, 2, 3, 0]]},
        {"plaquettes": [[1, 2, 3, -4]]},
    ],
)
def test_invalid_magnetic_input_is_refused(changes):
    with pytest.raises(ValueError):
        _square(**changes)


@pytest.mark.parametrize("value", [None, [], (), "certificate", 1, True, {}])
def test_replay_rejects_malformed_top_level_without_throwing(value):
    assert replay(value) is False


@pytest.mark.parametrize(
    "path",
    [
        ("honesty", "continuum_claim"),
        ("honesty", "yang_mills_mass_gap_claim"),
        ("honesty", "infinite_volume_claim"),
        ("honesty", "theorem_prover_verified"),
        ("honesty", "mathlib_verified"),
        ("payload", "witness", "arithmetic", "neutral_gap_lower"),
        ("payload", "witness", "arithmetic", "bilinear_constant"),
        ("payload", "witness", "arithmetic", "seed_norm_upper"),
        ("payload", "witness", "candidate"),
        ("payload", "witness", "energy_units"),
        ("payload", "witness", "gauss_constraint"),
    ],
)
def test_rehashed_forged_claims_and_arithmetic_are_rejected(path):
    certificate = deepcopy(family("su2", 27, correction_radius=Q(1, 9))["certificate"])
    node = certificate
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = True if path[0] == "honesty" else "forged"
    assert not replay(certificate)
    certificate = seal_certificate(certificate)
    assert verify_certificate_digest(certificate)
    assert not replay(certificate)


def test_graph_replay_checks_structure_after_rehash():
    certificate = deepcopy(_square()["certificate"])
    witness = certificate["payload"]["witness"]
    witness["edges"].append([0, 2])
    forged = seal_certificate(certificate)
    assert verify_certificate_digest(forged)
    assert not replay(forged)


@pytest.mark.parametrize(
    "path", [("payload",), ("payload", "witness"), ("payload", "witness", "family")]
)
def test_nested_malformed_graph_certificate_is_a_false_result(path):
    certificate = deepcopy(_square()["certificate"])
    node = certificate
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = None
    assert not replay(seal_certificate(certificate))


def test_factorization_recovers_the_gap_after_global_curvature_fails():
    parameters = dict(group="su3", kappa=45, correction_radius=Q(9, 20))
    curvature = family(**parameters)
    assert curvature["status"] == "INCONCLUSIVE"
    old = curvature["witness"]["arithmetic"]
    assert old["fixed_point_verified"]
    assert old["factorization_bound_verified"]
    assert Q(old["curvature_candidate_lower"]) < 0
    assert not old["neutral_gap_verified"]
    assert not replay(curvature["certificate"])

    factorization = family(**parameters, gap_method="factorization")
    assert factorization["status"] == "PASS"
    assert factorization["volume_uniform_finite_graph_family_verified"]
    assert factorization["volume_uniform_charged_linear_bound_verified"]
    assert factorization["volume_uniform_factorization_bound_verified"]
    a = factorization["witness"]["arithmetic"]
    assert Q(a["self_map_slack"]) == Q(357, 80000)
    assert Q(a["contraction_upper"]) == Q(327, 400)
    assert Q(a["actual_tv_influence_row_upper"]) == Q(421, 600)
    assert Q(a["seed_conditional_log_density_oscillation_upper"]) == Q(4, 225)
    assert Q(a["conditional_log_density_oscillation_upper"]) == Q(1279, 3600)
    assert a["exp_negative_lower_steps"] == 1
    assert Q(a["exp_negative_rational_lower"]) == Q(2321, 3600)
    assert Q(a["conditional_block_poincare_lower"]) == Q(415459, 1620000)
    assert a["curvature_gap_lower"] == "0"
    assert Q(a["factorization_gap_lower"]) == Q(415459, 72000)
    # Haar gap 4/3; conditional log-density oscillation <=1279/3600.
    independently_factored_gap = Q(45, 2) * Q(4, 3) * Q(179, 600) * Q(2321, 3600)
    assert independently_factored_gap == Q(415459, 72000)
    assert Q(a["neutral_gap_lower"]) == independently_factored_gap
    assert Q(a["charged_linear_lower_coefficient"]) == independently_factored_gap
    assert Q(a["charged_linear_upper_coefficient"]) == 30
    assert Q(a["vacuum_resolvent_contraction_upper"]) == Q(72000, 487459)
    assert replay(factorization["certificate"])


def test_factorization_at_kappa48_is_stronger_than_the_curvature_floor():
    parameters = dict(group="su3", kappa=48, correction_radius=Q(1, 4))
    curvature = family(**parameters, gap_method="curvature")
    factorization = family(**parameters, gap_method="factorization")
    assert Q(curvature["witness"]["arithmetic"]["neutral_gap_lower"]) == Q(47, 8)
    assert Q(factorization["witness"]["arithmetic"]["neutral_gap_lower"]) == Q(3927, 256)
    for report in (curvature, factorization):
        a = report["witness"]["arithmetic"]
        assert Q(a["curvature_gap_lower"]) == Q(47, 8)
        assert Q(a["factorization_gap_lower"]) == Q(3927, 256)
        assert Q(a["conditional_log_density_oscillation_upper"]) == Q(13, 64)
    # eta=51/128 and conditional oscillation=13/64 give this exact product.
    assert Q(32) * Q(77, 128) * Q(51, 64) == Q(3927, 256)
    assert replay(curvature["certificate"])
    assert replay(factorization["certificate"])


def test_default_su2_gap_method_preserves_the_curvature_certificate():
    parameters = dict(group="su2", kappa=27, correction_radius=Q(1, 9))
    default = family(**parameters)
    explicit = family(**parameters, gap_method="curvature")
    assert default == explicit
    assert Q(default["witness"]["arithmetic"]["neutral_gap_lower"]) == Q(1411, 324)


def test_factorization_on_a_concrete_graph_still_requires_structural_caps():
    parameters = dict(
        group="su3",
        kappa=45,
        correction_radius=Q(9, 20),
        gap_method="factorization",
        plaquettes=LOOP,
    )
    report = bounds(4, SQUARE, **parameters)
    assert report["explicit_graph_neutral_gap_verified"]
    assert replay(report["certificate"])
    assert Q(report["witness"]["arithmetic"]["neutral_gap_lower"]) == (
        Q(30) * Q(191, 600) * Q(2369, 3600)
    )
    outside = bounds(4, SQUARE, weighted_incidence_cap=Q(1, 2), **parameters)
    assert outside["witness"]["arithmetic"]["neutral_gap_verified"]
    assert not outside["witness"]["structural_caps_verified"]
    assert not outside["finite_gate_verified"]
    assert not replay(outside["certificate"])
    chord = bounds(4, SQUARE + [(0, 2)], **parameters)
    assert not chord["witness"]["structural_cap_checks"]["entire_electric_graph_girth"]
    assert not chord["finite_gate_verified"]
    assert not replay(chord["certificate"])


@pytest.mark.parametrize("method", ["", "unknown", "CURVATURE", None, True, 1])
def test_unknown_gap_method_is_refused(method):
    with pytest.raises(ValueError):
        family("su3", 45, correction_radius=Q(9, 20), gap_method=method)


@pytest.mark.parametrize("graph", [False, True])
def test_rehashed_gap_method_swap_is_not_a_valid_proof(graph):
    parameters = dict(group="su3", kappa=48, correction_radius=Q(1, 4), gap_method="factorization")
    report = bounds(4, SQUARE, plaquettes=LOOP, **parameters) if graph else family(**parameters)
    certificate = deepcopy(report["certificate"])
    witness = certificate["payload"]["witness"]
    sealed_family = witness["family"] if graph else witness
    assert sealed_family["gap_method"] == "factorization"
    sealed_family["gap_method"] = "curvature"
    forged = seal_certificate(certificate)
    assert verify_certificate_digest(forged)
    assert not replay(forged)


@pytest.mark.parametrize(
    "key",
    [
        "conditional_log_density_oscillation_upper",
        "conditional_block_poincare_lower",
        "factorization_gap_lower",
        "exp_negative_rational_lower",
        "exp_negative_lower_steps",
    ],
)
def test_rehashed_conditional_proof_inputs_are_recomputed(key):
    report = family("su3", 45, correction_radius=Q(9, 20), gap_method="factorization")
    certificate = deepcopy(report["certificate"])
    certificate["payload"]["witness"]["arithmetic"][key] = 2 if key.endswith("steps") else "2"
    forged = seal_certificate(certificate)
    assert verify_certificate_digest(forged)
    assert not replay(forged)
