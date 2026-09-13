# SPDX-License-Identifier: Apache-2.0
"""SU(3) thresholds, independent representation checks, graph scope and replay."""

from copy import deepcopy
from fractions import Fraction as Q

import pytest
from omnibias.core.proof.certificate import seal_certificate, verify_certificate_digest
from omnibias.geometry.gauge.transfer.su3_vacuum_fourier import (
    replay_su3_vacuum_fourier_certificate,
    su3_vacuum_fourier_bounds,
    su3_vacuum_fourier_family,
)

SQUARE = [(0, 1), (1, 2), (2, 3), (3, 0)]
LOOP = [[1, 2, 3, 4]]


def test_exact_cubic_threshold_normalization_and_charge_scope():
    report = su3_vacuum_fourier_family()
    assert report["status"] == "PASS"
    assert report["volume_uniform_finite_graph_family_verified"]
    assert report["volume_uniform_neutral_gap_claim"]
    assert report["volume_uniform_factorization_bound_verified"]
    assert report["volume_uniform_charged_linear_bound_verified"]
    assert not report["explicit_graph_neutral_gap_verified"]
    assert not report["explicit_graph_charged_linear_bound_verified"]
    assert replay_su3_vacuum_fourier_certificate(report["certificate"])
    witness = report["witness"]
    a = witness["arithmetic"]
    assert witness["group"] == "SU(3)"
    assert witness["fundamental_casimir"] == "4/3"
    assert witness["haar_ricci"] == "3/4"
    assert witness["lie_algebra_dimension"] == 8
    assert witness["normalization"] == (
        "aH=kappa/2*sum(C_e)+2/kappa*sum(v_p*(3-ReTr(U_p)))"
    )
    assert witness["energy_units"] == "dimensionless aH"
    assert witness["charged_scope"]["gauss_constraint"] == "imposed at every graph vertex"
    assert witness["charged_scope"]["dynamical_fundamental_matter"] is False
    assert witness["charged_scope"]["flux_absorbing_boundary"] is False
    assert "zeta=exp(2*pi*i/3)" in witness["charged_scope"]["cut_center_action"]
    expected = {
        "forcing_norm_upper": Q(1, 64),
        "bilinear_constant": 12,
        "quadratic_correction": Q(1, 192),
        "self_map_slack": 0,
        "contraction_upper": Q(1, 2),
        "actual_log_vacuum_hessian_row_upper": Q(1, 48),
        "curvature_candidate_lower": Q(17, 24),
        "neutral_gap_lower": 102,
        "actual_tv_influence_row_upper": Q(1, 32),
        "actual_factorization_constant_upper": Q(32, 31),
        "charged_linear_lower_coefficient": 102,
        "charged_linear_upper_coefficient": 192,
    }
    for key, value in expected.items():
        assert Q(a[key]) == value
    assert a["charged_linear_lower_verified"]
    assert a["charged_linear_upper_verified"]
    assert not a["exponential_hessian_tail_verified"]
    for flag in (
        "infinite_volume_claim",
        "uniform_in_a_claim",
        "continuum_claim",
        "yang_mills_claim",
        "static_confinement_claim",
        "string_tension_claim",
        "theorem_prover_verified",
        "mathlib_verified",
        "analytic_implication_formally_verified",
    ):
        assert report[flag] is False


def test_weighted_norm_tail_and_strict_contraction_boundary():
    report = su3_vacuum_fourier_family(576, 4, decay_base=2, tail_radius=3)
    a = report["witness"]["arithmetic"]
    assert report["finite_gate_verified"]
    assert a["exponential_hessian_tail_verified"]
    assert a["actual_log_vacuum_hessian_tail_row_upper"] == "1/384"
    assert a["actual_tv_influence_tail_upper"] == "1/256"
    assert a["neutral_gap_lower"] == "204"
    assert replay_su3_vacuum_fourier_certificate(report["certificate"])
    assert not su3_vacuum_fourier_family(10**6, 4, radius=Q(1, 24))[
        "finite_gate_verified"
    ]


@pytest.mark.parametrize("kappa", [1, 16, 64, 287])
def test_failed_sufficient_gate_earns_no_gap_or_charged_lower(kappa):
    report = su3_vacuum_fourier_family(kappa)
    assert report["status"] == "INCONCLUSIVE"
    assert not report["volume_uniform_neutral_gap_claim"]
    assert not report["volume_uniform_charged_linear_bound_verified"]
    assert not report["volume_uniform_factorization_bound_verified"]
    a = report["witness"]["arithmetic"]
    assert a["neutral_gap_lower"] == "0"
    assert a["charged_linear_lower_coefficient"] == "0"
    assert a["actual_log_vacuum_hessian_row_upper"] is None
    assert a["charged_linear_upper_verified"]
    assert Q(a["charged_linear_upper_coefficient"]) == Q(2 * kappa, 3)
    assert verify_certificate_digest(report["certificate"])
    assert not replay_su3_vacuum_fourier_certificate(report["certificate"])


def test_weighted_adjacent_cycles_and_orientations():
    edges = SQUARE + [(1, 4), (4, 5), (5, 2)]
    loops = LOOP + [[5, 6, 7, -2]]
    report = su3_vacuum_fourier_bounds(
        6, edges, plaquettes=loops, magnetic_weights=[Q(1, 2), Q(3, 2)]
    )
    w = report["witness"]
    assert w["cycle_ambient_line_graph_diameters"] == [2, 2]
    assert w["weighted_incidence"] == ["1/2", "2", "1/2", "1/2", "3/2", "3/2", "3/2"]
    assert w["forcing_by_edge"] == [
        "1/512", "1/128", "1/512", "1/512", "3/512", "3/512", "3/512"
    ]
    reverse = su3_vacuum_fourier_bounds(
        6,
        edges,
        plaquettes=[[-t for t in reversed(loop)] for loop in loops],
        magnetic_weights=[Q(1, 2), Q(3, 2)],
    )
    assert reverse["witness"]["arithmetic"] == w["arithmetic"]
    assert report["explicit_graph_charged_linear_bound_verified"]
    assert replay_su3_vacuum_fourier_certificate(report["certificate"])


@pytest.mark.parametrize("copies", [1, 3, 20])
def test_disconnected_volumes_and_free_edge_do_not_make_an_extensive_forcing(copies):
    edges = [(4 * i + u, 4 * i + v) for i in range(copies) for u, v in SQUARE]
    loops = [[4 * i + t for t in LOOP[0]] for i in range(copies)]
    edges.append((4 * copies, 4 * copies + 1))
    report = su3_vacuum_fourier_bounds(4 * copies + 2, edges, plaquettes=loops)
    w = report["witness"]
    assert w["arithmetic"]["forcing_norm_upper"] == "1/256"
    assert w["forcing_by_edge"][-1] == "0"
    assert report["finite_gate_verified"]
    assert "componentwise" in w["family"]["disconnected_components"]
    assert replay_su3_vacuum_fourier_certificate(report["certificate"])


def test_ambient_shortcuts_and_nonquadrilateral_forcing():
    hexagon = [(i, (i + 1) % 6) for i in range(6)]
    loop = [[1, 2, 3, 4, 5, 6]]
    plain = su3_vacuum_fourier_bounds(6, hexagon, plaquettes=loop, decay_base=2)
    # Opposite vertex chords make every pair of cycle edges at most two apart.
    shortcut = su3_vacuum_fourier_bounds(
        6, hexagon + [(0, 3), (1, 4), (2, 5)], plaquettes=loop, decay_base=2
    )
    assert plain["witness"]["cycle_ambient_line_graph_diameters"] == [3]
    assert shortcut["witness"]["cycle_ambient_line_graph_diameters"] == [2]
    assert Q(plain["witness"]["forcing_by_edge"][0]) == Q(3**6 * 8, 20736)
    assert Q(shortcut["witness"]["forcing_by_edge"][0]) == Q(3**6 * 4, 20736)
    assert shortcut["witness"]["forcing_by_edge"][-3:] == ["0"] * 3


def test_graph_and_coarse_family_are_independent_verified_claims():
    loose = su3_vacuum_fourier_bounds(
        4, SQUARE, plaquettes=LOOP, weighted_incidence_cap=100
    )
    assert loose["finite_gate_verified"]
    assert loose["explicit_graph_charged_linear_bound_verified"]
    assert not loose["volume_uniform_charged_linear_bound_verified"]
    assert replay_su3_vacuum_fourier_certificate(loose["certificate"])
    false_cap = su3_vacuum_fourier_bounds(
        4, SQUARE, plaquettes=LOOP, weighted_incidence_cap=Q(1, 2)
    )
    assert not false_cap["finite_gate_verified"]
    assert false_cap["explicit_graph_charged_linear_bound_verified"]
    assert not false_cap["volume_uniform_charged_linear_bound_verified"]
    assert not replay_su3_vacuum_fourier_certificate(false_cap["certificate"])
    for kwargs in [{"max_cycle_length": 3}, {"max_cycle_diameter": 1}]:
        assert not su3_vacuum_fourier_bounds(
            4, SQUARE, plaquettes=LOOP, **kwargs
        )["finite_gate_verified"]


@pytest.mark.parametrize(
    "path,value",
    [
        (("payload", "witness", "arithmetic", "neutral_gap_lower"), "103"),
        (("payload", "witness", "arithmetic", "charged_linear_lower_coefficient"), "103"),
        (("payload", "witness", "arithmetic", "charged_linear_upper_coefficient"), "191"),
        (("payload", "witness", "arithmetic", "actual_tv_influence_row_upper"), "1/64"),
        (("payload", "witness", "arithmetic", "forcing_norm_upper"), "1/256"),
        (("payload", "witness", "fundamental_casimir"), "3/4"),
        (("payload", "witness", "normalization"), "physical H"),
        (("payload", "witness", "charged_scope", "gauss_constraint"), "interior only"),
        (("payload", "witness", "charged_scope", "dynamical_fundamental_matter"), True),
        (("payload", "witness", "charged_scope", "cut_center_action"), "minus one"),
        (("honesty", "continuum_claim"), True),
        (("honesty", "yang_mills_claim"), True),
        (("meta", "scope"), "continuum theorem"),
    ],
)
def test_rehashing_false_bounds_and_scope_cannot_make_a_certificate(path, value):
    cert = deepcopy(su3_vacuum_fourier_family()["certificate"])
    target = cert
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    forged = seal_certificate(cert)
    assert verify_certificate_digest(forged)
    assert not replay_su3_vacuum_fourier_certificate(forged)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"kappa": 0}, {"kappa": -1}, {"kappa": 288.0}, {"kappa": True},
        {"radius": 0}, {"radius": 0.01}, {"decay_base": Q(1, 2)},
        {"tail_radius": -1}, {"tail_radius": True}, {"tail_radius": Q(1)},
        {"weighted_incidence_cap": -1}, {"weighted_incidence_cap": 4.0},
        {"max_cycle_length": 1}, {"max_cycle_length": True},
        {"max_cycle_diameter": -1}, {"max_cycle_diameter": 2.0},
    ],
)
def test_exact_parameter_validation(kwargs):
    with pytest.raises((TypeError, ValueError)):
        su3_vacuum_fourier_family(**kwargs)


@pytest.mark.parametrize(
    "n,edges,kwargs",
    [
        (2, [], {}),
        (2, [(0, 0)], {}),
        (2, [(0, 2)], {}),
        (4, SQUARE, {"plaquettes": [[1, 2, -2, -1]]}),
        (4, SQUARE, {"plaquettes": LOOP, "magnetic_weights": []}),
        (4, SQUARE, {"plaquettes": LOOP, "magnetic_weights": [-1]}),
        (4, SQUARE, {"plaquettes": LOOP, "magnetic_weights": [True]}),
        (4, SQUARE, {"plaquettes": LOOP, "magnetic_weights": [1.0]}),
    ],
)
def test_graph_and_weight_validation(n, edges, kwargs):
    with pytest.raises((TypeError, ValueError)):
        su3_vacuum_fourier_bounds(n, edges, **kwargs)


@pytest.mark.parametrize("certificate", [None, [], (), "certificate", 3, True, {}, {"payload": []}])
def test_malformed_replay_returns_false(certificate):
    assert not replay_su3_vacuum_fourier_certificate(certificate)


def test_su3_casimir_and_fusion_normalization_independent_finite_regression():
    # Finite representation identities check constants; the all-irrep proof
    # is the written Casimir/Minkowski argument, not this bounded test.
    def casimir(label):
        p, q = label
        return Q(p * p + p * q + q * q + 3 * p + 3 * q, 3)

    assert casimir((1, 0)) == casimir((0, 1)) == Q(4, 3)
    assert casimir((1, 1)) == 3
    for p in range(12):
        for q in range(12):
            if p or q:
                assert casimir((p, q)) >= Q(4, 3)
    decompositions = [
        ((1, 0), (1, 0), [(2, 0), (0, 1)]),
        ((1, 0), (0, 1), [(1, 1), (0, 0)]),
        ((1, 1), (1, 1), [(0, 0), (1, 1), (1, 1), (3, 0), (0, 3), (2, 2)]),
    ]
    for left, right, outputs in decompositions:
        a, b = casimir(left), casimir(right)
        for output in outputs:
            excess = casimir(output) - a - b
            assert excess <= 0 or excess * excess <= 4 * a * b
