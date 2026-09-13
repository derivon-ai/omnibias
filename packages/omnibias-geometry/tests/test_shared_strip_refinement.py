# SPDX-License-Identifier: Apache-2.0
"""Exact API and independent metric checks for the actual shared strip."""

from __future__ import annotations

import copy
import random
from fractions import Fraction as Q
from itertools import permutations, product
from typing import Any

import pytest
from omnibias.core.proof.certificate import seal_certificate, verify_certificate_digest
from omnibias.geometry.gauge.transfer import shared_strip_refinement as module
from omnibias.geometry.gauge.transfer.invariant_vacuum_fourier import (
    invariant_vacuum_fourier_bounds,
)
from omnibias.geometry.gauge.transfer.shared_strip_refinement import (
    replay_su2_shared_strip_refinement_certificate as replay,
)
from omnibias.geometry.gauge.transfer.shared_strip_refinement import (
    su2_shared_strip_refinement as certify,
)


def _good(**kwargs: Any) -> dict[str, Any]:
    return certify(19, correction_radius=Q(1, 9), **kwargs)


def _schur(result: dict[str, Any]) -> None:
    a = result["witness"]["arithmetic"]
    d_a, d_h = Q(a["compression_gap_lower"]), Q(a["all_physical_fiber_modes_gap_lower"])
    beta = Q(a["relative_cross_form_beta_squared_upper"])
    gap, root = Q(a["physical_gap_lower"]), Q(a["schur_sqrt_upper"])
    discriminant = (d_a - d_h)**2 + 4 * d_a * beta
    assert Q(a["schur_discriminant"]) == discriminant
    assert root**2 >= discriminant
    assert gap == (d_a + d_h - root) / 2
    assert 0 < gap < min(d_a, d_h)
    assert (d_a - gap) * (d_h - gap) >= beta * d_a


def _walk(edges: list[list[int]], path: list[int]) -> tuple[int, int]:
    directed = [edges[i - 1] if i > 0 else edges[-i - 1][::-1] for i in path]
    assert all(a[1] == b[0] for a, b in zip(directed, directed[1:], strict=False))
    return directed[0][0], directed[-1][1]


def test_thirteen_edge_geometry_and_all_signed_holonomies() -> None:
    r = _good()
    w = r["witness"]
    graph = w["graph"]
    edges = graph["edges"]
    assert graph["n_vertices"] == 10
    assert len(edges) == 13
    assert len({tuple(sorted(e)) for e in edges}) == 13
    assert len(graph["plaquettes"]) == 4
    for loop in graph["plaquettes"]:
        assert len(loop) == 4
        start, end = _walk(edges, loop)
        assert start == end
    for path in w["coarse_paths"].values():
        assert _walk(edges, path) == (2, 7)
    assert [len(w["coarse_paths"][p]) for p in ("X", "Y", "Z")] == [5, 5, 1]
    for loop in (*w["coarse_holonomies"].values(), *w["physical_fiber_holonomies"].values()):
        assert _walk(edges, loop) == (2, 2)
    coarse_edges = set().union(*(set(map(abs, p)) for p in w["coarse_paths"].values()))
    assert coarse_edges.isdisjoint(graph["eliminated_internal_edges"])
    assert coarse_edges | set(graph["eliminated_internal_edges"]) == set(range(1, 14))
    source = w["source_certificate"]["payload"]["witness"]
    assert source["actual_electric_graph_girth"] == 4
    assert source["weighted_incidence"] == ["1"] * 9 + ["2"] * 3 + ["1"]
    # The cells share precisely the center electric link, counted once.
    left = set(map(abs, graph["plaquettes"][0] + graph["plaquettes"][1]))
    right = set(map(abs, graph["plaquettes"][2] + graph["plaquettes"][3]))
    assert left & right == {graph["shared_cell_boundary_edge"]} == {11}


def test_exact_kappa19_gap_and_conditional_constants() -> None:
    r = _good()
    a = r["witness"]["arithmetic"]
    assert r["status"] == "PASS"
    assert a["fixed_point_self_map_slack"] == "6791/31668003"
    assert a["fixed_point_contraction_upper"] == "7496/9747"
    assert a["marginal_log_oscillation_upper"] == "6352/9747"
    assert a["fiber_log_oscillation_upper"] == "6928/9747"
    assert a["conditional_drift_mixed_hessian_upper"] == "1570/9747"
    gamma = Q(3, 4) * (1 - Q(6928, 9747 * 8))**8
    assert Q(a["conditional_pair_poincare_lower"]) == gamma
    assert Q(a["all_physical_fiber_modes_gap_lower"]) == Q(19, 2) * Q(11, 5) * gamma
    assert Q(a["relative_cross_form_beta_squared_upper"]) == Q(38, 5) * Q(1570, 9747)**2 / gamma
    assert a["coarse_path_kinetic_coefficients"] == ["5", "5", "1"]
    assert a["coarse_Haar_physical_gap"] == "9/2"
    assert Q(13, 2) < Q(r["physical_gap_lower"]) < Q(27, 4)
    _schur(r)
    assert replay(r["certificate"])


@pytest.mark.parametrize("steps", [1, 2, 8])
def test_exact_joint_norms_and_independent_log_ball_status(steps: int) -> None:
    r = _good(exponent_steps=steps)
    j = r["witness"]["joint_fourier"]
    k0 = Q(5269, 9747)
    k2 = Q(104, 3) * Q(4, 361) + Q(24, 9)
    exp_upper = (1 - k0 / steps)**(-steps)
    b0 = exp_upper - 1
    b2 = k2 * (1 + k0) * exp_upper
    assert j["normalizer_lower"] == "1"
    assert Q(j["twice_log_vacuum_A0_upper"]) == k0
    assert Q(j["twice_log_vacuum_A2_upper"]) == k2
    assert Q(j["density_minus_one_A0_upper"]) == b0
    assert Q(j["density_minus_one_A2_upper"]) == b2
    own_bound = max(b2, b2**2 / 4)
    assert Q(j["actual_joint_to_own_marginals_A2_upper"]) == own_bound
    assert Q(r["actual_joint_to_own_marginals_A2_upper"]) == own_bound
    assert r["actual_joint_to_own_marginals_enclosed"]
    assert j["actual_joint_to_own_marginals_enclosed"]
    assert r["actual_joint_density_bounds_verified"]
    assert r["actual_joint_log_ball_verified"] is (b0 < 1)
    assert r["status"] == "PASS"  # Separate physical conclusion survives.
    if b0 < 1:
        assert Q(j["joint_log_density_A0_upper"]) == b0 / (1 - b0)
        assert Q(j["joint_log_density_A2_upper"]) == b2 / (1 - b0)**2
        assert Q(j["joint_log_vacuum_A2_upper"]) == b2 / (2 * (1 - b0)**2)
    else:
        assert j["joint_log_density_A0_upper"] is None
        assert j["joint_log_density_A2_upper"] is None
        assert j["failed_constraints"] == ["joint_A0_log_ball"]
    assert replay(r["certificate"])


def test_quadratic_coarse_marginal_branch() -> None:
    r = certify(64, correction_radius=Q(1, 1000))
    a = r["witness"]["arithmetic"]
    expected = 16 * Q(1, 1024)**2 / 9 + Q(26, 3000)
    assert a["marginal_bound_method"] == "quadratic_Haar_cancellation"
    assert Q(a["marginal_log_oscillation_upper"]) == expected
    _schur(r)


@pytest.mark.parametrize("offset,passes", [(Q(-1, 10**30), False), (Q(0), True),
                                           (Q(1, 10**30), True)])
def test_exact_fixed_point_boundary(offset: Q, passes: bool) -> None:
    r = certify(Q(64, 3), correction_radius=Q(3, 64) + offset)
    assert r["actual_vacuum_embedding_verified"] is passes
    assert r["physical_finite_graph_gap_verified"] is passes
    if not offset:
        assert r["witness"]["arithmetic"]["fixed_point_self_map_slack"] == "0"
    assert replay(r["certificate"])


@pytest.mark.parametrize("kappa,radius,steps,failure,embedding", [
    (18, Q(1, 9), 8, "actual_vacuum_fixed_point", False),
    (64, Q(1, 4), 1, "strict_rational_exponential_domain", True),
    (64, Q(1, 4), 8, "relative_form_coercivity", True),
])
def test_refusal_keeps_the_exact_earned_tier(
    kappa: int, radius: Q, steps: int, failure: str, embedding: bool,
) -> None:
    r = certify(kappa, correction_radius=radius, exponent_steps=steps)
    assert r["status"] == "INCONCLUSIVE"
    assert r["actual_vacuum_embedding_verified"] is embedding
    assert failure in r["witness"]["failed_constraints"]
    assert r["physical_gap_lower"] is None
    assert not r["physical_finite_graph_gap_verified"]
    if not embedding:
        assert not r["actual_joint_density_bounds_verified"]
        assert r["witness"]["joint_fourier"]["twice_log_vacuum_A0_upper"] is None
    if not r["actual_joint_density_bounds_verified"]:
        assert not r["actual_joint_to_own_marginals_enclosed"]
        assert r["actual_joint_to_own_marginals_A2_upper"] is None
        assert r["witness"]["joint_fourier"]["actual_joint_to_own_marginals_A2_upper"] is None
    assert replay(r["certificate"])


def test_source_gap_and_factorization_are_not_inputs(monkeypatch: pytest.MonkeyPatch) -> None:
    original = invariant_vacuum_fourier_bounds
    expected = _good()

    def stripped(*args: Any, **kwargs: Any) -> dict[str, Any]:
        r = original(*args, **kwargs)
        r["witness"] = copy.deepcopy(r["witness"])
        r["status"] = "INCONCLUSIVE"
        r["finite_gate_verified"] = False
        for values in (r["witness"]["arithmetic"], r["witness"]["family"]["arithmetic"]):
            for key in tuple(values):
                if "gap" in key or "factorization" in key:
                    del values[key]
        return r

    monkeypatch.setattr(module, "invariant_vacuum_fourier_bounds", stripped)
    assert _good() == expected


def _add(*vectors: tuple[Q, ...]) -> tuple[Q, ...]:
    return tuple(sum(row, Q(0)) for row in zip(*vectors, strict=True))


def _norm(v: tuple[Q, ...]) -> Q:
    return sum((x * x for x in v), Q(0))


def test_fiber_metric_bound_with_independent_exact_rotation_frames() -> None:
    """Exact rational probes of the covector inequality, not tests of all spins."""
    rng = random.Random(20260911)
    rotations = []
    for perm in permutations(range(3)):
        inversions = sum(perm[i] > perm[j] for i in range(3) for j in range(i + 1, 3))
        for signs in product((-1, 1), repeat=3):
            if (-1)**inversions * signs[0] * signs[1] * signs[2] == 1:
                rotations.append((perm, signs))

    def rotate(v: tuple[Q, ...]) -> tuple[Q, ...]:
        perm, signs = rng.choice(rotations)
        return tuple(signs[i] * v[perm[i]] for i in range(3))

    for _ in range(128):
        p, q, s, t = [tuple(Q(rng.randint(-9, 9), 7) for _ in range(3)) for _ in range(4)]
        # Left and right partial frames are kept as different orthogonal maps.
        energy = (3 * _norm(p) + _norm(q) + 3 * _norm(s) + _norm(t)
                  + _norm(_add(p, q)) + _norm(_add(p, rotate(q)))
                  + _norm(_add(s, t)) + _norm(_add(s, rotate(t)))
                  + _norm(_add(p, q, s, t)))
        assert energy >= Q(11, 5) * (_norm(q) + _norm(t))
    # Aligned frames and antisymmetric fibers saturate the sharp scalar floor.
    q = (Q(1), Q(2), Q(-3))
    t = tuple(-x for x in q)
    p = tuple(-Q(2, 5) * x for x in q)
    s = tuple(-x for x in p)
    energy = (3 * _norm(p) + _norm(q) + 3 * _norm(s) + _norm(t)
              + 2 * _norm(_add(p, q)) + 2 * _norm(_add(s, t)))
    assert energy == Q(11, 5) * (_norm(q) + _norm(t))
    assert _norm(_add(p, q, s, t)) == 0


def test_gauss_weight_inequality_for_all_small_strip_admissible_labels() -> None:
    # Enumerate 0/half spins on 13 edges independently of the source gate.
    # At each trivalent/bivalent vertex singlet admissibility for these labels
    # is exactly an even number (zero or two) of half-spin incident edges.
    edges = _good()["witness"]["graph"]["edges"]
    vertices = [[i for i, edge in enumerate(edges) if v in edge] for v in range(10)]
    count = 0
    for labels in product((0, 1), repeat=13):
        if not any(labels) or any(sum(labels[i] for i in pack) % 2 for pack in vertices):
            continue
        count += 1
        energy = Q(3, 4) * sum(labels)
        chord_spin = Q(sum(labels[i] for i in (8, 9, 11, 12)), 2)
        assert chord_spin >= Q(1, 2)
        assert energy >= max(Q(3), 3 * chord_spin)
        assert (1 + 2 * chord_spin)**2 <= 3 * energy * chord_spin
    assert count == 15  # Four independent cycle-parity degrees of freedom.


@pytest.mark.parametrize("path,value", [
    (("arithmetic", "physical_vertical_kinetic_coefficient"), "5/2"),
    (("arithmetic", "physical_gap_lower"), "100"),
    (("arithmetic", "relative_cross_form_beta_squared_upper"), "0"),
    (("arithmetic", "fiber_log_oscillation_upper"), "0"),
    (("arithmetic", "coarse_path_kinetic_coefficients"), ["3", "3", "1"]),
    (("joint_fourier", "density_minus_one_A0_upper"), "0"),
    (("joint_fourier", "twice_log_vacuum_A2_upper"), "0"),
    (("joint_fourier", "joint_log_density_A2_upper"), "0"),
    (("joint_fourier", "actual_joint_to_own_marginals_A2_upper"), "0"),
    (("joint_fourier", "connected_interaction_A2_bound_verified"), True),
    (("graph", "shared_cell_boundary_edge"), 10),
    (("graph", "n_vertices"), 9),
    (("coarse_paths", "Z"), [10]),
    (("inputs", "exponent_steps"), 2),
    (("inputs", "sqrt_bits"), 4),
])
def test_resealed_witness_tampering_is_rejected(path: tuple[str, ...], value: Any) -> None:
    certificate = copy.deepcopy(_good()["certificate"])
    data = certificate["payload"]["witness"]
    for key in path[:-1]:
        data = data[key]
    data[path[-1]] = value
    certificate = seal_certificate(certificate)
    assert verify_certificate_digest(certificate)
    assert not replay(certificate)


def test_resealed_nested_source_forgery_is_rejected() -> None:
    c = copy.deepcopy(_good()["certificate"])
    s = c["payload"]["witness"]["source_certificate"]
    s["payload"]["witness"]["arithmetic"]["self_map_slack"] = "100"
    c["payload"]["witness"]["source_certificate"] = seal_certificate(s)
    c = seal_certificate(c)
    assert verify_certificate_digest(c)
    assert verify_certificate_digest(c["payload"]["witness"]["source_certificate"])
    assert not replay(c)


@pytest.mark.parametrize("key", ["all_scale_refinement_claim", "continuum_claim",
                                 "yang_mills_mass_gap_claim", "theorem_prover_verified",
                                 "independent_cell_replacement_verified"])
def test_resealed_parent_or_formal_flag_is_rejected(key: str) -> None:
    c = copy.deepcopy(_good()["certificate"])
    c["honesty"][key] = True
    c = seal_certificate(c)
    assert verify_certificate_digest(c)
    assert not replay(c)


@pytest.mark.parametrize("bad", [None, [], True, 1, "", {}, {"digest": "sha256:bad"}])
def test_malformed_replay_refuses(bad: Any) -> None:
    assert not replay(bad)


@pytest.mark.parametrize("key,value", [
    ("kappa", True), ("kappa", 19.0), ("kappa", "19"), ("kappa", 0),
    ("correction_radius", False), ("correction_radius", 0.1), ("correction_radius", 0),
    ("exponent_steps", True), ("exponent_steps", 8.0), ("exponent_steps", 0),
    ("sqrt_bits", True), ("sqrt_bits", 64.0), ("sqrt_bits", -1),
])
def test_strict_exact_input_guards(key: str, value: Any) -> None:
    arguments: dict[str, Any] = {
        "kappa": 19, "correction_radius": Q(1, 9), "exponent_steps": 8, "sqrt_bits": 64,
    }
    arguments[key] = value
    with pytest.raises((TypeError, ValueError)):
        certify(**arguments)


def test_scopes_never_promote_the_parent_or_independence() -> None:
    r = _good()
    for key in ("independent_cell_replacement_verified", "connected_interaction_A2_bound_verified",
                "all_scale_refinement_claim", "coarse_wilson_family_closed",
                "infinite_volume_claim", "uniform_in_a_claim", "continuum_claim",
                "yang_mills_claim", "yang_mills_mass_gap_claim", "theorem_prover_verified",
                "mathlib_verified"):
        assert r[key] is False
    assert "physical thirteen-edge" in r["witness"]["gap_scope"]
    assert "not be separately central or independent" in r["witness"]["physical_coordinate_scope"]
