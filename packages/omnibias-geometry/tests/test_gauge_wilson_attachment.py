# SPDX-License-Identifier: Apache-2.0
"""Independent attachment geometry, rational derivatives and replay regressions."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from fractions import Fraction as Q
from random import Random
from typing import Any, cast

import mpmath as mp  # type: ignore[import-untyped]
import pytest
from omnibias.core.proof.certificate import seal_certificate, verify_certificate_digest
from omnibias.geometry.gauge.transfer.wilson_attachment import (
    replay_su2_wilson_attachment_certificate as replay,
)
from omnibias.geometry.gauge.transfer.wilson_attachment import su2_wilson_attachment as attachment

Edge = tuple[int, int]
Face = list[int]
Quaternion = tuple[Q, Q, Q, Q]


def _edges(face: Face) -> list[Edge]:
    return [(min(face[i], face[(i+1) % 4]), max(face[i], face[(i+1) % 4])) for i in range(4)]


def _pages(count: int) -> list[Face]:
    return [[0, 1, 2 + 2*i, 3 + 2*i] for i in range(count)]


def _strip(count: int) -> list[Face]:
    return [[i, i+1, count+2+i, count+1+i] for i in range(count)]


def _attached_cube() -> tuple[list[Edge], list[Face]]:
    return _edges([0, 1, 3, 2]), [
        [0, 1, 5, 4], [1, 3, 7, 5], [3, 2, 6, 7], [2, 0, 4, 6], [4, 5, 7, 6],
    ]


def _arithmetic(report: dict[str, Any]) -> dict[str, Any]:
    return cast(dict[str, Any], report["witness"]["arithmetic"])


def test_public_transfer_exports_match_the_audited_apis() -> None:
    from omnibias.geometry.gauge import transfer

    assert transfer.su2_wilson_attachment is attachment
    assert transfer.replay_su2_wilson_attachment_certificate is replay


@pytest.mark.parametrize("fresh", (1, 2, 3, 4))
def test_each_possible_positive_fresh_edge_count_earns_same_trial(fresh: int) -> None:
    face = [0, 1, 2, 3]
    old = _edges(face)[:4-fresh]
    row = attachment(Q(1, 100), old, [face], action_threshold=Q(1, 10))
    a = _arithmetic(row)
    assert row["status"] == "PASS" and replay(row["certificate"])
    assert row["witness"]["stages"][0]["fresh_edge_count"] == fresh
    assert a["vacuum_increment_upper"] == "3"
    assert a["actual_added_action_mean_upper"] == "3/200"
    assert a["large_action_probability_upper"] == "3/20"
    assert a["good_localized_vacuum_norm_squared_lower"] == "7/10"
    assert a["vacuum_subtracted_bad_support_floor"] == "7"
    assert Q(a["universal_ims_error_upper"]) == Q(484, 245)
    assert Q(a["added_action_gradient_square_mean_upper"]) == Q(2391, 40000)
    assert Q(a["vacuum_localization_form_cost_upper"]) == Q(289311, 980000)
    assert row["positive_bad_support_floor_verified"]


def test_alternating_old_and_new_edges_are_allowed() -> None:
    row = attachment(1, [(0, 1), (2, 3)], [[0, 1, 2, 3]])
    assert row["status"] == "PASS"
    assert row["witness"]["stages"][0]["fresh_edges"] == [[0, 3], [1, 2]]
    assert _arithmetic(row)["vacuum_increment_upper"] == "3"


@pytest.mark.parametrize("count", (2, 3, 9))
def test_sequential_composition_reuses_previously_new_edges(count: int) -> None:
    faces = _strip(count)
    row = attachment(Q(1, 100), [(0, count+1)], faces)
    assert row["status"] == "PASS" and replay(row["certificate"])
    a = _arithmetic(row)
    assert a["face_count"] == a["fresh_stage_count"] == count
    assert a["max_added_face_edge_incidence"] == 2
    assert Q(a["vacuum_increment_upper"]) == 3*count
    assert [stage["fresh_edge_count"] for stage in row["witness"]["stages"]] == [3]*count
    first_new = {tuple(edge) for edge in row["witness"]["stages"][0]["fresh_edges"]}
    assert first_new & set(_edges(faces[1]))


@pytest.mark.parametrize("count", (4, 5, 7))
def test_star_and_non_cubic_incidence_use_the_actual_graph_count(count: int) -> None:
    row = attachment(Q(1, 100), [(0, 1)], _pages(count), action_threshold=1)
    a = _arithmetic(row)
    assert row["status"] == "PASS"
    assert a["max_added_face_edge_incidence"] == count
    assert Q(a["universal_ims_error_upper"]) == Q(968, 49)*count/100
    assert row["ambient_graph_membership_verified"] is False
    assert row["full_cubic_block_cheap_increment_verified"] is False


@pytest.mark.parametrize("coupling", (Q(1, 100), Q(1), Q(4, 3), Q(10)))
def test_closed_cube_top_is_not_falsely_earned_as_a_fresh_attachment(coupling: Q) -> None:
    old, faces = _attached_cube()
    row = attachment(coupling, old, faces)
    a = _arithmetic(row)
    assert row["status"] == "INCONCLUSIVE" and replay(row["certificate"])
    assert [stage["fresh_edge_count"] for stage in row["witness"]["stages"]] == [3, 2, 2, 1, 0]
    assert row["witness"]["unresolved_cheap_stages"] == [4]
    assert row["cheap_attachment_route_verified"] is False
    assert row["complete_normalized_attachment_compression_verified"] is False
    assert row["actual_vacuum_increment_bound_verified"]
    assert a["initially_touched_face_count"] == 5
    sequential = 4*min(Q(3), 4/coupling) + 8/coupling
    assert Q(a["sequential_increment_upper"]) == sequential
    assert Q(a["global_haar_increment_upper"]) == 20/coupling
    assert Q(a["vacuum_increment_upper"]) == min(sequential, 20/coupling)


def test_closed_all_old_face_uses_eight_over_kappa_not_haar_four() -> None:
    face = [0, 1, 2, 3]
    row = attachment(Q(1, 100), _edges(face), [face])
    a = _arithmetic(row)
    assert row["status"] == "INCONCLUSIVE" and replay(row["certificate"])
    assert a["initially_touched_face_count"] == 0
    assert a["vacuum_increment_upper"] == "800"
    assert a["actual_added_action_mean_upper"] == "4"
    assert a["large_action_probability_upper"] == "1"
    assert a["good_localized_vacuum_norm_squared_lower"] == "0"
    assert a["good_normalized_energy_above_vacuum_upper"] is None
    assert not row["positive_bad_support_floor_verified"]


@pytest.mark.parametrize("coupling", (Q(1, 1000), Q(1, 7), Q(4, 3), Q(2), Q(100)))
def test_all_exact_arithmetic_against_separately_counted_geometry(coupling: Q) -> None:
    examples = [([], [[0, 1, 2, 3]]), ([(0, 1)], _pages(7)), _attached_cube()]
    for old, faces in examples:
        known = {tuple(sorted(edge)) for edge in old}
        initial = set(known)
        incidence: Counter[Edge] = Counter()
        cheap = 0
        touched = 0
        for face in faces:
            edges = set(_edges(face))
            cheap += bool(edges-known)
            touched += bool(edges-initial)
            incidence.update(edges)
            known.update(edges)
        count, d = len(faces), max(incidence.values())
        for threshold in (Q(1, 7), Q(1), Q(4*count)):
            row = attachment(coupling, old, faces, action_threshold=threshold)
            a = _arithmetic(row)
            increment = min(cheap*min(Q(3), 4/coupling)+8*(count-cheap)/coupling,
                            (4*touched+8*(count-touched))/coupling)
            mean = min(Q(4*count), coupling*increment/2)
            point = min(coupling*increment/2, Q(2*count))
            gradient = d*point*(4-point/count)
            norm = max(Q(0), 1-2*mean/threshold)
            cost = coupling*Q(242, 49)*gradient/threshold**2
            assert Q(a["vacuum_increment_upper"]) == increment
            assert Q(a["actual_added_action_mean_upper"]) == mean
            assert Q(a["large_action_probability_upper"]) == min(Q(1), mean/threshold)
            assert Q(a["added_action_gradient_square_mean_upper"]) == gradient
            assert Q(a["good_localized_vacuum_norm_squared_lower"]) == norm
            assert Q(a["vacuum_localization_form_cost_upper"]) == cost
            assert a["good_normalized_energy_above_vacuum_upper"] == (str(cost/norm) if norm else None)
            assert Q(a["vacuum_subtracted_bad_support_floor"]) == threshold/coupling-increment
            assert Q(a["bad_floor_minus_ims"]) == threshold/coupling-increment-Q(968, 49)*coupling*d/threshold


@pytest.mark.parametrize("t", (Q(1, 16), Q(1, 64), Q(1, 1000)))
def test_weak_one_face_subtraction_and_ims_have_distinct_scalings(t: Q) -> None:
    row = attachment(t**5, [(0, 1), (1, 2), (2, 3)], [[0, 1, 2, 3]], action_threshold=t**4)
    a = _arithmetic(row)
    assert Q(a["vacuum_subtracted_bad_support_floor"]) == 1/t-3
    assert Q(a["universal_ims_error_upper"]) == Q(968, 49)*t
    assert Q(a["bad_floor_minus_ims"]) > 0
    assert Q(a["large_action_probability_upper"]) == 3*t/2
    assert Q(a["good_localized_vacuum_norm_squared_lower"]) == 1-3*t


def test_normalizes_unoriented_old_edges_without_mutating_inputs() -> None:
    old, faces = [[2, 1], [0, 1], [1, 0]], [[0, 1, 2, 3]]
    original = deepcopy((old, faces))
    row = attachment(1, old, faces)
    assert (old, faces) == original
    assert row["witness"]["inputs"]["old_edges"] == [[0, 1], [1, 2]]
    assert row["certificate"] == attachment(1, [(0, 1), (1, 2)], faces)["certificate"]


@pytest.mark.parametrize("bad", (True, False, 0.1, 1.0, "1", None, [], {}))
@pytest.mark.parametrize("field", ("kappa", "action_threshold"))
def test_non_exact_numeric_inputs_are_refused(field: str, bad: Any) -> None:
    kwargs: dict[str, Any] = {"kappa": 1, "old_edges": [], "added_faces": [[0, 1, 2, 3]], "action_threshold": 1}
    kwargs[field] = bad
    with pytest.raises((TypeError, ValueError)):
        attachment(**kwargs)


@pytest.mark.parametrize("coupling,threshold", [(0, 1), (-1, 1), (1, 0), (1, -1), (1, Q(401, 100))])
def test_empty_numeric_domains_are_refused(coupling: int, threshold: int | Q) -> None:
    with pytest.raises(ValueError):
        attachment(coupling, [], [[0, 1, 2, 3]], action_threshold=threshold)


@pytest.mark.parametrize("old,faces", [
    ([], []), ([(0, 0)], [[0, 1, 2, 3]]), ([(0, 1, 2)], [[0, 1, 2, 3]]),
    ([(True, 1)], [[0, 1, 2, 3]]), ([(0.0, 1)], [[0, 1, 2, 3]]),
    ([], [[0, 1, 2]]), ([], [[0, 1, 2, 0]]), ([], [[0, 1, 2, 3, 4]]),
    ([], [[0, 1, 2, True]]), ([], [[0, 1, 2, 3.0]]),
    ([], [[0, 1, 2, 3], [1, 2, 3, 0]]),
    ([], [[0, 1, 2, 3], [0, 3, 2, 1]]),
    (None, [[0, 1, 2, 3]]), ([], None), ([], [None]),
])
def test_malformed_geometry_and_duplicate_faces_refuse(old: Any, faces: Any) -> None:
    with pytest.raises((TypeError, ValueError)):
        attachment(1, old, faces)


@pytest.mark.parametrize("value", (None, [], (), "certificate", 1, True, {}, {"payload": {}}))
def test_malformed_replay_cannot_earn_a_certificate(value: Any) -> None:
    assert replay(value) is False


@pytest.mark.parametrize("tamper", (
    "coupling", "threshold", "old_edges", "added_faces", "stage", "incidence", "cheap",
    "increment", "gradient", "ims", "norm", "bad_floor", "family", "compression",
    "dependent", "parent", "formal", "metadata", "claim", "unknown",
))
def test_resealed_inputs_arithmetic_and_honesty_forgery_is_refused(tamper: str) -> None:
    cert = deepcopy(attachment(Q(1, 100), [(0, 1)], _pages(5))["certificate"])
    witness = cert["payload"]["witness"]
    if tamper == "coupling":
        witness["inputs"]["kappa"] = "1/99"
    elif tamper == "threshold":
        witness["inputs"]["action_threshold"] = "1/2"
    elif tamper == "old_edges":
        witness["inputs"]["old_edges"].append([2, 3])
    elif tamper == "added_faces":
        witness["inputs"]["added_faces"].reverse()
    elif tamper == "stage":
        witness["stages"][0]["fresh_edge_count"] = 0
    elif tamper == "incidence":
        witness["edge_incidence"][0]["added_face_count"] = 4
    elif tamper == "cheap":
        witness["arithmetic"]["fresh_stage_count"] = 4
    elif tamper in ("increment", "gradient", "ims", "norm", "bad_floor"):
        key = {"increment": "vacuum_increment_upper", "gradient": "added_action_gradient_square_mean_upper",
               "ims": "universal_ims_error_upper", "norm": "good_localized_vacuum_norm_squared_lower",
               "bad_floor": "vacuum_subtracted_bad_support_floor"}[tamper]
        witness["arithmetic"][key] = str(Q(witness["arithmetic"][key]) + Q(1, 1000))
    elif tamper == "family":
        witness["family"]["normalization"] = "physical continuum Hamiltonian"
    elif tamper == "compression":
        witness["trial"]["compression"] = "the subspace is invariant"
    elif tamper == "dependent":
        witness["unresolved_cheap_stages"] = [0]
    elif tamper == "parent":
        cert["honesty"]["yang_mills_mass_gap_claim"] = True
    elif tamper == "formal":
        cert["honesty"]["theorem_prover_verified"] = True
    elif tamper == "metadata":
        cert["meta"]["transcend_backend"] = "libm"
    elif tamper == "claim":
        cert["claim"] = "continuum Yang Mills"
    else:
        witness["theorem_assumed"] = True
    forged = seal_certificate(cert)
    assert verify_certificate_digest(forged)
    assert replay(forged) is False


def test_scope_flags_cannot_turn_localized_energy_into_a_global_gap() -> None:
    row = attachment(Q(1, 1000), [], [[0, 1, 2, 3]], action_threshold=Q(1, 10))
    assert row["positive_bad_support_floor_verified"]
    for flag in (
        "ambient_graph_membership_verified", "full_cubic_block_cheap_increment_verified",
        "uniform_conditional_bad_probability_verified", "actual_conditional_gap_verified",
        "embedded_subspace_invariant_verified", "spectral_gap_claim", "infinite_volume_claim",
        "uniform_in_a_claim", "continuum_claim", "yang_mills_claim", "yang_mills_mass_gap_claim",
        "theorem_prover_verified", "mathlib_verified",
    ):
        assert row[flag] is False
    assert row["certificate"]["meta"]["transcend_backend"] == "not_used"


def _mul(a: Quaternion, b: Quaternion) -> Quaternion:
    return (a[0]*b[0]-sum((a[i]*b[i] for i in range(1, 4)), Q(0)),
            a[0]*b[1]+a[1]*b[0]+a[2]*b[3]-a[3]*b[2],
            a[0]*b[2]+a[2]*b[0]+a[3]*b[1]-a[1]*b[3],
            a[0]*b[3]+a[3]*b[0]+a[1]*b[2]-a[2]*b[1])


def _inverse(q: Quaternion) -> Quaternion:
    return q[0], -q[1], -q[2], -q[3]


def _stereographic(v: tuple[Q, Q, Q]) -> Quaternion:
    norm = sum((x*x for x in v), Q(0))
    denominator = 1+norm
    return (1-norm)/denominator, 2*v[0]/denominator, 2*v[1]/denominator, 2*v[2]/denominator


def _face_action_and_gradient(face: Face, links: dict[Edge, Quaternion]) -> tuple[Q, dict[tuple[Edge, int], Q]]:
    factors = []
    edges = _edges(face)
    for i, edge in enumerate(edges):
        q = links[edge]
        factors.append(q if face[i] < face[(i+1) % 4] else _inverse(q))
    holonomy: Quaternion = (Q(1), Q(0), Q(0), Q(0))
    for factor in factors:
        holonomy = _mul(holonomy, factor)
    gradient = {}
    for i, edge in enumerate(edges):
        for axis in range(3):
            vector = [Q(0)]*4
            vector[axis+1] = Q(1, 2)
            generator: Quaternion = (vector[0], vector[1], vector[2], vector[3])
            derivative = _mul(generator, factors[i]) if face[i] < face[(i+1) % 4] else _mul(factors[i], generator)
            if face[i] > face[(i+1) % 4]:
                derivative = (-derivative[0], -derivative[1], -derivative[2], -derivative[3])
            value: Quaternion = (Q(1), Q(0), Q(0), Q(0))
            for j, factor in enumerate(factors):
                value = _mul(value, derivative if j == i else factor)
            gradient[edge, axis] = -2*value[0]
    return 2-2*holonomy[0], gradient


@pytest.mark.parametrize("face_count", (1, 4, 7))
def test_original_link_gradient_majorant_on_exact_grid_and_random_noncommuting_fields(face_count: int) -> None:
    """Finite mathematical diagnostics; sampling is not the universal proof."""
    faces = _pages(face_count)
    edges = sorted({edge for face in faces for edge in _edges(face)})
    row = attachment(Q(1, 100), [(0, 1)], faces)
    d = _arithmetic(row)["max_added_face_edge_incidence"]
    rng = Random(4200+face_count)
    fields = []
    for value in (Q(0), Q(1, 10), Q(1, 2), Q(1)):
        fields.append({edge: _stereographic((value, Q(i % 3, 10), Q((i+1) % 3, 10)))
                       for i, edge in enumerate(edges)})
    for _ in range(8):
        fields.append({edge: _stereographic((Q(rng.randrange(-4, 5), 10), Q(rng.randrange(-4, 5), 10),
                                            Q(rng.randrange(-4, 5), 10))) for edge in edges})
    for links in fields:
        actions: list[Q] = []
        combined: dict[tuple[Edge, int], Q] = {}
        for face in faces:
            action, gradient = _face_action_and_gradient(face, links)
            assert 0 <= action <= 4
            assert sum((value**2 for value in gradient.values()), Q(0)) == action*(4-action)
            actions.append(action)
            for key, value in gradient.items():
                combined[key] = combined.get(key, Q(0)) + value
        total = sum(actions, Q(0))
        exact_gradient = sum((value**2 for value in combined.values()), Q(0))
        assert exact_gradient <= d*(4*total-sum((a*a for a in actions), Q(0)))
        assert exact_gradient <= d*total*(4-total/face_count)
        assert exact_gradient <= 4*d*total


def test_haar_radial_moment_trial_cost_and_barrier_on_grid_and_random_values() -> None:
    """Independent high-precision diagnostics for the analytic trial formula."""
    rng = Random(110)
    parameters = [Q(1, 100), Q(1, 10), Q(1), Q(10), Q(100)]
    parameters += [Q(rng.randrange(1, 300), 37) for _ in range(15)]
    with mp.workdps(60):
        for t in parameters:
            x = mp.mpf(t.numerator)/t.denominator
            mean = mp.besseli(2, 4*x)/mp.besseli(1, 4*x)
            assert 0 <= mean <= 1
            assert 1-mean <= 3/(8*x)
            kappa = 1/x
            energy = 3*kappa*x*mean/2 + 4*(1-mean)/kappa
            assert energy <= 3
        for t in (Q(1, 10), Q(1), Q(3)):
            x = mp.mpf(t.numerator)/t.denominator
            denominator = mp.quad(lambda q, x=x: mp.exp(4*x*q)*mp.sqrt(1-q*q), [-1, 0, 1])
            mean = mp.quad(lambda q, x=x: q*mp.exp(4*x*q)*mp.sqrt(1-q*q), [-1, 0, 1])/denominator
            gradient = x*x*mp.quad(lambda q, x=x: (1-q*q)*mp.exp(4*x*q)*mp.sqrt(1-q*q), [-1, 0, 1])/denominator
            assert abs(gradient-3*x*mean/4) < mp.mpf("1e-55")
