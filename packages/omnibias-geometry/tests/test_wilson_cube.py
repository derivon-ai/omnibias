# SPDX-License-Identifier: Apache-2.0
"""Independent geometry, exact energy envelopes, source and scope regressions."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from fractions import Fraction as Q
from random import Random
from typing import Any

import pytest
from omnibias.core.proof.certificate import seal_certificate
from omnibias.geometry.gauge.transfer import wilson_cube
from omnibias.geometry.gauge.transfer.wilson_cube import (
    replay_su2_wilson_attached_cube_certificate as replay,
)
from omnibias.geometry.gauge.transfer.wilson_cube import (
    su2_wilson_attached_cube as certify,
)
from omnibias.geometry.gauge.transfer.wilson_large_field import su2_wilson_large_field


def _edges(face: list[int]) -> set[tuple[int, int]]:
    return {(min(face[i], face[(i + 1) % 4]), max(face[i], face[(i + 1) % 4])) for i in range(4)}


def test_cube_face_incidence_and_independent_cycle_rank() -> None:
    report = certify(1)
    geometry = report["witness"]["geometry"]
    old = _edges([0, 1, 2, 3])
    current = set(old)
    counts = []
    incidence: Counter[tuple[int, int]] = Counter()
    for face in [*geometry["side_faces"], geometry["top_face"]]:
        edges = _edges(face)
        counts.append(len(edges - current))
        incidence.update(edges)
        current |= edges
    assert counts == [3, 2, 2, 1, 0]
    assert len(current - old) == geometry["new_edge_count"] == 8
    assert max(incidence.values()) == 2
    # The full cube surface has six faces but cycle rank12-8+1=5.
    # The old bottom supplies one cycle; four sides supply the other four.
    assert len(current) - len({v for e in current for v in e}) + 1 == 5
    assert report["witness"]["arithmetic"]["zero_fresh_stage_count"] == 1
    assert not geometry["top_is_independent_fresh_attachment"]
    assert replay(report["certificate"])


@pytest.mark.parametrize("kappa", [Q(1, 1000), Q(1, 10), Q(1, 4), Q(1, 2), 1, Q(4, 3), 2, 10, 1000])
def test_exact_envelope_combines_feedback_and_distinct_trial_families(kappa: Q | int) -> None:
    k = Q(kappa)
    report = certify(k)
    a = report["witness"]["arithmetic"]
    # Independently eliminate the per-stage minimum and the old Haar cap.
    # At k<=4/3 the feedback bound45 applies; at larger k the global Haar
    # trial20/k already dominates both feedback and sequential24/k.
    assert Q(a["vacuum_increment_upper"]) == min(Q(45), 12 + 8 / k, 20 / k)
    assert Q(a["old_bottom_action_mean_upper"]) == min(3 * k / 2, Q(2))
    assert Q(a["vacuum_increment_upper"]) <= 45
    assert report["actual_curvature_feedback_operator_comparison_verified"]
    assert report["actual_old_moment_applicability_verified"]
    assert replay(report["certificate"])


def test_seeded_exact_couplings_preserve_actual_moment_and_concavity_caps() -> None:
    rng = Random(10873)
    for _ in range(24):
        k = Q(rng.randint(1, 300), rng.randint(1, 300))
        s = Q(rng.randint(1, 200), 10)
        report = certify(k, action_threshold=s)
        a = report["witness"]["arithmetic"]
        delta = Q(a["vacuum_increment_upper"])
        mean = Q(a["actual_added_action_mean_upper"])
        assert 0 <= mean <= min(Q(20), k * delta / 2)
        assert Q(a["large_action_probability_upper"]) == min(Q(1), mean / s)
        assert Q(a["good_localized_vacuum_norm_squared_lower"]) == max(Q(0), 1 - 2 * mean / s)
        # The concave gradient majorant is maximized at total action10.
        assert 0 <= Q(a["added_action_gradient_square_mean_upper"]) <= 40
        assert Q(a["vacuum_subtracted_bad_support_floor"]) == s / k - delta
        assert replay(report["certificate"])


def test_weak_coupling_energy_subtraction_is_local_and_positive() -> None:
    report = certify(Q(1, 1000))
    a = report["witness"]["arithmetic"]
    assert a["vacuum_increment_upper"] == "45"
    assert a["actual_added_action_mean_upper"] == "9/400"
    assert a["good_localized_vacuum_norm_squared_lower"] == "191/200"
    assert a["vacuum_subtracted_bad_support_floor"] == "955"
    assert a["universal_ims_error_upper"] == "242/6125"
    assert report["positive_bad_support_floor_verified"]
    assert Q(a["good_normalized_energy_above_vacuum_upper"]) > 0


def test_nonpositive_supported_floor_does_not_become_a_gap_or_source_failure() -> None:
    report = certify(1)
    a = report["witness"]["arithmetic"]
    assert report["status"] == "PASS"
    assert a["vacuum_subtracted_bad_support_floor"] == "-19"
    assert not report["positive_bad_support_floor_verified"]
    assert a["good_localized_vacuum_norm_squared_lower"] == "0"
    assert a["good_normalized_energy_above_vacuum_upper"] is None
    assert not report["spectral_gap_claim"]
    assert replay(report["certificate"])


@pytest.mark.parametrize("kappa", [Q(1, 10**6), Q(1, 3), 1, 8, 100])
def test_weighted_single_face_cost_from_moment_barrier(kappa: Q | int) -> None:
    k = Q(kappa)
    t, weight = 2 / k, Q(6)
    m_lower = max(Q(0), 1 - 3 / (8 * t))

    # e(m) is affine. Its maximum on the proved m interval is attained at
    # an endpoint; no numerical Bessel evaluation or guessed m is used.
    def energy(m: Q) -> Q:
        return 3 * k * t * m / 2 + 4 * weight * (1 - m) / k

    assert max(energy(m_lower), energy(Q(1))) <= Q(15, 2)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"kappa": True},
        {"kappa": False},
        {"kappa": 0.1},
        {"kappa": "1/10"},
        {"kappa": None},
        {"kappa": 0},
        {"kappa": -1},
        {"kappa": 1, "action_threshold": True},
        {"kappa": 1, "action_threshold": 1.0},
        {"kappa": 1, "action_threshold": "1"},
        {"kappa": 1, "action_threshold": 0},
        {"kappa": 1, "action_threshold": Q(201, 10)},
    ],
)
def test_exact_input_domain_and_no_user_honesty_flags(kwargs: dict[str, Any]) -> None:
    with pytest.raises((TypeError, ValueError)):
        certify(**kwargs)


@pytest.mark.parametrize(
    "field", ["old_mean", "moment_verified", "continuum_claim", "old_graph", "side_length"]
)
def test_unsupported_external_premises_are_not_arguments(field: str) -> None:
    with pytest.raises(TypeError):
        certify(1, **{field: True})


@pytest.mark.parametrize("value", [None, [], (), "certificate", 1, True])
def test_nonmapping_replay_refuses_without_throwing(value: Any) -> None:
    assert not replay(value)


@pytest.mark.parametrize(
    "field,value",
    [
        ("vacuum_increment_upper", "1"),
        ("old_bottom_action_mean_upper", "0"),
        ("weighted_side_coefficient", "1"),
        ("weighted_per_side_increment_upper", "3"),
        ("max_added_face_edge_incidence", 1),
        ("face_count", 4),
        ("feedback_next_moment_coefficient_slope", "1/2"),
        ("vacuum_subtracted_bad_support_floor", "1000000"),
    ],
)
def test_resealed_derived_bound_tampering_is_rejected(field: str, value: Any) -> None:
    certificate = deepcopy(certify(Q(1, 1000))["certificate"])
    certificate["payload"]["witness"]["arithmetic"][field] = value
    assert not replay(seal_certificate(certificate))


@pytest.mark.parametrize(
    "source", ["actual_old_vacuum_certificate", "generic_attachment_certificate"]
)
def test_rehashed_nested_source_tampering_cannot_earn_an_actual_bound(source: str) -> None:
    certificate = deepcopy(certify(Q(1, 1000))["certificate"])
    nested = certificate["payload"]["witness"][source]
    nested["payload"]["witness"]["family"]["group"] = "SU(3)"
    certificate["payload"]["witness"][source] = seal_certificate(nested)
    assert not replay(seal_certificate(certificate))


def test_valid_actual_source_at_the_wrong_coupling_is_rejected() -> None:
    certificate = deepcopy(certify(Q(1, 1000))["certificate"])
    certificate["payload"]["witness"]["actual_old_vacuum_certificate"] = su2_wilson_large_field(1)[
        "certificate"
    ]
    assert not replay(seal_certificate(certificate))


@pytest.mark.parametrize(
    "field",
    [
        "arbitrary_old_potential_moment_bound_verified",
        "embedded_full_cubic_increment_verified",
        "iterable_curvature_feedback_class_verified",
        "uniform_conditional_gap_verified",
        "uniform_conditional_bad_probability_verified",
        "embedded_subspace_invariant_verified",
        "infinite_volume_claim",
        "uniform_in_a_claim",
        "continuum_claim",
        "yang_mills_mass_gap_claim",
    ],
)
def test_scope_flags_cannot_be_promoted_by_resealing(field: str) -> None:
    report = certify(Q(1, 1000))
    assert report[field] is False
    certificate = deepcopy(report["certificate"])
    certificate["honesty"][field] = True
    assert not replay(seal_certificate(certificate))


def test_quantified_torus_family_and_fresh_vertex_geometry_are_replayed() -> None:
    certificate = deepcopy(certify(1)["certificate"])
    certificate["payload"]["witness"]["family"]["old_graph"] = "arbitrary old potential and graph"
    assert not replay(seal_certificate(certificate))
    certificate = deepcopy(certify(1)["certificate"])
    certificate["payload"]["witness"]["geometry"]["new_vertices"][0] = 0
    assert not replay(seal_certificate(certificate))


def test_trial_preservation_cannot_be_resealed_as_actual_vacuum_preservation() -> None:
    certificate = deepcopy(certify(1)["certificate"])
    feedback = certificate["payload"]["witness"]["curvature_feedback"]
    assert "not in the new actual vacuum" in feedback["old_observables"]
    feedback["old_observables"] = "the new actual vacuum preserves every old moment"
    assert not replay(seal_certificate(certificate))


def test_source_replay_failure_stops_construction(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(wilson_cube, "replay_su2_wilson_large_field_certificate", lambda _: False)
    with pytest.raises(ValueError, match="old-vacuum source"):
        certify(1)


def test_only_sealed_source_arithmetic_is_consumed(monkeypatch: pytest.MonkeyPatch) -> None:
    def polluted_summary(kappa: int | Q) -> dict[str, Any]:
        source = su2_wilson_large_field(kappa)
        source["plaquette_action_mean_upper"] = "0"
        return source

    monkeypatch.setattr(wilson_cube, "su2_wilson_large_field", polluted_summary)
    assert certify(Q(1, 1000))["witness"]["arithmetic"]["old_bottom_action_mean_upper"] == "3/2000"
