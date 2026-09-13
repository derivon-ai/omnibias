# SPDX-License-Identifier: Apache-2.0
"""Exact contracts and refusal/replay tests for isolated physical metric bounds."""

from __future__ import annotations

from copy import deepcopy
from fractions import Fraction as Q
from typing import Any

import pytest
from omnibias.core.proof.certificate import seal_certificate
from omnibias.geometry.gauge.transfer.isolated_block_kinetic import (
    replay_su2_isolated_block_kinetic_certificate as replay,
)
from omnibias.geometry.gauge.transfer.isolated_block_kinetic import (
    su2_isolated_block_kinetic as certify,
)


@pytest.mark.parametrize("side", [1, 2, 3, 8])
def test_open_cube_counts_match_independent_cellular_euler_characteristic(side: int) -> None:
    result = certify(Q(1, 10**8), block_side=side)
    arithmetic = result["witness"]["arithmetic"]
    vertices = (side + 1) ** 3
    faces = 3 * side**2 * (side + 1)
    cubes = side**3
    # Contractibility gives V-E+F-C=1; hence the cycle rank is F-C.
    assert arithmetic["original_edge_count"] == vertices + faces - cubes - 1
    assert arithmetic["chord_count"] == faces - cubes
    assert arithmetic["tree_edge_count"] == vertices - 1
    assert result["status"] == "PASS"
    assert replay(result["certificate"])


def test_tangent_metric_identity_and_small_positive_neighborhood() -> None:
    at_zero = certify(0)
    small = certify(Q(1, 1000))
    assert at_zero["kinetic_relative_error_upper"] == "0"
    assert at_zero["kinetic_comparison_lower"] == "1"
    assert at_zero["kinetic_comparison_upper"] == "1"
    assert small["status"] == "PASS"
    assert 0 < Q(small["kinetic_relative_error_upper"]) < Q(11, 100)
    assert Q(small["kinetic_comparison_lower"]) > Q(89, 100)
    assert small["isolated_physical_tree_gauge_form_verified"]
    assert small["isolated_block_positive_kinetic_comparison_verified"]


@pytest.mark.parametrize("radius", [Q(1, 10), 1, 2])
def test_large_error_is_replayable_inconclusive_not_a_lower_comparison(radius: Q | int) -> None:
    result = certify(radius)
    assert result["status"] == "INCONCLUSIVE"
    assert result["isolated_block_kinetic_error_bound_verified"]
    assert not result["isolated_block_positive_kinetic_comparison_verified"]
    assert Q(result["kinetic_comparison_lower"]) <= 0
    assert replay(result["certificate"])


@pytest.mark.parametrize("side", [2, 3, 4, 16, 1024, 4096])
def test_explicit_growing_block_sequence_has_a_uniform_rational_error_rate(side: int) -> None:
    result = certify(Q(1, side**12), block_side=side)
    a = result["witness"]["arithmetic"]
    delta = Q(a["right_gradient_frame_relative_error_upper"])
    eta = Q(a["right_jacobian_inverse_error_upper"])
    linear = Q(a["linear_metric_eigenvalue_upper"])
    assert a["tree_edge_count"] <= 7 * side**3
    assert a["chord_count"] <= 5 * side**3
    assert delta <= Q(71, side**6)
    assert eta <= Q(3, 5 * side**12)
    assert linear * (2 * eta + eta**2) <= Q(45, side**6)
    assert Q(result["kinetic_relative_error_upper"]) <= Q(116, side**6) + Q(3195, side**12)
    assert Q(result["kinetic_relative_error_upper"]) < Q(166, side**6)
    assert replay(result["certificate"])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"edge_radius": True},
        {"edge_radius": False},
        {"edge_radius": 0.001},
        {"edge_radius": "1/1000"},
        {"edge_radius": None},
        {"edge_radius": Q(-1, 100)},
        {"edge_radius": Q(201, 100)},
        {"edge_radius": 0, "block_side": True},
        {"edge_radius": 0, "block_side": 1.0},
        {"edge_radius": 0, "block_side": Q(1)},
        {"edge_radius": 0, "block_side": 0},
        {"edge_radius": 0, "block_side": -1},
    ],
)
def test_strict_exact_inputs_and_chart_domain(kwargs: dict[str, Any]) -> None:
    with pytest.raises((TypeError, ValueError)):
        certify(**kwargs)


@pytest.mark.parametrize("value", [None, [], (), "certificate", 1, True])
def test_nonmapping_replay_inputs_are_false(value: Any) -> None:
    assert not replay(value)


@pytest.mark.parametrize(
    "field,value",
    [
        ("kinetic_comparison_lower", "999"),
        ("log_coordinate_relative_error_upper", "0"),
        ("tree_edge_count", 6),
        ("chord_count", 6),
        ("linear_metric_eigenvalue_upper", "1"),
        ("right_jacobian_inverse_error_upper", "0"),
    ],
)
def test_resealed_mathematical_tampering_is_rejected(field: str, value: str | int) -> None:
    certificate = deepcopy(certify(Q(1, 1000))["certificate"])
    certificate["payload"]["witness"]["arithmetic"][field] = value
    assert not replay(seal_certificate(certificate))


@pytest.mark.parametrize(
    "field",
    [
        "actual_vacuum_approximation_verified",
        "embedded_block_conditional_metric_verified",
        "flat_measure_operator_comparison_verified",
        "uniform_in_block_size_claim",
        "spectral_gap_claim",
        "continuum_claim",
        "yang_mills_mass_gap_claim",
    ],
)
def test_isolated_kinetic_bound_cannot_promote_other_objects(field: str) -> None:
    result = certify(Q(1, 1000))
    assert result[field] is False
    certificate = deepcopy(result["certificate"])
    certificate["honesty"][field] = True
    assert not replay(seal_certificate(certificate))


def test_replay_rebuilds_geometry_and_root_gauss_constraint() -> None:
    result = certify(Q(1, 1000))
    assert "internal chords" in result["witness"]["derivative_scope"]
    for key, value in (
        ("tree", "all horizontal edges"),
        ("measure", "flat independent logarithms"),
        ("exterior_scope", "arbitrary conditioned exterior"),
    ):
        certificate = deepcopy(result["certificate"])
        certificate["payload"]["witness"][key] = value
        assert not replay(seal_certificate(certificate))
    certificate = deepcopy(result["certificate"])
    certificate["payload"]["witness"]["family"]["gauss_law"] = "root unconstrained"
    assert not replay(seal_certificate(certificate))
