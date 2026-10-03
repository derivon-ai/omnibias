# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Exact lower-hull and Farkas regressions for Viro patchworks."""

from dataclasses import replace
from fractions import Fraction as Q

import pytest
from omnibias.geometry.patchwork import staircase_triangulation
from omnibias.geometry.patchwork_height_lp import (
    RationalInequalitySystem,
    certify_rational_feasibility,
    certify_rational_infeasibility,
    certify_regular_heights,
    clear_denominators,
    lower_hull_inequalities,
    propose_farkas_infeasibility,
    propose_regular_heights,
    regular_height_system,
    verify_rational_feasibility,
    verify_rational_infeasibility,
    verify_regular_height_certificate,
)


def test_staircase_heights_have_an_exact_unit_lower_hull_margin():
    for degree in (2, 4, 8):
        triangulation = staircase_triangulation(degree)
        heights = {
            point: point[0] ** 2 + point[1] ** 2 + point[0] * point[1]
            for point in triangulation.vertices
        }
        certificate = certify_regular_heights(triangulation, heights)
        assert min(certificate.feasibility.residuals) == 0
        assert verify_regular_height_certificate(certificate)
        assert certificate.feasibility.seal["honesty"][
            "finite_rational_feasibility_verified"
        ]


def test_wrong_diagonal_heights_and_float_claims_are_rejected():
    triangulation = staircase_triangulation(2)
    with pytest.raises(ValueError, match="violates"):
        certify_regular_heights(
            triangulation,
            {point: point[0] * point[1] for point in triangulation.vertices},
        )
    with pytest.raises(TypeError, match="floats are proposer-only"):
        certify_regular_heights(
            triangulation,
            {point: float(point[0] + point[1]) for point in triangulation.vertices},
        )


def test_float_convex_proposer_is_accepted_only_after_exact_q_replay():
    pytest.importorskip("torch")
    certificate = propose_regular_heights(staircase_triangulation(2), bound=100)
    assert min(certificate.feasibility.residuals) >= 0
    assert verify_regular_height_certificate(certificate)
    forged = replace(
        certificate,
        heights=(
            certificate.heights[0] + 1,
            *certificate.heights[1:],
        ),
    )
    assert not verify_regular_height_certificate(forged)


def test_exact_farkas_alternative_certifies_infeasibility():
    # x >= 1 and -x >= 1 cannot both hold.
    system = RationalInequalitySystem(((Q(1),), (Q(-1),)), (Q(1), Q(1)))
    certificate = certify_rational_infeasibility(system, (1, 1))
    assert certificate.annihilator == (0,)
    assert certificate.contradiction == 2
    assert verify_rational_infeasibility(system, certificate)
    proposed = propose_farkas_infeasibility(system)
    assert proposed.multipliers == (Q(1, 2), Q(1, 2))
    assert verify_rational_infeasibility(system, proposed)
    with pytest.raises(ValueError, match="nonnegative"):
        certify_rational_infeasibility(system, (1, -1))
    with pytest.raises(ValueError, match="contradiction"):
        certify_rational_infeasibility(system, (1, 0))


def test_known_nonregular_six_point_triangulation_has_exact_farkas_witness():
    # Two nested lattice triangles and the nonregular T1 triangulation from the
    # standard six-point planar example.
    points = ((4, 0), (0, 4), (0, 0), (2, 1), (1, 2), (1, 1))
    triangles = (
        (points[0], points[1], points[3]),
        (points[1], points[2], points[4]),
        (points[0], points[2], points[5]),
        (points[1], points[3], points[4]),
        (points[2], points[4], points[5]),
        (points[0], points[3], points[5]),
        (points[3], points[4], points[5]),
    )
    system = lower_hull_inequalities(points, triangles)
    certificate = propose_farkas_infeasibility(system)
    assert system.n_variables == 6
    assert system.n_constraints == 21
    assert certificate.contradiction == 1
    assert sum(value != 0 for value in certificate.multipliers) == 3
    assert verify_rational_infeasibility(system, certificate)


def test_generic_feasibility_replay_and_tamper_rejection():
    system = RationalInequalitySystem(
        ((Q(1), Q(1)), (Q(-1), Q(2))),
        (Q(1), Q(-3)),
    )
    certificate = certify_rational_feasibility(system, (2, 0))
    assert certificate.residuals == (1, 1)
    assert verify_rational_feasibility(system, certificate)
    assert not verify_rational_feasibility(
        system,
        replace(certificate, residuals=(Q(100), Q(100))),
    )
    with pytest.raises(ValueError, match="violates"):
        certify_rational_feasibility(system, (0, 0))


def test_height_system_shape_and_integer_scaling():
    triangulation = staircase_triangulation(4)
    system = regular_height_system(triangulation)
    assert system.inequalities.n_variables == 15
    assert system.inequalities.n_constraints == 16 * 12
    assert clear_denominators((Q(1, 2), Q(-2, 3), 5)) == (3, -4, 30)
