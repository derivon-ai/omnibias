# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Scientific certificates bind their finite operands and preserve unresolved domains."""

import random
from copy import deepcopy
from fractions import Fraction as Q

import pytest
from omnibias.core.proof.certificate import seal_certificate, verify_certificate_digest
from omnibias.core.verified.interval import Interval as I
from omnibias.verify.neuromanifold.scientific import (
    certify_identifiability,
    certify_quantum_geometry,
    certify_reduction,
    replay_quantum_geometry,
)


def test_information_box_full_rank_and_degenerate_nuisance():
    def j(box):
        return [[I.point(1), box[0]], [I.point(0), I.point(1)], [I.point(1), I.point(0)]]

    result = certify_identifiability(
        j,
        [I(-0.01, 0.01)],
        physical=[0],
        nuisance=[1],
        eigenvalue_floor=0.4,
        provider_assumption="analytic whitened derivative of the supplied observation map",
    )
    assert result.certified and verify_certificate_digest(result.certificate)
    for x in [k / 10000 for k in range(-100, 101)] + [
        random.Random(7 + i).uniform(-0.01, 0.01) for i in range(20)
    ]:
        assert result.gram[0][1].lo <= x <= result.gram[0][1].hi

    def duplicate(box):
        return [[I.point(1), I.point(1)], [I.point(0), I.point(0)]]

    failed = certify_identifiability(
        duplicate, [I(0, 1)], physical=[0], nuisance=[1], provider_assumption="duplicate columns"
    )
    assert not failed.certified
    with pytest.raises(TypeError, match="Interval"):
        certify_identifiability(
            lambda box: [[1.0]],
            [I(0, 1)],
            physical=[0],
            provider_assumption="invalid float provider",
        )


def test_uniform_reduction_derivative_coverage_and_nonuniform_refusal():
    # full-minus-reduced = eps*x², derivatives in x of orders 0,1,2.
    def errors(box):
        e, x = box
        return {0: e * x * x, 1: 2 * e * x, 2: 2 * e}

    result = certify_reduction(
        errors,
        [I(0, 0.001), I(-1, 1)],
        orders=(0, 1, 2),
        error_budget=0.0021,
        provider_assumption="exact polynomial value and x-derivative errors",
    )
    assert result.certified and not result.unresolved_regions

    # At x=eps=0 the ratio is not uniformly small, despite pointwise limit
    # zero at every fixed x>0. Singular corner must remain unresolved.
    def nonuniform(box):
        e, x = box
        return {0: e / (e + x)}

    failure = certify_reduction(
        nonuniform,
        [I(0, 0.01), I(0, 1)],
        error_budget=0.1,
        max_boxes=40,
        provider_assumption="rational error on nonsingular boxes only",
    )
    assert not failure.certified and failure.unresolved_regions
    assert failure.certificate["payload"]["unresolved"]


def test_subdivision_covers_dependency_error_and_budget_exhaustion():
    # Dependency loss x-x shrinks with domain subdivision.
    result = certify_reduction(
        lambda box: {0: box[0] - box[0]},
        [I(0, 1)],
        error_budget=0.13,
        max_boxes=31,
        provider_assumption="interval enclosure of exact zero",
    )
    assert result.certified
    intervals = sorted((box[0].lo, box[0].hi) for box in result.accepted_regions)
    assert intervals[0][0] == 0 and intervals[-1][1] == 1
    assert all(a[1] == b[0] for a, b in zip(intervals, intervals[1:], strict=False))
    failed = certify_reduction(
        lambda box: {0: box[0] - box[0]},
        [I(0, 1)],
        error_budget=0.13,
        max_boxes=2,
        provider_assumption="same exact identity",
    )
    assert not failed.certified


def test_exact_quantum_covariance_complete_kernel_and_replay():
    scores = [[Q(1), Q(0), Q(0)], [Q(1), Q(1), Q(1)], [Q(1), Q(2), Q(2)]]
    weights = [1, 2, 1]
    result = certify_quantum_geometry(scores, weights)
    assert result.certified and result.rank == 1 and len(result.null_basis) == 2
    assert result.covariance[1][1] == Q(1, 2)
    assert result.covariance[0][0] == 0
    assert replay_quantum_geometry(result.certificate)
    modified = deepcopy(result.certificate)
    modified["payload"]["covariance"][1][1] = "999"
    resealed = seal_certificate(modified)
    assert verify_certificate_digest(resealed) and not replay_quantum_geometry(resealed)
    inflated = deepcopy(result.certificate)
    inflated["claim"] = "a different scientific statement"
    assert not replay_quantum_geometry(seal_certificate(inflated))
    sampled = deepcopy(result.certificate)
    sampled["meta"]["sampling_kind"] = "monte_carlo"
    assert not replay_quantum_geometry(seal_certificate(sampled))
    incomplete = certify_quantum_geometry(scores, weights, null_basis=[[1, 0, 0]])
    assert not incomplete.certified and not incomplete.complete_null_space
    lost_physical = certify_quantum_geometry(scores, weights, physical_basis=[])
    assert not lost_physical.certified


def test_complex_scores_zero_weight_and_monte_carlo_refusal():
    result = certify_quantum_geometry(
        [[0, 0], [0, 0], [99, 99]], [1, 1, 0], imaginary_scores=[[0, 0], [1, 0], [0, 0]]
    )
    assert result.certified and result.rank == 1 and result.covariance[0][0] == Q(1, 4)
    constant = certify_quantum_geometry([[1, 2], [1, 2]], [1, 1])
    assert constant.certified and constant.rank == 0 and constant.projected == ()
    with pytest.raises(TypeError, match="Monte Carlo"):
        certify_quantum_geometry([[0.0], [1.0]], [1, 1])
    with pytest.raises(ValueError, match="exact finite"):
        certify_quantum_geometry([[0], [1]], [1, 1], sampling_kind="monte_carlo")
