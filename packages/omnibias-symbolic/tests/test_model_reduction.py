# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Reduction grammar creates generic models with measured, scoped remainders."""

import numpy as np
import pytest
from omnibias.symbolic.reduction import (
    ParameterReduction,
    ParameterTerm,
    ReductionCandidate,
    boundary_direction,
    evaluate_reduction,
    reduction_candidate,
)


def test_amplitude_removal_and_coalescence():
    def model(t, x):
        return t[0] * np.exp(-t[1] * x) + t[2] * np.exp(-t[3] * x)

    transform = ParameterReduction(
        (ParameterTerm(0), ParameterTerm(1), ParameterTerm(2, power=1), ParameterTerm(1)), 3
    )
    candidate = reduction_candidate(model, transform)
    x = np.linspace(0, 3, 31)
    eta = np.array([2.0, 0.4, 1.0])
    report = evaluate_reduction(candidate, [0.1, 0.05, 0.025], eta, x)
    np.testing.assert_allclose(candidate.reduced_model(eta, x), 2 * np.exp(-0.4 * x))
    np.testing.assert_allclose(report.observed_orders, 1, atol=1e-12)
    assert not report.certified


def test_singular_scaling_requires_explicit_limit_and_callback_works():
    # V and K grow together; V*x/(K+x) tends to eta0/eta1*x uniformly on finite x.
    transform = ParameterReduction((ParameterTerm(0, power=-1), ParameterTerm(1, power=-1)), 2)

    def model(t, x):
        return t[0] * x / (t[1] + x)

    with pytest.raises(ValueError, match="explicit"):
        reduction_candidate(model, transform)

    def reduced(eta, x):
        return eta[0] / eta[1] * x

    candidate = ReductionCandidate(model, reduced, transform)
    report = evaluate_reduction(
        candidate, [1e-3, 5e-4, 2.5e-4], np.array([2.0, 3.0]), np.linspace(0, 2, 30)
    )
    assert np.all(report.observed_orders > 0.999)
    assert report.max_errors[-1] < report.max_errors[0]


def test_weak_visible_direction_excludes_redundancy():
    direction = boundary_direction(np.diag([3.0, 0.2, 0.0]))
    np.testing.assert_allclose(abs(direction), [0, 1, 0])
    with pytest.raises(ValueError, match="visible"):
        boundary_direction(np.zeros((2, 2)))
