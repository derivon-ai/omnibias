# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Likelihood-score information and finite observation-jet visibility."""

import math
from dataclasses import replace

import numpy as np
import pytest
from omnibias.pinn.inverse.observation import (
    LikelihoodScores,
    ObservationModel,
    candidate_information,
    identifiability_report,
    observation_information,
    observation_visibility,
    profile_information,
)


def gaussian_scores(theta, _design):
    nodes, weights = np.polynomial.hermite.hermgauss(3)
    z = np.sqrt(2.0) * nodes
    # Actual likelihood derivatives of N(mu, exp(2*eta)). Three-node
    # Gauss-Hermite integrates their degree-four products exactly in real arithmetic.
    scores = np.column_stack((z * np.exp(-theta[1]), z * z - 1.0))
    return LikelihoodScores(scores, weights, "quadrature", "three-node Gauss-Hermite")


def gaussian_model():
    return ObservationModel(
        lambda t, d: np.array([t[0]]),
        lambda t, d: np.array([[1.0, 0.0]]),
        ("mean", "log_scale"),
        score_provider=gaussian_scores,
    )


def test_heteroscedastic_gaussian_includes_scale_information_and_profiles_nuisance():
    theta = np.array([0.7, 0.4])
    design = np.ones((1, 1))
    model = gaussian_model()

    def covariance(t, d):
        return np.exp(2.0 * t[1])

    info = observation_information(model, theta, design, covariance=covariance)
    expected = np.diag([np.exp(-2.0 * theta[1]), 2.0])
    np.testing.assert_allclose(info.fisher, expected, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(
        info.information_factor.T @ info.information_factor, expected, rtol=1e-13, atol=1e-13
    )
    assert info.score_provenance == "quadrature" and info.information_kind == "likelihood_scores"
    rank = identifiability_report(info)
    assert rank.rank == 2 and not rank.certified and "likelihood-score" in rank.scope

    fixed = observation_information(
        replace(model, score_provider=None), theta, design, covariance=covariance(theta, design)
    )
    np.testing.assert_array_equal(fixed.fisher[1], np.zeros(2))
    assert identifiability_report(fixed).rank == 1
    np.testing.assert_array_equal(info.whitened_jacobian, fixed.whitened_jacobian)
    assert fixed.information_factor is fixed.whitened_jacobian
    assert profile_information(info.information_factor, [1], [0]).fisher[0, 0] == pytest.approx(2.0)
    assert profile_information(info.information_factor, [0], [1]).fisher[0, 0] == pytest.approx(
        np.exp(-2.0 * theta[1])
    )
    np.testing.assert_allclose(
        candidate_information(model, theta, np.ones((2, 1)), covariance=covariance),
        np.stack([expected, expected]),
        atol=1e-13,
    )


def test_score_provider_override_and_redundant_nuisance_scores():
    def redundant_scores(theta, design):
        supplied = gaussian_scores(theta, design)
        return LikelihoodScores(
            supplied.scores[:, [0, 1, 0]], supplied.weights, supplied.provenance
        )

    model = ObservationModel(
        lambda t, d: np.array([t[0] + t[2]]),
        lambda t, d: np.array([[1.0, 0.0, 1.0]]),
        ("mean", "log_scale", "offset"),
    )
    info = observation_information(
        model,
        np.array([0.3, 0.2, 0.4]),
        np.ones(1),
        covariance=lambda t, d: np.exp(2 * t[1]),
        score_provider=redundant_scores,
    )
    profile = profile_information(info.information_factor, [1], [0, 2])
    assert profile.nuisance_rank == 1 and identifiability_report(info).rank == 2
    assert profile.fisher[0, 0] == pytest.approx(2.0)


@pytest.mark.parametrize("provenance", ["exact_finite", "quadrature", "monte_carlo"])
def test_score_expectation_uses_nonnegative_normalized_weights_and_records_provenance(provenance):
    supplied = LikelihoodScores(
        np.array([[1.0], [2.0], [100.0]]), np.array([2.0, 6.0, 0]), provenance
    )
    model = ObservationModel(
        lambda t, d: t.copy(),
        lambda t, d: np.eye(1),
        ("a",),
        score_provider=lambda t, d: supplied,
    )
    info = observation_information(model, np.ones(1), np.ones(1))
    assert info.fisher[0, 0] == pytest.approx(0.25 + 0.75 * 4)
    assert info.score_provenance == provenance and not identifiability_report(info).certified
    huge = LikelihoodScores(np.ones((2, 1)), np.full(2, 1e308), provenance)
    np.testing.assert_array_equal(huge.weights, [0.5, 0.5])


@pytest.mark.parametrize("weights", [[1, -1], [0, 0], [1, np.inf], [1, np.nan], [1], [[1], [1]]])
def test_reject_malformed_score_weights(weights):
    with pytest.raises(ValueError, match="weight"):
        LikelihoodScores(np.ones((2, 2)), np.array(weights), "monte_carlo")


@pytest.mark.parametrize("scores", [[], [1, 2], [[np.nan, 1]], np.ones((2, 0))])
def test_reject_malformed_score_matrix(scores):
    with pytest.raises(ValueError, match="score"):
        LikelihoodScores(np.array(scores), np.ones(1), "quadrature")


def test_explicit_provider_and_covariance_validation():
    model = replace(gaussian_model(), score_provider=None)
    theta, design = np.zeros(2), np.ones((1, 1))
    with pytest.raises(ValueError, match="requires an explicit score_provider"):
        observation_information(model, theta, design, covariance=lambda t, d: 1.0)
    with pytest.raises(ValueError, match="LikelihoodScores"):
        observation_information(model, theta, design, score_provider=lambda t, d: np.ones((2, 2)))
    with pytest.raises(ValueError, match="column"):
        observation_information(
            model,
            theta,
            design,
            score_provider=lambda t, d: LikelihoodScores(
                np.ones((2, 1)), np.ones(2), "exact_finite"
            ),
        )
    for covariance in (-1.0, np.array([1.0, 2.0]), np.ones((2, 2)), np.nan):
        with pytest.raises(ValueError, match="covariance|variance"):
            observation_information(
                gaussian_model(), theta, design, covariance=lambda t, d, c=covariance: c
            )
    with pytest.raises(ValueError, match="provenance"):
        LikelihoodScores(np.ones((1, 2)), np.ones(1), "unverified_exact")
    supplied = gaussian_scores(theta, design)
    supplied.scores[0, 0] = np.nan
    with pytest.raises(ValueError, match="score"):
        observation_information(model, theta, design, score_provider=lambda t, d: supplied)


def power_model(power):
    def directional(t, d, v, order):
        return np.stack(
            [
                np.full(
                    len(d),
                    math.factorial(power)
                    / math.factorial(power - k)
                    * t[0] ** (power - k)
                    * v[0] ** k,
                )
                if k <= power
                else np.zeros(len(d))
                for k in range(order + 1)
            ]
        )

    return ObservationModel(
        lambda t, d: np.full(len(d), t[0] ** power),
        lambda t, d: np.full((len(d), 1), power * t[0] ** (power - 1)),
        ("t",),
        directional_jet=directional,
    )


@pytest.mark.parametrize("power", [2, 3])
def test_zero_first_order_information_has_actual_higher_order_visibility(power):
    model = power_model(power)
    theta, design = np.zeros(1), np.ones(3)
    assert identifiability_report(observation_information(model, theta, design)).rank == 0
    short = observation_visibility(model, theta, design, np.ones(1), max_order=power - 1)
    assert short.status == "inconclusive_at_supplied_order" and short.leading_vector is None
    report = observation_visibility(model, theta, design, np.array([-2.0]), max_order=4)
    assert report.first_visible_order == power and not report.certified
    assert report.max_order == 4 and report.threshold == 1e-12
    assert report.status == "visible" and "local observation" in report.scope
    np.testing.assert_array_equal(
        report.leading_vector, np.full(3, math.factorial(power) * (-2) ** power)
    )
    np.testing.assert_array_equal(report.derivatives[: power - 1], 0)


def test_directional_provider_contract_and_malformed_jet_guards():
    model = power_model(2)
    theta, design, direction = np.ones(1), np.ones(2), np.ones(1)
    with pytest.raises(ValueError, match="explicit directional_jet"):
        observation_visibility(replace(model, directional_jet=None), theta, design, direction)
    with pytest.raises(ValueError, match="direction"):
        observation_visibility(model, theta, design, np.zeros(1))
    for order in (0, -1, 1.5, True):
        with pytest.raises(ValueError, match="integer"):
            observation_visibility(model, theta, design, direction, max_order=order)
    for tower in (np.zeros((5, 3)), np.full((5, 2), np.nan), np.zeros((5, 2))):
        with pytest.raises(ValueError, match="tower|match"):
            observation_visibility(
                replace(model, directional_jet=lambda t, d, v, n, result=tower: result),
                theta,
                design,
                direction,
            )
