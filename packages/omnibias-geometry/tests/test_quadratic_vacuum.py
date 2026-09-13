# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Finite Riccati enclosures, noncommuting witnesses and scale covariance."""

from copy import deepcopy
from fractions import Fraction as Q
from typing import Any

import numpy as np
import pytest
from omnibias.core.proof.certificate import seal_certificate
from omnibias.geometry.gauge.transfer.quadratic_vacuum import (
    quadratic_vacuum_root as certify,
)
from omnibias.geometry.gauge.transfer.quadratic_vacuum import (
    replay_quadratic_vacuum_root_certificate as replay,
)


def _rehash(certificate: dict[str, Any]) -> dict[str, Any]:
    return seal_certificate(certificate)


def test_adjacent_plaquette_root_and_additive_seed_failure() -> None:
    k = [[4, -1], [-1, 4]]
    p = [[Q(248, 125), Q(-63, 250)], [Q(-63, 250), Q(248, 125)]]
    result = certify(k, p, frequency_lower=Q(17, 10))
    assert result["status"] == "PASS" and replay(result["certificate"])
    arithmetic = result["witness"]["arithmetic"]
    assert arithmetic["residual_norm_upper"] == "19/62500"
    assert arithmetic["root_error_upper"] == "19/214500"
    wrong = certify(k, [[2, Q(-1, 2)], [Q(-1, 2), 2]], frequency_lower=Q(17, 10))
    assert wrong["status"] == "INCONCLUSIVE" and replay(wrong["certificate"])
    assert wrong["witness"]["arithmetic"]["residual_norm_upper"] == "5/4"


@pytest.mark.parametrize("scale", [Q(1), Q(1, 8), Q(1, 1024), Q(3, 7)])
def test_exact_scale_covariance(scale: Q) -> None:
    k = [[4, -1], [-1, 4]]
    p = [[Q(2), Q(-1, 4)], [Q(-1, 4), Q(2)]]
    original = certify(k, p, frequency_lower=Q(17, 10))["witness"]["arithmetic"]
    scaled = certify([[x * scale**2 for x in row] for row in k],
                     [[x * scale for x in row] for row in p],
                     frequency_lower=Q(17, 10) * scale, scale=scale)
    assert scaled["status"] == "PASS"
    arithmetic = scaled["witness"]["arithmetic"]
    for field in ("normalized_root_error_upper", "normalized_riccati_inverse_upper"):
        assert arithmetic[field] == original[field]
    assert Q(arithmetic["residual_norm_upper"]) == scale**2 * Q(original["residual_norm_upper"])


@pytest.mark.parametrize("seed", range(8))
def test_noncommuting_root_error_contains_grid_and_random_directions(seed: int) -> None:
    rng = np.random.default_rng(seed)
    n = 4
    p = [[Q(int(i == j) * 3) for j in range(n)] for i in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            p[i][j] = p[j][i] = Q(int(rng.integers(-2, 3)), 20)
    k = [[sum((p[i][t] * p[t][j] for t in range(n)), Q(0))
          + (Q(i + 1, 100) if i == j else Q(0)) for j in range(n)] for i in range(n)]
    result = certify(k, p, frequency_lower=Q(5, 2))
    assert result["status"] == "PASS"
    kf, pf = np.array(k, dtype=float), np.array(p, dtype=float)
    assert np.linalg.norm(kf @ pf - pf @ kf) > 0
    eigenvalues, vectors = np.linalg.eigh(kf)
    root = (vectors * np.sqrt(eigenvalues)) @ vectors.T
    bound = float(Q(result["witness"]["arithmetic"]["root_error_upper"]))
    directions = [np.array([1, x, y, z]) for x in np.linspace(-1, 1, 7)
                  for y in np.linspace(-1, 1, 7) for z in np.linspace(-1, 1, 7)]
    directions.extend(rng.normal(size=(200, n)))
    for vector in directions:
        assert np.linalg.norm((root - pf) @ vector) <= bound * np.linalg.norm(vector) + 1e-12
    assert np.linalg.norm(root - pf, ord=2) <= bound + 1e-12


@pytest.mark.parametrize("k,p", [([[0]], [[0]]), ([[1]], [[-1]]), ([[-1]], [[1]])])
def test_zero_negative_and_wrong_square_root_are_not_accepted(k: Any, p: Any) -> None:
    result = certify(k, p, frequency_lower=Q(1, 10))
    assert result["status"] == "INCONCLUSIVE"
    assert result["witness"]["arithmetic"]["root_error_upper"] is None
    assert replay(result["certificate"])


@pytest.mark.parametrize("bad", [0.5, True, "1", None])
def test_float_bool_string_inputs_refused(bad: Any) -> None:
    with pytest.raises(TypeError):
        certify([[bad]], [[1]], frequency_lower=1)


@pytest.mark.parametrize("k,p", [([], []), ([[1, 0]], [[1]]), ([[1, 1], [0, 1]], [[1]]),
                                 ([[1]], [[1, 0], [0, 1]])])
def test_matrix_contract(k: Any, p: Any) -> None:
    with pytest.raises(ValueError):
        certify(k, p, frequency_lower=1)


@pytest.mark.parametrize("flag", ["continuum_claim", "yang_mills_mass_gap_claim",
                                 "nonlinear_wilson_vacuum_verified", "uniform_in_a_claim"])
def test_rehashed_parent_promotion_rejected(flag: str) -> None:
    result = certify([[4]], [[2]], frequency_lower=2)
    assert not result[flag]
    certificate = deepcopy(result["certificate"])
    certificate["honesty"][flag] = True
    assert not replay(_rehash(certificate))


def test_rehashed_arithmetic_tampering_rejected() -> None:
    certificate = deepcopy(certify([[4]], [[2]], frequency_lower=2)["certificate"])
    certificate["payload"]["witness"]["arithmetic"]["quadratic_gap_lower"] = "3"
    assert not replay(_rehash(certificate))


@pytest.mark.parametrize("malformed", [None, [], {}, {"payload": {"type": "wrong"}}])
def test_bad_certificate(malformed: Any) -> None:
    assert not replay(malformed)
