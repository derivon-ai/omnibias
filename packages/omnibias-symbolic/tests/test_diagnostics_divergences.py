# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Generalized-divergence diagnostics: numpy twins + matched-Gaussian objectives.

The new numpy divergence twins match the backend operators bit-for-bit (in x64),
and every matched-Gaussian objective is minimised (``-> 0``) by white Gaussian
residuals while flagging structured (bimodal / skewed) residuals.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from omnibias.symbolic.diagnostics import (
    DIVERGENCE_OBJECTIVES,
    chi_squared_divergence,
    chi_squared_to_gaussian,
    divergence_objective_term,
    hellinger_distance,
    hellinger_to_gaussian,
    js_to_gaussian,
    kl_divergence,
    kl_to_gaussian,
    renyi_divergence,
    renyi_to_gaussian,
    total_variation_distance,
    total_variation_to_gaussian,
    wasserstein2_to_gaussian,
    wasserstein_to_gaussian,
)

_P = np.array([0.5, 0.3, 0.2])
_Q = np.array([0.2, 0.5, 0.3])


# ----- numpy twins ----------------------------------------------------------


def test_twin_truths() -> None:
    assert float(total_variation_distance(_P, _Q)) == pytest.approx(0.5 * np.abs(_P - _Q).sum())
    bc = float(np.sqrt(_P * _Q).sum())
    assert float(hellinger_distance(_P, _Q)) == pytest.approx(math.sqrt(1.0 - bc), rel=1e-12)
    assert float(chi_squared_divergence(_P, _Q)) == pytest.approx(float(((_P - _Q) ** 2 / _Q).sum()))
    assert float(renyi_divergence(_P, _Q, 1.0 + 1e-7)) == pytest.approx(
        float(kl_divergence(_P, _Q)), rel=1e-3
    )


def test_renyi_rejects_alpha_one() -> None:
    with pytest.raises(ValueError, match="alpha != 1"):
        renyi_divergence(_P, _Q, 1.0)


def test_twins_match_jax_backend() -> None:
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    import omnibias.jax as oj

    pj, qj = jnp.asarray(_P), jnp.asarray(_Q)
    assert float(total_variation_distance(_P, _Q)) == pytest.approx(
        float(oj.total_variation_distance(pj, qj)), abs=1e-12
    )
    assert float(hellinger_distance(_P, _Q)) == pytest.approx(
        float(oj.hellinger_distance(pj, qj)), abs=1e-12
    )
    assert float(chi_squared_divergence(_P, _Q)) == pytest.approx(
        float(oj.chi_squared_divergence(pj, qj)), abs=1e-12
    )
    assert float(renyi_divergence(_P, _Q, 0.5)) == pytest.approx(
        float(oj.renyi_divergence(pj, qj, 0.5)), abs=1e-12
    )


# ----- matched-Gaussian objectives ------------------------------------------

_GAUSSIAN_OBJS = [
    kl_to_gaussian,
    js_to_gaussian,
    total_variation_to_gaussian,
    hellinger_to_gaussian,
    chi_squared_to_gaussian,
    wasserstein_to_gaussian,
    wasserstein2_to_gaussian,
]


@pytest.mark.parametrize("fn", _GAUSSIAN_OBJS)
def test_gaussian_residuals_score_low_structured_high(fn) -> None:  # type: ignore[no-untyped-def]
    rng = np.random.default_rng(0)
    gauss = rng.normal(size=6000)
    bimodal = np.sign(rng.normal(size=6000)) + 0.05 * rng.normal(size=6000)
    assert fn(gauss) < fn(bimodal)


@pytest.mark.parametrize("fn", _GAUSSIAN_OBJS)
def test_degenerate_residual_is_zero(fn) -> None:  # type: ignore[no-untyped-def]
    assert fn(np.zeros(100)) == 0.0


def test_renyi_to_gaussian_behaviour() -> None:
    rng = np.random.default_rng(1)
    gauss = rng.normal(size=6000)
    skewed = rng.exponential(size=6000)
    assert renyi_to_gaussian(gauss, 0.5) < renyi_to_gaussian(skewed, 0.5)
    assert renyi_to_gaussian(np.full(50, 1.0), 0.5) == 0.0


# ----- objective dispatch ---------------------------------------------------


def test_all_objectives_dispatch_and_separate() -> None:
    rng = np.random.default_rng(2)
    n = 6000
    x = rng.normal(size=(n, 1))
    gauss = rng.normal(size=n)
    bimodal = np.sign(rng.normal(size=n)) + 0.05 * rng.normal(size=n)
    for name in DIVERGENCE_OBJECTIVES:
        if name == "residual_mi":
            continue  # dependence objective, exercised below
        good = divergence_objective_term(name, x, gauss)
        bad = divergence_objective_term(name, x, bimodal)
        assert good >= 0.0 and bad > good


def test_residual_mi_objective_detects_dependence() -> None:
    rng = np.random.default_rng(3)
    x = np.linspace(-3.0, 3.0, 4000).reshape(-1, 1)
    white = rng.normal(scale=0.1, size=4000)
    structured = np.sin(3.0 * x[:, 0])  # deterministic function of x
    assert divergence_objective_term("residual_mi", x, structured) > divergence_objective_term(
        "residual_mi", x, white
    )


def test_unknown_objective_raises() -> None:
    with pytest.raises(ValueError, match="unknown divergence objective"):
        divergence_objective_term("nope", np.zeros((4, 1)), np.zeros(4))
