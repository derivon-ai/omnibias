# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Per-region (piecewise) symbolic law discovery on the omnibias-partition substrate.

A switched first-order ODE ``du = f_region(u)`` (two linear regimes glued at a switch
surface) is recovered as a hybrid automaton: one sparse equation per region + the hardened
``if x[0] > 0`` switch. Global SINDy on the same data finds a single averaged law that fits
neither regime.
"""

from __future__ import annotations

import os

import numpy as np

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from omnibias.partition import PartitionConfig  # noqa: E402
from omnibias.partition._core.params import PartitionParams  # noqa: E402
from omnibias.symbolic.discovery import rmse  # noqa: E402
from omnibias.symbolic.piecewise import (  # noqa: E402
    HybridAutomaton,
    fit_piecewise_law,
    fit_piecewise_ode_law,
    global_sparse_law,
    polynomial_value_library,
)


def _axis_partition() -> PartitionParams:
    # depth-1 axis split on feature 0 at the origin: region 0 = {x<0}, region 1 = {x>0}.
    cfg = PartitionConfig(n_features=1, depth=1, split_kind="axis", beta_final=32.0, anneal_steps=1)
    return PartitionParams(cfg, W=np.array([[1.0]]), t=np.array([0.0]))


# Two linear regimes: A (x<0): du = 1 - u ; B (x>0): du = -2 u.
def _switched_law(x: np.ndarray, u: np.ndarray) -> np.ndarray:
    du = np.where(x < 0.0, 1.0 - 1.0 * u, -2.0 * u)
    return du


def test_exact_recovery_of_both_regimes() -> None:
    rng = np.random.default_rng(0)
    n = 600
    x = rng.uniform(-2.0, 2.0, size=n)
    u = rng.uniform(-1.0, 2.0, size=n)  # sample the state space directly (algebraic SINDy)
    du = _switched_law(x, u)
    design, names = polynomial_value_library(u, degree=2)
    partition = _axis_partition()
    automaton = fit_piecewise_law(
        partition, x.reshape(-1, 1), design, du, names, lhs_name="du", alpha=1e-12, threshold=1e-5
    )

    laws = {law.region: law for law in automaton.laws}
    assert set(laws) == {0, 1}

    # region 0 = {x<0}: du = 1 - u   -> intercept 1, coef(y) -1, coef(y^2) 0
    a = laws[0].equation
    assert abs(a.intercept - 1.0) < 1e-3
    assert abs(a.coefficients[0] - (-1.0)) < 1e-3
    assert abs(a.coefficients[1]) < 1e-6

    # region 1 = {x>0}: du = -2 u    -> intercept 0, coef(y) -2, coef(y^2) 0
    b = laws[1].equation
    assert abs(b.intercept) < 1e-3
    assert abs(b.coefficients[0] - (-2.0)) < 1e-3
    assert abs(b.coefficients[1]) < 1e-6


def test_piecewise_beats_global_average() -> None:
    rng = np.random.default_rng(1)
    n = 600
    x = rng.uniform(-2.0, 2.0, size=n)
    u = rng.uniform(-1.0, 2.0, size=n)
    du = _switched_law(x, u)
    design, names = polynomial_value_library(u, degree=2)
    partition = _axis_partition()

    automaton = fit_piecewise_law(partition, x.reshape(-1, 1), design, du, names, alpha=1e-12)
    piece_pred = automaton.predict(x.reshape(-1, 1), design)
    piece_rmse = rmse(du, piece_pred)

    glob = global_sparse_law(design, du, names, alpha=1e-12)
    glob_rmse = rmse(du, glob.predict(design))

    # the single averaged law cannot fit both regimes; piecewise is near-exact.
    assert piece_rmse < 1e-6
    assert glob_rmse > 0.3
    assert piece_rmse < 1e-3 * glob_rmse


def test_switch_conditions_and_routing() -> None:
    partition = _axis_partition()
    rng = np.random.default_rng(2)
    n = 200
    x = rng.uniform(-2.0, 2.0, size=n)
    u = rng.uniform(-1.0, 2.0, size=n)
    du = _switched_law(x, u)
    design, names = polynomial_value_library(u, degree=2)
    automaton = fit_piecewise_law(partition, x.reshape(-1, 1), design, du, names)

    assert automaton.switch_conditions() == ["x[0] > 0"]
    idx = automaton.region_of(x.reshape(-1, 1))
    assert np.array_equal(idx, (x > 0.0).astype(np.int64))
    report = automaton.report()
    assert "x[0] > 0" in report
    assert "NOT (x[0] > 0)" in report  # region 0 clause


def test_too_few_samples_raises() -> None:
    partition = _axis_partition()
    x = np.array([-1.0, -0.5])  # all in region 0, and fewer than n_terms + 1
    u = np.array([0.3, 0.7])
    du = _switched_law(x, u)
    design, names = polynomial_value_library(u, degree=2)
    try:
        fit_piecewise_law(partition, x.reshape(-1, 1), design, du, names)
    except ValueError as exc:
        assert "min_samples" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError for too few samples")


# ----------------------------------------------------------------------
# The full neural pipeline: fit a global field, read closed-form (u, du) jets,
# discover per-region laws. Uses a switched TRAJECTORY (u continuous, du kinked).
# ----------------------------------------------------------------------
def _switched_trajectory(n_fine: int = 4000) -> tuple[np.ndarray, np.ndarray]:
    # A (x<0): du = 0.5 - 0.5 u ; B (x>0): du = -1.0 u ; u continuous at 0.
    xs = np.linspace(-2.0, 2.0, n_fine)
    dx = xs[1] - xs[0]
    u = np.empty_like(xs)
    u[0] = 0.0
    for i in range(1, n_fine):
        xi = xs[i - 1]
        du = 0.5 - 0.5 * u[i - 1] if xi < 0.0 else -1.0 * u[i - 1]
        u[i] = u[i - 1] + dx * du
    return xs, u


def test_neural_ode_pipeline_beats_global() -> None:
    xs, u = _switched_trajectory()
    # subsample for the field fit
    sel = np.linspace(0, xs.size - 1, 400).astype(int)
    x = xs[sel].reshape(-1, 1)
    y = u[sel]
    partition = _axis_partition()

    automaton, field = fit_piecewise_ode_law(
        x, y, partition, degree=2, hidden=200, seed=0, min_samples=20
    )
    assert isinstance(automaton, HybridAutomaton)
    assert {law.region for law in automaton.laws} == {0, 1}
    assert automaton.switch_conditions() == ["x[0] > 0"]

    # rebuild the (u, du) jet the driver used, to score piecewise vs global honestly
    from omnibias.symbolic.field_discovery import extract_field_jet

    jet = extract_field_jet(field, x, max_order=1)
    uu = jet.value()
    du = jet.partial((1,))
    design, names = polynomial_value_library(uu, degree=2)

    piece_rmse = rmse(du, automaton.predict(x, design))
    glob = global_sparse_law(design, du, names)
    glob_rmse = rmse(du, glob.predict(design))

    # a single averaged law fits neither slope; the piecewise automaton is clearly better.
    assert piece_rmse < 0.7 * glob_rmse
