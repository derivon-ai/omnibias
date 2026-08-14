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
import pytest

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


def test_learned_partition_recovers_switched_ode() -> None:
    pytest.importorskip("torch")
    from omnibias.symbolic.piecewise import fit_learned_piecewise_ode

    rng = np.random.default_rng(0)
    n = 600
    x = rng.uniform(-2.0, 2.0, size=n)
    u = rng.uniform(-1.0, 2.0, size=n)
    du = _switched_law(x, u)
    automaton, state = fit_learned_piecewise_ode(
        x.reshape(-1, 1),
        u,
        du,
        n_gates=1,
        degree=2,
        steps=350,
        lr=0.08,
        beta=12.0,
        l1=1e-3,
        entropy=1e-3,
        seed=0,
        alpha=1e-12,
        threshold=1e-5,
    )
    laws = {law.region: law for law in automaton.laws}
    assert set(laws) == {0, 1}
    a, b = laws[0].equation, laws[1].equation
    # regimes may swap with the learned orientation of W
    recovered = []
    for eq in (a, b):
        recovered.append((float(eq.intercept), float(eq.coefficients[0]), float(eq.coefficients[1])))
    def _is_a(row: tuple[float, float, float]) -> bool:
        return abs(row[0] - 1.0) < 0.08 and abs(row[1] - (-1.0)) < 0.08 and abs(row[2]) < 1e-4
    def _is_b(row: tuple[float, float, float]) -> bool:
        return abs(row[0]) < 0.08 and abs(row[1] - (-2.0)) < 0.08 and abs(row[2]) < 1e-4
    assert any(_is_a(r) for r in recovered)
    assert any(_is_b(r) for r in recovered)
    t = float(np.asarray(state["t"]).reshape(-1)[0])
    w = np.asarray(state["W"]).reshape(-1)
    dom = int(np.argmax(np.abs(w)))
    assert dom == 0
    assert abs(t) < 0.2
    report = automaton.report()
    assert "x[0]" in report
    design, names = polynomial_value_library(u, degree=2)
    piece_rmse = rmse(du, automaton.predict(x.reshape(-1, 1), design))
    glob = global_sparse_law(design, du, names, alpha=1e-12)
    glob_rmse = rmse(du, glob.predict(design))
    assert piece_rmse < 1e-3 * glob_rmse
    assert glob_rmse > 0.3


def _trajectory_field_jet() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    from omnibias.symbolic.field_discovery import extract_field_jet, fit_neural_field_nd

    xs, u = _switched_trajectory()
    sel = np.linspace(0, xs.size - 1, 400).astype(int)
    x = xs[sel].reshape(-1, 1)
    u_obs = np.asarray(u[sel], dtype=float)
    field = fit_neural_field_nd(x, u_obs, hidden=200, seed=0)
    jet = extract_field_jet(field, x, max_order=1)
    return x, u_obs, jet.value(), jet.partial((1,))


def _assert_switched_tab_head_recovery(
    W: np.ndarray,
    t: np.ndarray,
    x: np.ndarray,
    u_jet: np.ndarray,
    du_jet: np.ndarray,
    partition,
) -> None:
    """Fitted ``W``/``t`` (no post-hoc polish) recover both field-jet laws."""
    design, names = polynomial_value_library(u_jet, degree=1)
    automaton = fit_piecewise_law(
        partition,
        x,
        design,
        du_jet,
        names,
        lhs_name="du",
        alpha=1e-12,
        threshold=1e-5,
        min_samples=20,
    )
    laws = {law.region: law for law in automaton.laws}
    assert set(laws) == {0, 1}
    recovered = [
        (float(eq.intercept), float(eq.coefficients[0]))
        for eq in (laws[0].equation, laws[1].equation)
    ]

    def _is_relax(row: tuple[float, float]) -> bool:
        return abs(row[0] - 0.5) < 0.20 and abs(row[1] - (-0.5)) < 0.20

    def _is_decay(row: tuple[float, float]) -> bool:
        return abs(row[0]) < 0.20 and abs(row[1] - (-1.0)) < 0.20

    assert any(_is_relax(r) for r in recovered)
    assert any(_is_decay(r) for r in recovered)
    w = np.asarray(W, dtype=float).reshape(-1)
    assert int(np.argmax(np.abs(w))) == 0
    x_split = float(np.asarray(t).reshape(-1)[0]) / float(w[0])
    assert abs(x_split) < 0.25
    assert "x[0]" in automaton.report()
    piece_rmse = rmse(du_jet, automaton.predict(x, design))
    glob = global_sparse_law(design, du_jet, names, alpha=1e-12)
    glob_rmse = rmse(du_jet, glob.predict(design))
    assert glob_rmse > 0.05
    assert piece_rmse < 0.5 * glob_rmse


def test_learned_partition_recovers_switched_trajectory() -> None:
    pytest.importorskip("torch")
    from omnibias.symbolic.piecewise import fit_learned_piecewise_ode

    x, _u_obs, u_jet, du_jet = _trajectory_field_jet()
    automaton, state = fit_learned_piecewise_ode(
        x,
        u_jet,
        du_jet,
        n_gates=1,
        degree=1,
        steps=200,
        lr=0.05,
        beta=8.0,
        l1=1e-3,
        entropy=1e-3,
        seed=0,
        alpha=1e-12,
        threshold=1e-5,
    )
    laws = {law.region: law for law in automaton.laws}
    assert set(laws) == {0, 1}
    recovered = []
    for eq in (laws[0].equation, laws[1].equation):
        recovered.append((float(eq.intercept), float(eq.coefficients[0])))

    def _is_relax(row: tuple[float, float]) -> bool:
        return abs(row[0] - 0.5) < 0.20 and abs(row[1] - (-0.5)) < 0.20

    def _is_decay(row: tuple[float, float]) -> bool:
        return abs(row[0]) < 0.20 and abs(row[1] - (-1.0)) < 0.20

    assert any(_is_relax(r) for r in recovered)
    assert any(_is_decay(r) for r in recovered)
    t = float(np.asarray(state["t"]).reshape(-1)[0])
    w = np.asarray(state["W"]).reshape(-1)
    assert int(np.argmax(np.abs(w))) == 0
    assert abs(t) < 0.25
    assert "x[0]" in automaton.report()
    design, names = polynomial_value_library(u_jet, degree=1)
    piece_rmse = rmse(du_jet, automaton.predict(x, design))
    glob = global_sparse_law(design, du_jet, names, alpha=1e-12)
    glob_rmse = rmse(du_jet, glob.predict(design))
    assert glob_rmse > 0.05
    assert piece_rmse < 0.5 * glob_rmse


@pytest.mark.parametrize("kind", ["softtree", "arrangement"])
def test_tab_head_hardened_partition_recovers_switched_trajectory(kind: str) -> None:
    """Tab head trained on observed trajectory ``du``, hardened from fitted ``W``/``t``.

    Distinct from :func:`fit_learned_piecewise_ode` (differentiable partition
    control). No oracle :class:`~omnibias.partition.PartitionParams`.
    """
    pytest.importorskip("torch")
    pytest.importorskip("omnibias.tab")
    import torch
    from omnibias.tab import arrangement_params, tree_params

    x, u_obs, u_jet, du_jet = _trajectory_field_jet()
    # Observed trajectory derivative (kinked at the switch). The field-jet
    # ``du`` is smoothed, so a 2-leaf constant model fitted to it does not
    # place the split at ``x=0``. STLSQ still runs on the field jet.
    du_obs = np.gradient(u_obs, x[:, 0])
    if kind == "softtree":
        from omnibias.tab import SoftTreeConfig
        from omnibias.tab.torch.model import SoftTreeEnsemble
        from omnibias.tab.torch.train import fit_second_order

        cfg = SoftTreeConfig(
            n_features=1,
            n_trees=1,
            depth=1,
            task="regression",
            n_outputs=1,
            beta_final=16.0,
            seed=0,
            leaf_l2=1e-6,
        )
        torch.manual_seed(0)
        model = SoftTreeEnsemble(cfg)
        fit_second_order(
            model,
            x,
            du_obs,
            steps=80,
            anneal=False,
            leaf_l2=1e-6,
            weight_l2=1e-6,
        )
        p = model.to_params()
        W = np.asarray(p.W[0], dtype=float)
        t = np.asarray(p.t[0], dtype=float).reshape(-1)
        partition = tree_params(W, t, n_features=1, beta_final=32.0)
    else:
        from omnibias.tab.torch.arrangement import ArrangementClassifier

        torch.manual_seed(0)
        model = ArrangementClassifier(1, 1, beta=32.0, task="regression")
        xt = torch.as_tensor(x, dtype=torch.float64)
        yt = torch.as_tensor(du_obs.reshape(-1, 1), dtype=torch.float64)
        opt = torch.optim.Adam(model.parameters(), lr=0.05)
        for _ in range(250):
            opt.zero_grad(set_to_none=True)
            loss = torch.nn.functional.mse_loss(model(xt), yt)
            loss.backward()
            opt.step()
        W = model.W.detach().cpu().numpy()
        t = model.t.detach().cpu().numpy()
        partition = arrangement_params(W, t, n_features=1, beta_final=32.0)
    _assert_switched_tab_head_recovery(W, t, x, u_jet, du_jet, partition)


def test_vector_hybrid_automaton_shared_gates() -> None:
    from omnibias.symbolic.piecewise import polynomial_vector_library

    rng = np.random.default_rng(3)
    n = 700
    x = rng.uniform(-2.0, 2.0, size=n)
    u0 = rng.uniform(-1.0, 2.0, size=n)
    u1 = rng.uniform(-1.0, 1.0, size=n)
    U = np.stack([u0, u1], axis=1)
    du0 = np.where(x < 0.0, 1.0 - u0, -2.0 * u0)
    du1 = np.where(x < 0.0, u1, -u1)
    du = np.stack([du0, du1], axis=1)
    design, names = polynomial_vector_library(U, degree=2)
    partition = _axis_partition()
    automaton = fit_piecewise_law(
        partition,
        x.reshape(-1, 1),
        design,
        du,
        names,
        lhs_name="du",
        lhs_names=("du0", "du1"),
        alpha=1e-12,
        threshold=1e-5,
    )
    report = automaton.report()
    assert "du0" in report and "du1" in report
    pred = automaton.predict(x.reshape(-1, 1), design)
    assert pred.shape == (n, 2)
    piece_rmse = rmse(du.reshape(-1), pred.reshape(-1))
    glob0 = global_sparse_law(design, du0, names, alpha=1e-12)
    glob_rmse = rmse(du0, glob0.predict(design))
    assert piece_rmse < 0.05 * glob_rmse
    laws = {law.region: law for law in automaton.laws}
    assert len(laws[0].equations) == 2


def test_learned_vector_hybrid_automaton_shared_gates() -> None:
    pytest.importorskip("torch")
    from omnibias.symbolic.piecewise import fit_learned_piecewise_ode, polynomial_vector_library

    rng = np.random.default_rng(3)
    n = 700
    x = rng.uniform(-2.0, 2.0, size=n)
    u0 = rng.uniform(-1.0, 2.0, size=n)
    u1 = rng.uniform(-1.0, 1.0, size=n)
    U = np.stack([u0, u1], axis=1)
    du0 = np.where(x < 0.0, 1.0 - u0, -2.0 * u0)
    du1 = np.where(x < 0.0, u1, -u1)
    du = np.stack([du0, du1], axis=1)
    automaton, state = fit_learned_piecewise_ode(
        x.reshape(-1, 1),
        U,
        du,
        n_gates=1,
        degree=2,
        steps=350,
        lr=0.08,
        beta=12.0,
        l1=1e-3,
        entropy=1e-3,
        seed=0,
        alpha=1e-12,
        threshold=1e-5,
    )
    report = automaton.report()
    assert "du0" in report and "du1" in report
    laws = {law.region: law for law in automaton.laws}
    assert set(laws) == {0, 1}
    assert len(laws[0].equations) == 2
    names = list(automaton.term_names)
    i0 = names.index("u0")
    i1 = names.index("u1")

    def _row(eq: object) -> tuple[float, float, float]:
        coef = np.asarray(eq.coefficients, dtype=float)
        return (float(eq.intercept), float(coef[i0]), float(coef[i1]))

    rec0 = [_row(laws[r].equations[0]) for r in (0, 1)]
    rec1 = [_row(laws[r].equations[1]) for r in (0, 1)]

    def _du0_relax(row: tuple[float, float, float]) -> bool:
        return abs(row[0] - 1.0) < 0.08 and abs(row[1] - (-1.0)) < 0.08 and abs(row[2]) < 0.08

    def _du0_decay(row: tuple[float, float, float]) -> bool:
        return abs(row[0]) < 0.08 and abs(row[1] - (-2.0)) < 0.08 and abs(row[2]) < 0.08

    def _du1_pos(row: tuple[float, float, float]) -> bool:
        return abs(row[0]) < 0.08 and abs(row[1]) < 0.08 and abs(row[2] - 1.0) < 0.08

    def _du1_neg(row: tuple[float, float, float]) -> bool:
        return abs(row[0]) < 0.08 and abs(row[1]) < 0.08 and abs(row[2] - (-1.0)) < 0.08

    assert any(_du0_relax(r) for r in rec0)
    assert any(_du0_decay(r) for r in rec0)
    assert any(_du1_pos(r) for r in rec1)
    assert any(_du1_neg(r) for r in rec1)
    t = float(np.asarray(state["t"]).reshape(-1)[0])
    w = np.asarray(state["W"]).reshape(-1)
    assert int(np.argmax(np.abs(w))) == 0
    assert abs(t) < 0.25
    design, names = polynomial_vector_library(U, degree=2)
    pred = automaton.predict(x.reshape(-1, 1), design)
    assert pred.shape == (n, 2)
    piece_rmse = rmse(du.reshape(-1), pred.reshape(-1))
    glob0 = global_sparse_law(design, du0, names, alpha=1e-12)
    glob_rmse = rmse(du0, glob0.predict(design))
    assert glob_rmse > 0.3
    assert piece_rmse < 0.2 * glob_rmse


def test_differentiable_soft_piecewise_forward_parity() -> None:
    pytest.importorskip("torch")
    pytest.importorskip("jax")
    from omnibias.symbolic.piecewise import (
        soft_piecewise_forward_jax,
        soft_piecewise_forward_np,
    )

    rng = np.random.default_rng(6)
    x = rng.standard_normal((20, 1))
    u = rng.standard_normal(20)
    phi, _ = polynomial_value_library(u, 2)
    phi = np.concatenate([np.ones((20, 1)), phi], axis=1)
    W = np.array([[1.1]])
    t = np.array([0.05])
    xi = rng.standard_normal((2, phi.shape[1], 1))
    beta = 6.0
    np_f = soft_piecewise_forward_np(W, t, xi, x, phi, beta)
    import jax.numpy as jnp

    jax_f = np.asarray(
        soft_piecewise_forward_jax(
            jnp.asarray(W), jnp.asarray(t), jnp.asarray(xi), jnp.asarray(x), jnp.asarray(phi), beta
        )
    )
    assert np.max(np.abs(np_f - jax_f)) < 1e-9
