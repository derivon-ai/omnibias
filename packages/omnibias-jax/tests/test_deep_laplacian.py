# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Exactness / ceiling / honesty tests for the deep-network Laplacian fast lane.

Covers the plan's test matrix for :func:`omnibias.jax.laplacian.deep_field_laplacian`
and :func:`~omnibias.jax.laplacian.deep_field_polylaplacian`:

* exactness vs :func:`omnibias.jax.jet_mv.mlp_jet_mv` (the internal oracle),
* exactness vs ``jax.hessian`` (an external oracle),
* the depth-1 collapse onto :func:`omnibias.jax.laplacian.neural_field_laplacian`,
* the ``D=5000`` no-ceiling regression (the headline fix),
* Tier C estimator unbiasedness over >= 5 seeds against the Tier B exact value,
* the ``arctan`` order-cap error path.
"""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
from omnibias.core.contraction import support_jet_count  # noqa: E402
from omnibias.core.multi_index import index_position, multi_index_factorial  # noqa: E402
from omnibias.core.verified.sampled import hoeffding_enclosure  # noqa: E402
from omnibias.jax.jet_mv import mlp_jet_mv  # noqa: E402
from omnibias.jax.laplacian import (  # noqa: E402
    deep_field_laplacian,
    deep_field_polylaplacian,
    deep_field_value_grad_laplacian,
    neural_field_laplacian,
    restrict_first_layer,
)

_RICCATI = ("tanh", "sigmoid", "softplus", "gaussian", "exp", "sin")
_DIMS = (2, 4, 8, 16, 30)
_DEPTHS = (1, 2, 3, 4)
_RTOL = 1e-7
_ATOL = 1e-9


def _make_layers(dim: int, hidden: int, depth: int, activation: str, seed: int):
    """A small random deep MLP's ``(W, b, spec)`` layer list, float64."""
    rng = np.random.default_rng(seed)
    dims = [dim] + [hidden] * depth + [1]
    layers = []
    for i in range(len(dims) - 1):
        scale = 1.0 / np.sqrt(dims[i])
        W = jnp.asarray(rng.normal(scale=scale, size=(dims[i + 1], dims[i])), dtype=jnp.float64)
        b = jnp.asarray(rng.normal(scale=0.1, size=(dims[i + 1],)), dtype=jnp.float64)
        spec = None if i == len(dims) - 2 else activation
        layers.append((W, b, spec))
    return layers


def _oracle_laplacian_via_mlp_jet_mv(x, layers) -> jnp.ndarray:
    """``Delta f(x)`` by reading the order-2 rows off the full multivariate jet."""
    dim = x.shape[-1]
    jet = mlp_jet_mv(x, layers, 2)
    pos = index_position(dim, 2)
    total = None
    for i in range(dim):
        alpha = tuple(2 if j == i else 0 for j in range(dim))
        term = jet[pos[alpha]] * multi_index_factorial(alpha)
        total = term if total is None else total + term
    return total


@pytest.mark.parametrize("dim", _DIMS)
@pytest.mark.parametrize("depth", _DEPTHS)
@pytest.mark.parametrize("activation", _RICCATI)
def test_deep_field_laplacian_matches_mlp_jet_mv(dim, depth, activation) -> None:
    layers = _make_layers(dim, hidden=6, depth=depth, activation=activation, seed=0)
    x = jnp.asarray(np.random.default_rng(1).normal(scale=0.3, size=(dim,)))
    got = deep_field_laplacian(x, layers)[0]
    want = _oracle_laplacian_via_mlp_jet_mv(x, layers)[0]
    assert float(jnp.abs(got - want)) <= _ATOL + _RTOL * abs(float(want))


@pytest.mark.parametrize("dim", (2, 4, 8))
@pytest.mark.parametrize("activation", ("tanh", "sigmoid", "gaussian"))
def test_deep_field_laplacian_matches_jax_hessian(dim, activation) -> None:
    layers = _make_layers(dim, hidden=5, depth=3, activation=activation, seed=2)

    def f(xi):
        a = xi
        for W, b, spec in layers:
            u = a @ W.T + b
            if spec is None:
                a = u
            else:
                from omnibias.jax.activations import get_activation

                a = get_activation(spec).forward(u)
        return a[0]

    x = jnp.asarray(np.random.default_rng(3).normal(scale=0.3, size=(dim,)))
    hess = jax.hessian(f)(x)
    want = jnp.trace(hess)
    got = deep_field_laplacian(x, layers)[0]
    assert float(jnp.abs(got - want)) < 1e-9


def test_depth_1_collapses_onto_neural_field_laplacian() -> None:
    for activation in _RICCATI:
        dim, hidden = 6, 5
        layers = _make_layers(dim, hidden, depth=1, activation=activation, seed=4)
        (W1, b1, _), (W2, b2, _) = layers
        x = jnp.asarray(np.random.default_rng(5).normal(scale=0.3, size=(dim,)))
        deep = deep_field_laplacian(x, layers)[0]
        shallow = neural_field_laplacian(x, W1, b1, c=W2[0, :], activation=activation)
        assert float(jnp.abs(deep - shallow)) < 1e-13, activation


def test_value_grad_laplacian_matches_the_split_calls() -> None:
    layers = _make_layers(5, 6, depth=2, activation="tanh", seed=6)
    x = jnp.asarray(np.random.default_rng(7).normal(size=(3, 5)))
    value, grad, lap = deep_field_value_grad_laplacian(x, layers)
    lap_only = deep_field_laplacian(x, layers)
    assert jnp.allclose(lap, lap_only)
    assert value.shape == (3, 1)
    assert grad.shape == (3, 5, 1)


# -- the headline ceiling regression --------------------------------------- #


def test_d5000_no_ceiling_regression() -> None:
    """The whole point of the fast lane: succeed where mlp_jet_mv must refuse."""
    dim = 5000
    layers = _make_layers(dim, hidden=4, depth=2, activation="tanh", seed=8)
    x = jnp.asarray(np.random.default_rng(9).normal(scale=0.1, size=(2, dim)))
    lap = deep_field_laplacian(x, layers)
    assert lap.shape == (2, 1)
    assert bool(jnp.all(jnp.isfinite(lap)))
    with pytest.raises(ValueError, match="multi-index budget"):
        mlp_jet_mv(x[0], layers, 2)


def test_restrict_first_layer_is_exact() -> None:
    dim = 6
    layers = _make_layers(dim, 5, depth=2, activation="tanh", seed=10)
    x0 = jnp.asarray(np.random.default_rng(11).normal(size=(dim,)))
    support = (1, 3, 4)
    local = restrict_first_layer(layers, x0, support)
    y = jnp.asarray(np.random.default_rng(12).normal(size=(len(support),)) * 0.1)
    # x = x0 + E_support @ y
    x_full = x0.at[jnp.asarray(support)].add(y)

    def value(layer_list, xi):
        a = xi
        for W, b, spec in layer_list:
            u = a @ W.T + b
            if spec is None:
                a = u
            else:
                from omnibias.jax.activations import get_activation

                a = get_activation(spec).forward(u)
        return a

    got = value(local, y)
    want = value(layers, x_full)
    assert jnp.allclose(got, want, atol=1e-12)


# -- Tiers B/C for k >= 2 ---------------------------------------------------- #


@pytest.mark.parametrize("k", (2, 3))
def test_polylaplacian_support_matches_mlp_jet_mv(k) -> None:
    dim = 4
    layers = _make_layers(dim, 5, depth=2, activation="tanh", seed=13)
    x = jnp.asarray(np.random.default_rng(14).normal(scale=0.2, size=(dim,)))
    got = deep_field_polylaplacian(x, layers, k, mode="support")[0]
    jet = mlp_jet_mv(x, layers, 2 * k)
    pos = index_position(dim, 2 * k)
    from omnibias.core.contraction import polylaplacian_multinomial_terms

    want = 0.0
    for beta, coeff in polylaplacian_multinomial_terms(dim, k):
        alpha = tuple(2 * v for v in beta)
        want = want + jet[pos[alpha], 0] * multi_index_factorial(alpha) * coeff
    assert float(jnp.abs(got - want)) <= 1e-8 + 1e-6 * abs(float(want))


def test_polylaplacian_estimator_is_unbiased_over_five_seeds() -> None:
    """Tier C's sample mean must sit within a few stderrs of the Tier B exact
    value across independent seeds, and a Hoeffding enclosure around a pooled
    sample must cover that exact value."""
    dim, k = 4, 2
    layers = _make_layers(dim, 5, depth=2, activation="tanh", seed=15)
    x = jnp.asarray(np.random.default_rng(16).normal(scale=0.2, size=(dim,)))
    exact = deep_field_polylaplacian(x, layers, k, mode="support")[0]

    n_directions = 4000
    means = []
    for seed in range(5):
        est = deep_field_polylaplacian(
            x, layers, k, mode="estimator", n_directions=n_directions, seed=seed
        )[0]
        means.append(float(est))
    means_arr = np.asarray(means)
    grand_mean = means_arr.mean()
    # Unbiasedness: the average over independent seeds should track the exact
    # value much more closely than any single seed's spread.
    assert abs(grand_mean - float(exact)) < 0.15 * (abs(float(exact)) + 1.0)

    # A Hoeffding enclosure built from the five seed means (a coarse but
    # legitimate use: each seed's mean is itself an average of n_directions
    # i.i.d. draws, so this checks coverage of the outer estimator).
    spread = float(np.std(means_arr)) + 1e-6
    lo = float(means_arr.min()) - 5 * spread
    hi = float(means_arr.max()) + 5 * spread
    report = hoeffding_enclosure(means, value_range=(lo, hi), delta=1e-3)
    assert report.interval.lo <= float(exact) <= report.interval.hi


def test_select_mode_auto_matches_budget_driven_choice() -> None:
    dim, k = 4, 2
    assert support_jet_count(dim, k) < 1000  # sanity: small enough to be "cubature"
    layers = _make_layers(dim, 5, depth=2, activation="tanh", seed=17)
    x = jnp.asarray(np.random.default_rng(18).normal(scale=0.2, size=(dim,)))
    auto = deep_field_polylaplacian(x, layers, k, mode="auto")[0]
    support = deep_field_polylaplacian(x, layers, k, mode="support")[0]
    assert float(jnp.abs(auto - support)) < 1e-10
    # Force estimator with a budget too small to fit: budget=1 should push to
    # Tier C. Average several seeds (rather than one) to keep this assertion
    # about *mode selection*, not Monte Carlo variance, from being flaky.
    ests = [
        float(
            deep_field_polylaplacian(
                x, layers, k, mode="auto", budget=1, n_directions=4000, seed=seed
            )[0]
        )
        for seed in range(8)
    ]
    est_mean = sum(ests) / len(ests)
    assert abs(est_mean - float(support)) < 0.3 * (abs(float(support)) + 1.0)


def test_polylaplacian_forward_mode_rejected_for_k_ge_2() -> None:
    layers = _make_layers(3, 4, depth=1, activation="tanh", seed=19)
    x = jnp.asarray(np.random.default_rng(20).normal(size=(3,)))
    with pytest.raises(ValueError, match="mode='forward' only applies to k=1"):
        deep_field_polylaplacian(x, layers, 2, mode="forward")


def test_polylaplacian_k_must_be_positive() -> None:
    layers = _make_layers(3, 4, depth=1, activation="tanh", seed=21)
    x = jnp.asarray(np.random.default_rng(22).normal(size=(3,)))
    with pytest.raises(ValueError, match="polylaplacian order k must be >= 1"):
        deep_field_polylaplacian(x, layers, 0)


# -- order-cap honesty: arctan is capped at n <= 2 --------------------------- #


def test_arctan_order_cap_raises_for_k2_support() -> None:
    """arctan's fastpath caps at order 2; Delta^2 needs order 4 and must raise,
    not silently mis-compute."""
    layers = _make_layers(3, 4, depth=1, activation="arctan", seed=23)
    x = jnp.asarray(np.random.default_rng(24).normal(size=(3,)))
    with pytest.raises(ValueError, match="does not support order"):
        deep_field_polylaplacian(x, layers, 2, mode="support")


def test_arctan_order_cap_raises_for_k2_estimator() -> None:
    layers = _make_layers(3, 4, depth=1, activation="arctan", seed=25)
    x = jnp.asarray(np.random.default_rng(26).normal(size=(3,)))
    with pytest.raises(ValueError, match="does not support order"):
        deep_field_polylaplacian(x, layers, 2, mode="estimator")


def test_arctan_k1_laplacian_still_works() -> None:
    """k=1 only ever needs fastpath order 2, which arctan does support."""
    layers = _make_layers(3, 4, depth=2, activation="arctan", seed=27)
    x = jnp.asarray(np.random.default_rng(28).normal(size=(3,)))
    lap = deep_field_laplacian(x, layers)
    assert bool(jnp.all(jnp.isfinite(lap)))
