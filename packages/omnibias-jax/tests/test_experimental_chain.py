# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Layer-convention adapters and the descriptor of a layer list."""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
from omnibias.core.chain import ChainDescriptor  # noqa: E402
from omnibias.jax import neural_field_laplacian  # noqa: E402
from omnibias.jax.architectures import make_jet_mlp  # noqa: E402
from omnibias.jax.experimental import (  # noqa: E402
    as_batch,
    describe_layers,
    layers_of,
    mlp_axis_tables,
    projected_deep_laplacian,
    ridge_layers,
    separable_axis_layers,
)


def _ridge(rng, D=5, H=7):
    W = jnp.asarray(rng.standard_normal((H, D)) / np.sqrt(D))
    b = jnp.asarray(0.3 * rng.standard_normal(H))
    c = jnp.asarray(rng.standard_normal(H))
    return W, b, c


def test_ridge_layers_reproduce_the_one_layer_laplacian() -> None:
    rng = np.random.default_rng(0)
    W, b, c = _ridge(rng)
    x = jnp.asarray(rng.standard_normal((6, 5)))
    layers = ridge_layers(W, b, c, "tanh")
    _, _, lap, _ = projected_deep_laplacian(x, layers)
    expected = neural_field_laplacian(x, W, b, c, "tanh")
    np.testing.assert_allclose(np.asarray(lap)[:, 0], np.asarray(expected), rtol=1e-12, atol=1e-13)


def test_ridge_layers_readout_bias_and_multiple_outputs() -> None:
    rng = np.random.default_rng(1)
    W, b, c = _ridge(rng)
    layers = ridge_layers(W, b, jnp.stack([c, 2 * c]), "sin", readout_bias=[0.5, -1.0])
    x = jnp.asarray(rng.standard_normal((3, 5)))
    u, _, _, _ = projected_deep_laplacian(x, layers)
    base = jnp.sin(x @ W.T + b) @ c
    np.testing.assert_allclose(np.asarray(u[:, 0]), np.asarray(base + 0.5), rtol=1e-13)
    np.testing.assert_allclose(np.asarray(u[:, 1]), np.asarray(2 * base - 1.0), rtol=1e-13)
    chain = describe_layers(layers)
    assert chain.is_ridge and chain.output_dim == 2 and chain.layers[1].bias


@pytest.mark.parametrize(
    ("W", "b", "c"),
    [
        (jnp.ones(3), jnp.ones(3), jnp.ones(3)),
        (jnp.ones((3, 2)), jnp.ones(4), jnp.ones(3)),
        (jnp.ones((3, 2)), jnp.ones(3), jnp.ones(4)),
        (jnp.ones((3, 2)), jnp.ones(3), jnp.ones((2, 2, 3))),
    ],
)
def test_ridge_layers_validation(W, b, c) -> None:
    with pytest.raises(ValueError):
        ridge_layers(W, b, c)


def test_describe_layers_matches_construction() -> None:
    rng = np.random.default_rng(2)
    layers = [
        (jnp.asarray(rng.standard_normal((6, 11))), jnp.zeros(6), "tanh"),
        (jnp.asarray(rng.standard_normal((6, 6))), None, "sin"),
        (jnp.asarray(rng.standard_normal((2, 6))), jnp.zeros(2), None),
    ]
    chain = describe_layers(layers)
    expected = ChainDescriptor.from_widths(
        [11, 6, 6, 2], ["tanh", "sin", None], bias=[True, False, True]
    )
    assert chain == expected
    assert describe_layers(layers, dtype="float32").dtype == "float32"
    with pytest.raises(ValueError, match="dtype"):
        describe_layers(layers, dtype="int8")


def test_layers_of_accepts_models_and_validates() -> None:
    model = make_jet_mlp(4, 6, out_dim=1, depth=2, activation="tanh", seed=3)
    layers = layers_of(model)
    chain = describe_layers(layers)
    assert chain.widths == (4, 6, 6, 1) and chain.activations == ("tanh", "tanh", None)
    with pytest.raises(TypeError):
        layers_of(object())
    with pytest.raises(ValueError, match="non-empty"):
        layers_of([])
    with pytest.raises(ValueError, match="triple"):
        layers_of([(jnp.ones((2, 2)), None)])
    with pytest.raises(ValueError, match="layer 1 weight must have shape"):
        layers_of([(jnp.ones((3, 2)), None, "tanh"), (jnp.ones((2, 4)), None, None)])


def test_separable_axis_layers_transposes_one_axis() -> None:
    rng = np.random.default_rng(4)
    D, widths = 3, (1, 5, 4, 1)
    tables = [
        (jnp.asarray(rng.standard_normal((D, a, b))), jnp.asarray(rng.standard_normal((D, b))))
        for a, b in zip(widths[:-1], widths[1:], strict=True)
    ]
    nodes = jnp.asarray(rng.uniform(-1, 1, (D, 7)))
    value = mlp_axis_tables(tables, nodes, 0, activation="tanh")[:, 0]  # (D, N, R) order-0 table
    for axis in range(D):
        layers = separable_axis_layers(tables, axis, "tanh")
        assert describe_layers(layers).activations == ("tanh", "tanh", None)
        u, *_ = projected_deep_laplacian(nodes[axis][:, None], layers)
        np.testing.assert_allclose(
            np.asarray(u).reshape(-1), np.asarray(value[axis]).reshape(-1), rtol=1e-12
        )
    with pytest.raises(ValueError, match="axis"):
        separable_axis_layers(tables, D)
    with pytest.raises(ValueError, match="layer weights"):
        separable_axis_layers([(jnp.ones((2, 2)), jnp.ones((2, 2)))], 0)


def test_as_batch_normalizes_a_single_point() -> None:
    a, batched = as_batch(jnp.ones(4))
    assert a.shape == (1, 4) and not batched
    a, batched = as_batch(jnp.ones((3, 4)))
    assert a.shape == (3, 4) and batched
    with pytest.raises(ValueError, match="x must have shape"):
        as_batch(jnp.ones((2, 3, 4)))
