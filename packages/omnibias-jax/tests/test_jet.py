# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Validation suite for the jax Faà di Bruno jet kernel.

Oracles (float64):

* single-layer reduction to the existing closed-form fast path,
* :mod:`jax.experimental.jet` Taylor-mode autodiff,
* nested ``jax.jacfwd``,
* the explicit Bell-polynomial decomposition
  (:func:`omnibias.core.bell.faa_di_bruno_terms`) vs the shifted-power kernel,
* the dense (pre-reduction) composition kernel as a bit-for-bit value oracle,
* the order-cap error path for a bounded-order activation.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
from jax.experimental import jet as jax_jet  # noqa: E402
from omnibias.core.bell import faa_di_bruno_terms  # noqa: E402
from omnibias.jax.activations import get_activation, list_activations  # noqa: E402
from omnibias.jax.jet import (  # noqa: E402
    _path_jet,
    _sigma_tower,
    affine_jet,
    antiderivative_jet,
    compose_jet,
    compose_jet_riccati,
    derivative_jet,
    jet_to_tower,
    layer_jet,
    mlp_jet,
)


def _dense_compose_jet(u_jet, sigma_tower):
    """The dense shifted-power kernel ``compose_jet`` replaced.

    Kept verbatim as the value oracle: it forms every ``(k, n)`` product,
    including the ones that vanish because ``w = u - u_0`` has valuation 1.
    """
    np1 = u_jet.shape[0]
    zero = jnp.zeros_like(u_jet[0])
    w = [zero] + [u_jet[j] for j in range(1, np1)]
    p = [jnp.ones_like(u_jet[0])] + [zero for _ in range(np1 - 1)]
    result = [sigma_tower[0] * p[0]] + [zero for _ in range(np1 - 1)]
    fact = 1.0
    for k in range(1, np1):
        fact *= k
        new_p = []
        for n in range(np1):
            acc = zero
            for i in range(n + 1):
                acc = acc + p[i] * w[n - i]
            new_p.append(acc)
        p = new_p
        dk = sigma_tower[k] / fact
        for n in range(np1):
            result[n] = result[n] + dk * p[n]
    return jnp.stack(result, axis=0)


_RICCATI_NAMES = sorted(
    {name for name in list_activations() if get_activation(name).riccati_polynomial}
)
# Base points that keep tan / cot / coth away from their poles.
_RICCATI_BASE = {"tan": 0.4, "cot": 1.0, "coth": 1.2}


def _build_mlp(seed: int = 0, dims=(3, 5, 4, 2), act: str = "tanh"):
    rng = np.random.default_rng(seed)
    layers = []
    for i in range(len(dims) - 1):
        din, dout = dims[i], dims[i + 1]
        W = jnp.asarray(rng.normal(scale=0.7, size=(dout, din)))
        b = jnp.asarray(rng.normal(scale=0.5, size=(dout,)))
        spec = None if i == len(dims) - 2 else get_activation(act)
        layers.append((W, b, spec))
    x0 = jnp.asarray(rng.normal(size=(dims[0],)))
    v = jnp.asarray(rng.normal(size=(dims[0],)))
    return layers, x0, v


def _forward(layers):
    def f(x):
        z = x
        for W, b, spec in layers:
            z = W @ z + b
            if spec is not None:
                z = spec.forward(z)
        return z

    return f


# ----- single-layer analytic reduction -----


@pytest.mark.parametrize("act", ["tanh", "sigmoid", "gaussian", "sin"])
def test_single_layer_reduces_to_fastpath(act: str) -> None:
    rng = np.random.default_rng(3)
    D, H = 4, 6
    W = jnp.asarray(rng.normal(size=(H, D)))
    b = jnp.asarray(rng.normal(size=(H,)))
    x0 = jnp.asarray(rng.normal(size=(D,)))
    v = jnp.asarray(rng.normal(size=(D,)))
    spec = get_activation(act)
    order = 6
    x_jet = _path_jet(x0, v, order)
    tower = jet_to_tower(layer_jet(x_jet, W, b, spec, order))  # (order+1, H)

    z0 = W @ x0 + b
    wv = W @ v
    for k in range(order + 1):
        expected = spec.forward(z0) if k == 0 else spec.fastpath(z0, k) * wv**k
        assert jnp.allclose(tower[k], expected, rtol=1e-12, atol=1e-12)


# ----- deep MLP vs jax.experimental.jet -----


def test_deep_mlp_matches_experimental_jet() -> None:
    layers, x0, v = _build_mlp(seed=1)
    f = _forward(layers)
    order = 6
    series = [v] + [jnp.zeros_like(v) for _ in range(order - 1)]
    y0, terms = jax_jet.jet(f, (x0,), (series,))
    expected = [y0, *terms]  # derivative tower d^k/dt^k f
    tower = jet_to_tower(mlp_jet(x0, v, layers, order))
    for k in range(order + 1):
        assert jnp.allclose(tower[k], expected[k], rtol=1e-10, atol=1e-10)


# ----- deep MLP vs nested jacfwd -----


def test_deep_mlp_matches_nested_jacfwd() -> None:
    layers, x0, v = _build_mlp(seed=2)
    f = _forward(layers)
    order = 6

    def g(t):
        return f(x0 + t * v)

    def kth(k):
        fn = g
        for _ in range(k):
            fn = jax.jacfwd(fn)
        return fn(0.0)

    tower = jet_to_tower(mlp_jet(x0, v, layers, order))
    for k in range(order + 1):
        assert jnp.allclose(tower[k], kth(k), rtol=1e-9, atol=1e-9)


# ----- Bell-polynomial decomposition vs shifted-power kernel -----


def test_compose_jet_matches_bell_oracle() -> None:
    rng = np.random.default_rng(11)
    order = 6
    # Arbitrary pre-activation jet and an arbitrary "activation" derivative tower
    # (decoupled from any real sigma, so this tests the composition algebra).
    u_jet = jnp.asarray(rng.normal(size=(order + 1, 3)))
    sigma_tower = jnp.asarray(rng.normal(size=(order + 1, 3)))
    out = jet_to_tower(compose_jet(u_jet, sigma_tower))  # derivative tower of b

    # Independent Bell-polynomial re-summation.
    u_jet_np = np.asarray(u_jet)
    sigma_np = np.asarray(sigma_tower)
    u_deriv = np.array(
        [math.factorial(i) * u_jet_np[i] for i in range(order + 1)]
    )  # u^(i)
    expected = np.empty_like(sigma_np)
    expected[0] = sigma_np[0]  # b = sigma(u0)
    for n in range(1, order + 1):
        acc = np.zeros(u_jet_np.shape[1])
        for k, exps, coeff in faa_di_bruno_terms(n):
            prod = np.ones(u_jet_np.shape[1])
            for i, e in enumerate(exps, start=1):
                if e:
                    prod = prod * u_deriv[i] ** e
            acc = acc + coeff * sigma_np[k] * prod
        expected[n] = acc
    assert np.allclose(np.asarray(out), expected, rtol=1e-10, atol=1e-10)


# ----- order-cap error path -----


def test_order_cap_raises_value_error() -> None:
    rng = np.random.default_rng(5)
    D, H = 3, 4
    W = jnp.asarray(rng.normal(size=(H, D)))
    b = jnp.asarray(rng.normal(size=(H,)))
    x0 = jnp.asarray(rng.normal(size=(D,)))
    v = jnp.asarray(rng.normal(size=(D,)))
    # arctan caps at order 2; an order-3 jet must fail loudly.
    with pytest.raises(ValueError, match="does not support order"):
        layer_jet(_path_jet(x0, v, 3), W, b, "arctan", 3)


def test_affine_jet_is_linear_per_order() -> None:
    rng = np.random.default_rng(7)
    order = 4
    z_jet = jnp.asarray(rng.normal(size=(order + 1, 5)))
    W = jnp.asarray(rng.normal(size=(3, 5)))
    b = jnp.asarray(rng.normal(size=(3,)))
    out = affine_jet(z_jet, W, b)
    assert jnp.allclose(out[0], W @ z_jet[0] + b)
    for k in range(1, order + 1):
        assert jnp.allclose(out[k], W @ z_jet[k])


# ----- two-sided (integral) tower: antiderivative_jet / derivative_jet -----


def test_antiderivative_jet_ftc_part1_roundtrip() -> None:
    layers, x0, v = _build_mlp(seed=4)
    order = 6
    jet = mlp_jet(x0, v, layers, order)  # (order+1, C)
    anti = antiderivative_jet(jet, constant=0.3)
    assert anti.shape[0] == jet.shape[0] + 1
    assert jnp.allclose(anti[0], jnp.full_like(jet[0], 0.3))
    back = derivative_jet(anti)
    assert back.shape == jet.shape
    assert jnp.allclose(back, jet, rtol=1e-12, atol=1e-12)


def test_antiderivative_derivative_jit_vmap_safe() -> None:
    rng = np.random.default_rng(9)
    jet = jnp.asarray(rng.normal(size=(6, 3)))
    a1 = jax.jit(antiderivative_jet)(jet)
    a2 = antiderivative_jet(jet)
    assert jnp.array_equal(a1, a2)
    d1 = jax.jit(derivative_jet)(jet)
    assert jnp.array_equal(d1, derivative_jet(jet))
    batch = jnp.stack([jet, jet * 2.0])
    av = jax.vmap(antiderivative_jet)(batch)
    assert av.shape == (2, 7, 3)


# ----- valuation-reduced kernel vs the dense kernel it replaced -----


@pytest.mark.parametrize("order", list(range(0, 11)))
def test_compose_jet_matches_dense_kernel_bitwise(order: int) -> None:
    """Skipping the structurally-zero products may not move a single bit."""
    rng = np.random.default_rng(1000 + order)
    for _ in range(4):
        u_jet = jnp.asarray(rng.normal(size=(order + 1, 4)))
        sigma_tower = jnp.asarray(rng.normal(size=(order + 1, 4)))
        got = np.asarray(compose_jet(u_jet, sigma_tower))
        want = np.asarray(_dense_compose_jet(u_jet, sigma_tower))
        assert np.array_equal(got, want)
        assert np.array_equal(got.view(np.int64), want.view(np.int64))


def test_compose_jet_bitwise_on_path_jets_and_float32() -> None:
    """Structural zeros (a line jet) and float32 must also be bit-preserved."""
    rng = np.random.default_rng(17)
    order = 7
    u_jet = jnp.zeros((order + 1, 3)).at[0].set(jnp.asarray(rng.normal(size=3)))
    u_jet = u_jet.at[1].set(jnp.asarray(rng.normal(size=3)))
    tower = jnp.asarray(rng.normal(size=(order + 1, 3)))
    assert np.array_equal(
        np.asarray(compose_jet(u_jet, tower)),
        np.asarray(_dense_compose_jet(u_jet, tower)),
    )
    u32 = u_jet.astype(jnp.float32)
    t32 = tower.astype(jnp.float32)
    got32 = compose_jet(u32, t32)
    assert got32.dtype == jnp.float32
    assert np.array_equal(
        np.asarray(got32), np.asarray(_dense_compose_jet(u32, t32))
    )


def test_compose_jet_does_not_propagate_inf_to_lower_orders() -> None:
    """The one documented semantic change: no ``inf * 0`` contamination."""
    rng = np.random.default_rng(29)
    order = 4
    u_jet = jnp.asarray(rng.normal(size=(order + 1, 1)))
    tower = jnp.asarray(rng.normal(size=(order + 1, 1))).at[order].set(jnp.inf)
    dense = _dense_compose_jet(u_jet, tower)
    lean = compose_jet(u_jet, tower)
    assert bool(jnp.all(jnp.isnan(dense[:order])))
    assert bool(jnp.all(jnp.isfinite(lean[:order])))
    assert bool(jnp.all(jnp.isinf(lean[order])))


# ----- Riccati fastpath: sigma' = P(sigma) -----


def test_riccati_registry_is_not_empty() -> None:
    assert {"sigmoid", "tanh", "exp"} <= set(_RICCATI_NAMES)


@pytest.mark.parametrize("act", _RICCATI_NAMES)
def test_compose_jet_riccati_matches_tower_kernel(act: str) -> None:
    spec = get_activation(act)
    poly = spec.riccati_polynomial
    assert poly is not None
    # tan / cot / coth cap their fastpath at order 3, so the tower oracle
    # (and hence the comparison) is only available that far.
    order = 3 if act in _RICCATI_BASE else 8
    rng = np.random.default_rng(31)
    u_jet = jnp.zeros((order + 1, 3))
    u_jet = u_jet.at[0].set(
        _RICCATI_BASE.get(act, 0.3) + jnp.asarray(rng.normal(scale=0.05, size=3))
    )
    u_jet = u_jet.at[1].set(jnp.asarray(rng.normal(scale=0.3, size=3)))
    u_jet = u_jet.at[2].set(jnp.asarray(rng.normal(scale=0.2, size=3)))
    want = compose_jet(u_jet, _sigma_tower(spec, u_jet[0], order))
    got = compose_jet_riccati(u_jet, spec.forward(u_jet[0]), poly)
    assert got.shape == want.shape
    assert jnp.allclose(got, want, rtol=1e-11, atol=1e-12)


@pytest.mark.parametrize("act", sorted(_RICCATI_BASE))
def test_compose_jet_riccati_passes_the_fastpath_order_cap(act: str) -> None:
    """tan / cot / coth: the tower path raises, the Riccati recurrence does not."""
    spec = get_activation(act)
    poly = spec.riccati_polynomial
    assert poly is not None
    order = 7
    z0 = jnp.asarray([_RICCATI_BASE[act]])
    d = jnp.asarray([0.35])
    u_jet = jnp.zeros((order + 1, 1)).at[0].set(z0).at[1].set(d)
    with pytest.raises(ValueError, match="does not support order"):
        _sigma_tower(spec, z0, order)

    got = jet_to_tower(compose_jet_riccati(u_jet, spec.forward(z0), poly))

    def kth(k: int):
        fn = lambda t: spec.forward(z0 + t * d)  # noqa: E731
        for _ in range(k):
            fn = jax.jacfwd(fn)
        return fn(0.0)

    for k in range(order + 1):
        assert jnp.allclose(got[k], kth(k), rtol=1e-9, atol=1e-9), k


def test_layer_and_mlp_jet_riccati_flag_agrees_with_default() -> None:
    rng = np.random.default_rng(37)
    D, H, order = 3, 4, 7
    W = jnp.asarray(rng.normal(scale=0.6, size=(H, D)))
    b = jnp.asarray(rng.normal(scale=0.4, size=(H,)))
    x0 = jnp.asarray(rng.normal(size=(D,)))
    v = jnp.asarray(rng.normal(size=(D,)))
    z_jet = _path_jet(x0, v, order)
    assert jnp.allclose(
        layer_jet(z_jet, W, b, "tanh", order, riccati=True),
        layer_jet(z_jet, W, b, "tanh", order),
        rtol=1e-11,
        atol=1e-12,
    )
    layers = [
        (W, b, "tanh"),
        (jnp.asarray(rng.normal(scale=0.6, size=(2, H))), None, "sigmoid"),
        (jnp.asarray(rng.normal(size=(1, 2))), None, None),
    ]
    assert jnp.allclose(
        mlp_jet(x0, v, layers, order, riccati=True),
        mlp_jet(x0, v, layers, order),
        rtol=1e-11,
        atol=1e-12,
    )


def test_riccati_fastpath_is_jit_and_vmap_safe() -> None:
    rng = np.random.default_rng(47)
    u_jet = jnp.asarray(rng.normal(size=(7, 3)))
    s0 = jnp.asarray(rng.normal(size=(3,)))
    poly = (0.0, 1.0, -1.0)
    eager = compose_jet_riccati(u_jet, s0, poly)
    jitted = jax.jit(lambda u, s: compose_jet_riccati(u, s, poly))(u_jet, s0)
    assert jnp.allclose(jitted, eager, rtol=1e-13, atol=1e-15)
    batch = jnp.stack([u_jet, 0.5 * u_jet])
    out = jax.vmap(lambda u: compose_jet_riccati(u, s0, poly))(batch)
    assert out.shape == (2, 7, 3)
    assert jnp.allclose(out[0], eager, rtol=1e-13, atol=1e-15)


def test_riccati_order_zero_and_constant_polynomial() -> None:
    u_jet = jnp.asarray([[0.3], [0.5], [0.25], [0.1]])
    # P constant: sigma' = 3 everywhere, so sigma(u) = sigma(u0) + 3 (u - u0).
    got = compose_jet_riccati(u_jet, jnp.asarray([2.0]), (3.0,))
    assert jnp.allclose(got, jnp.asarray([[2.0], [1.5], [0.75], [0.3]]))
    scalar = compose_jet_riccati(u_jet[:1], jnp.asarray([2.0]), (0.0, 1.0, -1.0))
    assert scalar.shape == (1, 1)
    assert jnp.array_equal(scalar[0], jnp.asarray([2.0]))


def test_riccati_error_paths() -> None:
    u_jet = jnp.asarray([[0.3], [0.5]])
    with pytest.raises(ValueError, match="at least one coefficient"):
        compose_jet_riccati(u_jet, jnp.asarray([1.0]), ())
    rng = np.random.default_rng(43)
    W = jnp.asarray(rng.normal(size=(2, 1)))
    with pytest.raises(ValueError, match="not in the Riccati class"):
        layer_jet(u_jet, W, None, "relu", 1, riccati=True)


def test_antiderivative_jet_torch_parity() -> None:
    """Bit-identical antiderivative / derivative jets across the jax and torch twins."""
    torch = pytest.importorskip("torch")
    from omnibias.torch.jet import antiderivative_jet as t_anti
    from omnibias.torch.jet import derivative_jet as t_deriv

    rng = np.random.default_rng(13)
    arr = rng.normal(size=(6, 4))
    jx = jnp.asarray(arr)
    tx = torch.as_tensor(arr, dtype=torch.float64)

    aj = np.asarray(antiderivative_jet(jx, constant=0.7))
    at = t_anti(tx, constant=0.7).numpy()
    assert np.array_equal(aj, at)

    dj = np.asarray(derivative_jet(jx))
    dt = t_deriv(tx).numpy()
    assert np.array_equal(dj, dt)
