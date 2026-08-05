# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Jet-based ``lim`` operator (jax): L'Hopital ratios + asymptote metadata."""

from __future__ import annotations

import math

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
from omnibias.core.spec import saturation_limit  # noqa: E402
from omnibias.jax.activations import get_activation  # noqa: E402
from omnibias.jax.jet import (  # noqa: E402
    lhopital_ratio,
    limit_of_ratio,
    mlp_jet,
    removable_value,
)


def _deep_mlp(seed: int = 5, dims: tuple[int, ...] = (3, 5, 4, 2), act: str = "tanh"):
    """A small deep MLP ``(layers, x0, v)`` for the jet/forward helpers below."""
    rng = np.random.default_rng(seed)
    layers = []
    for i in range(len(dims) - 1):
        w = jnp.asarray(rng.normal(scale=0.7, size=(dims[i + 1], dims[i])))
        b = jnp.asarray(rng.normal(scale=0.5, size=(dims[i + 1],)))
        spec = None if i == len(dims) - 2 else get_activation(act)
        layers.append((w, b, spec))
    x0 = jnp.asarray(rng.normal(size=(dims[0],)))
    v = jnp.asarray(rng.normal(size=(dims[0],)))
    return layers, x0, v


def _mlp_forward(layers):  # type: ignore[no-untyped-def]
    def f(x):  # type: ignore[no-untyped-def]
        z = x
        for w, b, spec in layers:
            z = w @ z + b
            if spec is not None:
                z = spec.forward(z)
        return z

    return f

# Taylor jets a_k = f^(k)(0)/k! at the origin.
_SIN = jnp.array([0.0, 1.0, 0.0, -1.0 / 6.0, 0.0, 1.0 / 120.0])
_X = jnp.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
_ONE_MINUS_COS = jnp.array([0.0, 0.0, 0.5, 0.0, -1.0 / 24.0, 0.0])
_X2 = jnp.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])


def test_lhopital_sinc() -> None:
    assert float(lhopital_ratio(_SIN, _X, order=1)) == pytest.approx(1.0, abs=1e-15)


def test_lhopital_one_minus_cos_over_x2() -> None:
    assert float(lhopital_ratio(_ONE_MINUS_COS, _X2, order=2)) == pytest.approx(0.5, abs=1e-15)


def test_limit_of_ratio_autodetects_order() -> None:
    assert float(limit_of_ratio(_SIN, _X)) == pytest.approx(1.0, abs=1e-15)
    assert float(limit_of_ratio(_ONE_MINUS_COS, _X2)) == pytest.approx(0.5, abs=1e-15)


def test_limit_of_ratio_numerator_vanishes_faster_is_zero() -> None:
    # x^2 / x  -> 0
    assert float(limit_of_ratio(_X2, _X)) == 0.0


def test_limit_of_ratio_pole_is_infinite() -> None:
    # x / x^2 -> +inf
    assert math.isinf(float(limit_of_ratio(_X, _X2)))


def test_limit_of_ratio_zero_denominator_raises() -> None:
    zero = jnp.zeros(4)
    with pytest.raises(ValueError, match="vanishes to all"):
        limit_of_ratio(_SIN[:4], zero)


def test_lhopital_ratio_is_differentiable() -> None:
    # d/da  [ (a t) / t ]_{t->0} = d/da a = 1
    def limit_value(a: jax.Array) -> jax.Array:
        num = jnp.array([0.0, 1.0, 0.0]) * a
        den = jnp.array([0.0, 1.0, 0.0])
        return lhopital_ratio(num, den, order=1)

    g = jax.grad(limit_value)(2.0)
    assert float(g) == pytest.approx(1.0, abs=1e-15)


def test_removable_value_is_zeroth_coefficient() -> None:
    jet = jnp.array([3.5, 1.0, -2.0])
    assert float(removable_value(jet)) == 3.5


def test_limit_of_a_learned_field_along_a_ray() -> None:
    # The limit t->0 of a deep tanh MLP along x0 + t v is the network value at x0.
    key = jax.random.PRNGKey(0)
    k1, k2 = jax.random.split(key)
    W1 = jax.random.normal(k1, (4, 2), dtype=jnp.float64)
    b1 = jnp.zeros(4)
    W2 = jax.random.normal(k2, (1, 4), dtype=jnp.float64)
    b2 = jnp.zeros(1)
    x0 = jnp.array([0.3, -0.7])
    v = jnp.array([1.0, 0.5])
    layers = [(W1, b1, "tanh"), (W2, b2, None)]
    jet = mlp_jet(x0, v, layers, order=2)
    direct = (W2 @ jnp.tanh(W1 @ x0 + b1) + b2)
    assert float(removable_value(jet)[0]) == pytest.approx(float(direct[0]), abs=1e-12)


def test_saturation_metadata_is_populated() -> None:
    assert get_activation("tanh").limit_pos_inf == 1.0
    assert get_activation("tanh").limit_neg_inf == -1.0
    assert get_activation("sigmoid").limit_neg_inf == 0.0
    assert get_activation("gaussian").limit_pos_inf == 0.0
    assert saturation_limit(get_activation("arctan"), +1.0) == pytest.approx(math.pi / 2.0)
    assert saturation_limit(get_activation("exp"), -1.0) == 0.0
    # exp diverges as z -> +inf: no finite right asymptote recorded.
    assert saturation_limit(get_activation("exp"), +1.0) is None


# ----- the defining equation: derivative == limit of the difference quotient ----


def test_difference_quotient_limit_equals_directional_derivative() -> None:
    r"""``f'(x0).v == lim_{t->0} (f(x0 + t v) - f(x0)) / t`` for a deep MLP.

    This wires the two primitives against the textbook definition rather than
    against each other:

    * the *limit* operator (:func:`lhopital_ratio`) is applied to the jet of the
      difference quotient ``(f(x0 + t v) - f(x0)) / t``;
    * the *derivative* is taken independently from forward-mode autodiff
      (:func:`jax.jvp`), which shares no code with the sigma-tower jet.

    The closed-form Taylor coefficient ``jet[1]`` (== directional derivative) is
    recovered *exactly* by the limit, and matches autodiff to float64 eps. The
    second-order limit ``2 lim (f(x0+tv) - f(x0) - f'.v t) / t^2`` likewise equals
    the directional second derivative, exercising a genuine ``0/0`` L'Hopital.
    """
    layers, x0, v = _deep_mlp(seed=5)
    f = _mlp_forward(layers)
    order = 3
    jet = mlp_jet(x0, v, layers, order)  # jet[k] = (1/k!) d^k/dt^k f(x0 + t v)

    # N(t) = f(x0 + t v) - f(x0): drop the constant term -> [0, a1, a2, a3].
    num = jet.at[0].set(0.0)
    t_jet = jnp.array([0.0, 1.0, 0.0, 0.0])  # jet of D(t) = t
    limit1 = lhopital_ratio(num, t_jet, order=1)  # the lim operator

    _, jvp = jax.jvp(f, (x0,), (v,))  # independent autodiff derivative

    # lim of the difference quotient reproduces the derivative coefficient exactly
    assert bool(jnp.array_equal(limit1, jet[1]))
    # ... and equals the autodiff directional derivative.
    assert jnp.allclose(limit1, jvp, rtol=1e-12, atol=1e-12)

    # Second order: N2(t) = N(t) - (f'.v) t -> [0, 0, a2, a3]; D2(t) = t^2.
    num2 = num.at[1].set(0.0)
    t2_jet = jnp.array([0.0, 0.0, 1.0, 0.0])
    limit2 = lhopital_ratio(num2, t2_jet, order=2)
    d2 = jax.jvp(lambda x: jax.jvp(f, (x,), (v,))[1], (x0,), (v,))[1]
    assert bool(jnp.array_equal(limit2, jet[2]))
    assert jnp.allclose(2.0 * limit2, d2, rtol=1e-12, atol=1e-12)
