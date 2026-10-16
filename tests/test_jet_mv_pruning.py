# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Independent identities and derivative checks for shifted-power pruning."""
from __future__ import annotations

import importlib
from fractions import Fraction
from math import factorial

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

torch = pytest.importorskip("torch")
from omnibias.core.multi_index import multi_indices, multiply_table  # noqa: E402


def _dense(u, tower, dim, order, ops):
    """Frozen pre-pruning arithmetic, deliberately including structural zeros."""
    zero = ops.zeros_like(u[0])
    w = [zero] + list(u[1:])
    p = [ops.ones_like(u[0])] + [zero] * (len(u) - 1)
    result = [tower[0] * p[0]] + [zero] * (len(u) - 1)
    fact = 1.0
    for k in range(1, order + 1):
        fact *= k
        acc = [zero] * len(u)
        for g, a, b in multiply_table(dim, order):
            acc[g] = acc[g] + p[a] * w[b]
        p = acc
        for g in range(len(u)):
            result[g] = result[g] + (tower[k] / fact) * p[g]
    return ops.stack(result, 0)


def _tensor(backend, value, dtype="float64"):
    if backend == "jax":
        return jnp.asarray(value, dtype=getattr(jnp, dtype))
    return torch.tensor(value, dtype=getattr(torch, dtype))


def _kernel(backend):
    return importlib.import_module(f"omnibias.{backend}.jet_mv").compose_jet_mv


@pytest.mark.parametrize("backend", ["jax", "torch"])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("dim,order", [(1, 0), (1, 1), (1, 4), (2, 1), (2, 3), (3, 3)])
def test_finite_eager_values_match_dense_accumulation(backend, dtype, dim, order):
    rng = np.random.default_rng(70 + dim + order)
    u = _tensor(backend, rng.normal(size=(len(multi_indices(dim, order)), 2, 3)), dtype)
    tower = _tensor(backend, rng.normal(size=(order + 1, 2, 3)), dtype)
    ops = jnp if backend == "jax" else torch
    result = _kernel(backend)(u, tower, dim, order)
    reference = _dense(u, tower, dim, order, ops)
    np.testing.assert_array_equal(np.asarray(result), np.asarray(reference))


def _exact_dictionary_composition(indices, u, tower, order):
    """Fraction polynomial arithmetic without the shared multiplication table."""
    zero = (0,) * len(indices[0])
    w = {a: c for a, c in zip(indices, u, strict=True) if a != zero}
    power = {zero: Fraction(1)}
    result = {a: Fraction(0) for a in indices}
    for k, derivative in enumerate(tower):
        for a, coefficient in power.items():
            result[a] += derivative * coefficient / factorial(k)
        next_power = {}
        for a, ca in power.items():
            for b, cb in w.items():
                exponent = tuple(x + y for x, y in zip(a, b, strict=True))
                if sum(exponent) <= order:
                    next_power[exponent] = next_power.get(exponent, Fraction(0)) + ca * cb
        power = next_power
    return [float(result[a]) for a in indices]


@pytest.mark.parametrize("backend", ["jax", "torch"])
def test_fraction_polynomial_oracle_and_directional_restriction(backend):
    for dim, order in [(1, 0), (1, 1), (1, 5), (2, 4), (3, 3)]:
        indices = multi_indices(dim, order)
        u = [Fraction((i % 5) - 2, 8) for i in range(len(indices))]
        tower = [Fraction((-1)**i * (i + 1), 4) for i in range(order + 1)]
        u_array = _tensor(backend, [float(x) for x in u])
        tower_array = _tensor(backend, [float(x) for x in tower])
        result = _kernel(backend)(u_array, tower_array, dim, order)
        expected = _exact_dictionary_composition(indices, u, tower, order)
        np.testing.assert_allclose(np.asarray(result), expected, rtol=2e-15, atol=2e-15)
        if dim == 1:
            directional = importlib.import_module(f"omnibias.{backend}.jet").compose_jet
            np.testing.assert_array_equal(np.asarray(result), np.asarray(directional(u_array, tower_array)))


@pytest.mark.parametrize("backend", ["jax", "torch"])
def test_complete_first_gradients_and_changed_runtime_coefficients(backend):
    dim, order = 2, 3
    rng = np.random.default_rng(320)
    u_np = rng.normal(size=(len(multi_indices(dim, order)), 3)) / 4
    t_np = rng.normal(size=(order + 1, 3))
    cotangent = _tensor(backend, rng.normal(size=u_np.shape))
    kernel = _kernel(backend)
    if backend == "jax":
        def loss(u, tower, dense):
            output = _dense(u, tower, dim, order, jnp) if dense else kernel(u, tower, dim, order)
            return jnp.sum(output * cotangent)
        optimized = jax.jit(jax.value_and_grad(lambda u, t: loss(u, t, False), argnums=(0, 1)))
        original = jax.jit(jax.value_and_grad(lambda u, t: loss(u, t, True), argnums=(0, 1)))
        for shift in (0.0, 0.17):
            got = optimized(jnp.asarray(u_np + shift), jnp.asarray(t_np - shift))
            expected = original(jnp.asarray(u_np + shift), jnp.asarray(t_np - shift))
            for a, b in zip(jax.tree.leaves(got), jax.tree.leaves(expected), strict=True):
                np.testing.assert_allclose(a, b, rtol=5e-14, atol=5e-14)
        batched = jax.jit(jax.vmap(lambda u, t: kernel(u, t, dim, order)))
        stacked = batched(jnp.stack([u_np, u_np + 0.17]), jnp.stack([t_np, t_np - 0.17]))
        np.testing.assert_allclose(stacked[1], kernel(u_np + 0.17, t_np - 0.17, dim, order),
                                   rtol=5e-14, atol=5e-14)
    else:
        for shift in (0.0, 0.17):
            u = torch.tensor(u_np + shift, requires_grad=True)
            tower = torch.tensor(t_np - shift, requires_grad=True)
            actual = kernel(u, tower, dim, order)
            reference = _dense(u, tower, dim, order, torch)
            actual_grad = torch.autograd.grad((actual * cotangent).sum(), (u, tower))
            ref_grad = torch.autograd.grad((reference * cotangent).sum(), (u, tower))
            for a, b in zip(actual_grad, ref_grad, strict=True):
                np.testing.assert_allclose(a, b, rtol=5e-14, atol=5e-14)


@pytest.mark.parametrize("backend", ["jax", "torch"])
def test_nonfinite_highest_tower_entry_cannot_pollute_lower_orders(backend):
    dim, order = 2, 3
    indices = multi_indices(dim, order)
    u = _tensor(backend, [0.125] * len(indices))
    tower = _tensor(backend, [0.25, 0.5, -0.75, np.inf])
    reference_tower = _tensor(backend, [0.25, 0.5, -0.75, 0.0])
    kernel = _kernel(backend)
    actual = kernel(u, tower, dim, order)
    reference = kernel(u, reference_tower, dim, order)
    lower = [i for i, a in enumerate(indices) if sum(a) < order]
    np.testing.assert_array_equal(np.asarray(actual)[lower], np.asarray(reference)[lower])
    if backend == "jax":
        compiled = jax.jit(lambda a, b: kernel(a, b, dim, order))(u, tower)
        np.testing.assert_array_equal(np.asarray(compiled)[lower], np.asarray(reference)[lower])
