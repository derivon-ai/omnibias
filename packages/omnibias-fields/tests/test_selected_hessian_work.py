# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Selected generic Hessians must not evaluate discarded coordinate entries."""
from __future__ import annotations

import importlib
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from omnibias.fields._core.coords import CoordinateSpec


class _PolynomialField:
    """f(x)=weight*(x0**2*x2+x2**3), with independently known partials."""

    _omnibias_dispatch = "spectral"

    def __init__(self, weight):
        self.weight = weight
        self.calls = []

    def derivative(self, state, name, *, axis, order):
        assert name == "u" and order == 2
        self.calls.append((axis, axis))
        if axis == 0:
            return 2 * self.weight * state.coords[:, 2]
        if axis == 2:
            return 6 * self.weight * state.coords[:, 2]
        raise AssertionError("an unrequested coordinate was evaluated")

    def mixed_partial(self, state, name, axes, orders):
        assert name == "u" and orders == (1, 1)
        assert axes in ((0, 2), (2, 0))
        self.calls.append(axes)
        return 2 * self.weight * state.coords[:, 0]


def _evaluate(backend, coords, weight, axes):
    field = _PolynomialField(weight)
    state = SimpleNamespace(
        field=field, coords=coords,
        coordinate_spec=CoordinateSpec(("x", "y", "z", "w"), time_axis=None),
    )
    ops = importlib.import_module(f"omnibias.fields.{backend}.ops.high_order")
    return ops.hessian(state, "u", axes=axes), field.calls


@pytest.mark.parametrize("backend", ["jax", "torch"])
@pytest.mark.parametrize("axes,indices", [(("z", "x"), (2, 0)), ((2, 0, 2), (2, 0, 2))])
def test_selected_hessian_values_and_only_requested_calls(backend, axes, indices):
    x = np.array([[0.2, 0.3, -0.7, 0.1], [-0.5, 0.2, 0.4, 0.8]])
    coords = jnp.asarray(x) if backend == "jax" else torch.tensor(x)
    result, calls = _evaluate(backend, coords, 1.3, axes)
    expected = np.empty((2, len(indices), len(indices)))
    for i, a in enumerate(indices):
        for j, b in enumerate(indices):
            expected[:, i, j] = 1.3 * (
                (2 if a == 0 else 6) * x[:, 2] if a == b else 2 * x[:, 0]
            )
    np.testing.assert_allclose(np.asarray(result), expected, rtol=1e-14, atol=1e-14)
    assert calls == [(a, b) for a in indices for b in indices]


@pytest.mark.parametrize("backend", ["jax", "torch"])
def test_selected_hessian_retains_parameter_and_coordinate_gradients(backend):
    x = np.array([[0.2, 0.3, -0.7, 0.1], [-0.5, 0.2, 0.4, 0.8]])
    weight = 1.3
    expected_dw = 2 * weight * np.sum(148 * x[:, 2] ** 2 + 16 * x[:, 0] ** 2)
    expected_dx = np.zeros_like(x)
    expected_dx[:, 0] = 32 * weight**2 * x[:, 0]
    expected_dx[:, 2] = 296 * weight**2 * x[:, 2]
    if backend == "jax":
        def loss(w, coords):
            hessian, _ = _evaluate("jax", coords, w, (2, 0, 2))
            return jnp.sum(hessian * hessian)
        dw, dx = jax.jit(jax.grad(loss, argnums=(0, 1)))(weight, jnp.asarray(x))
    else:
        w = torch.tensor(weight, requires_grad=True)
        coords = torch.tensor(x, requires_grad=True)
        hessian, _ = _evaluate("torch", coords, w, (2, 0, 2))
        dw, dx = torch.autograd.grad((hessian * hessian).sum(), (w, coords))
    np.testing.assert_allclose(np.asarray(dw), expected_dw, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(np.asarray(dx), expected_dx, rtol=1e-13, atol=1e-13)
