# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Transformable estimator values and explicit, conditional host reports."""
from __future__ import annotations

import importlib
import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")
jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from omnibias.core.contraction import PolylaplacianReport
from omnibias.core.verified.sampled import hoeffding_enclosure


@pytest.fixture(autouse=True)
def _enable_jit_for_transform_checks():
    """Exercise compilation even if another package's tests disabled JIT."""
    with jax.disable_jit(False):
        yield


def case(backend, batched=False):
    arrays = [np.array([[.3, .2], [-.1, .4]]), np.array([.1, -.2]),
              np.array([[.7, -.4]]), np.array([.03])]
    convert = (lambda a: torch.tensor(a, dtype=torch.float64, requires_grad=True)) if backend == "torch" else jnp.asarray
    params = tuple(convert(a) for a in arrays)
    x = np.array([[.1, .2], [-.3, .4]]) if batched else np.array([.1, .2])
    x = torch.tensor(x, dtype=torch.float64) if backend == "torch" else jnp.asarray(x)
    return x, params


def layers(params):
    w, b, c, d = params
    return [(w, b, "tanh"), (c, d, None)]


def numpy(value):
    return value.detach().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)


@pytest.mark.parametrize("backend", ["torch", "jax"])
@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("directions", [1, 8])
def test_value_and_report_share_samples_without_inferred_bounds(backend, batched, directions):
    module = importlib.import_module(f"omnibias.{backend}.laplacian")
    x, params = case(backend, batched)
    args = dict(mode="estimator", n_directions=directions, seed=19)
    expected = module.deep_field_polylaplacian(x, layers(params), 2, **args)
    actual, report = module.deep_field_polylaplacian_with_report(x, layers(params), 2, **args)
    np.testing.assert_array_equal(numpy(actual), numpy(expected))
    assert (report.mode, report.dim, report.k, report.n_directions) == ("estimator", 2, 2, directions)
    assert report.concentration is None
    # Constructor compatibility: the existing positional fields are unchanged.
    assert PolylaplacianReport("support", 2, 2, 20000).concentration is None


@pytest.mark.parametrize("backend", ["torch", "jax"])
def test_value_only_never_constructs_a_host_report(backend, monkeypatch):
    module = importlib.import_module(f"omnibias.{backend}.laplacian")
    x, params = case(backend)

    def forbidden(*args, **kwargs):
        raise AssertionError("host report entered the value path")

    monkeypatch.setattr(module, "_tier_c_concentration", forbidden)
    value = module.deep_field_polylaplacian(x, layers(params), 2, mode="estimator", n_directions=4)
    assert np.isfinite(numpy(value)).all()


@pytest.mark.parametrize("backend", ["torch", "jax"])
def test_explicit_range_uses_normalized_samples_for_first_component(backend):
    module = importlib.import_module(f"omnibias.{backend}.laplacian")
    x, params = case(backend, batched=True)
    args = dict(mode="estimator", n_directions=8, seed=19)
    # A deliberately loose valid range for this fixed small network. The fourth
    # tanh derivative has coefficient 1-norm 80 and all weight norms are <1.
    value, report = module.deep_field_polylaplacian_with_report(
        x, layers(params), 2, value_range=(-1e6, 1e6), delta=.01, **args)
    _, samples = module._polylaplacian_estimator(x, layers(params), 2, n_directions=8, seed=19, return_samples=True)
    expected = hoeffding_enclosure(numpy(samples)[0, :, 0].tolist(), value_range=(-1e6, 1e6), delta=.01)
    assert report.concentration == expected
    assert np.isfinite(numpy(value)).all()
    if backend == "torch":
        grads = torch.autograd.grad(value.square().sum(), params, allow_unused=True, materialize_grads=True)
        assert all(bool(torch.isfinite(g).all()) for g in grads)


@pytest.mark.parametrize("backend", ["torch", "jax"])
@pytest.mark.parametrize("value_range,delta", [((1., -1.), .1), ((-math.inf, 1.), .1),
                                               ((-1e6, 1e6), 0.), ((-1e6, 1e6), 1.),
                                               ((1e6, 2e6), .1)])
def test_invalid_declared_bound_rejected(backend, value_range, delta):
    module = importlib.import_module(f"omnibias.{backend}.laplacian")
    x, params = case(backend)
    with pytest.raises(ValueError):
        module.deep_field_polylaplacian_with_report(x, layers(params), 2, mode="estimator",
                                                    n_directions=4, value_range=value_range, delta=delta)


@pytest.mark.parametrize("backend", ["torch", "jax"])
@pytest.mark.parametrize("directions", [0, -1])
def test_estimator_rejects_empty_direction_set(backend, directions):
    module = importlib.import_module(f"omnibias.{backend}.laplacian")
    x, params = case(backend)
    with pytest.raises(ValueError, match="n_directions"):
        module.deep_field_polylaplacian(x, layers(params), 2, mode="estimator", n_directions=directions)


@pytest.mark.parametrize("backend", ["torch", "jax"])
def test_exact_modes_do_not_construct_concentration(backend):
    module = importlib.import_module(f"omnibias.{backend}.laplacian")
    x, params = case(backend)
    for k, mode in [(1, "auto"), (2, "support")]:
        _, report = module.deep_field_polylaplacian_with_report(
            x, layers(params), k, mode=mode, value_range=(-1e6, 1e6))
        assert report.concentration is None
        assert report.n_directions is None
    _, report = module.deep_field_polylaplacian_with_report(
        x, layers(params), 2, mode="auto", budget=1, n_directions=4)
    assert report.mode == "estimator"


def test_jax_estimator_jit_vmap_and_all_parameter_gradients():
    from omnibias.jax.laplacian import deep_field_polylaplacian
    x, params = case("jax")
    key = jax.random.PRNGKey(19)

    def estimate(point, parameters, key):
        return deep_field_polylaplacian(point, layers(parameters), 2, mode="estimator", n_directions=8, key=key)

    compiled = jax.jit(estimate)
    actual = compiled(x, params, key)
    np.testing.assert_allclose(actual, estimate(x, params, key), rtol=2e-12, atol=2e-12)
    batched = jax.vmap(estimate, in_axes=(0, None, None))(jnp.stack([x, -x]), params, key)
    assert batched.shape == (2, 1)
    assert not np.array_equal(np.asarray(actual), np.asarray(compiled(x, params, jax.random.PRNGKey(20))))
    assert compiled._cache_size() == 1  # runtime key changes do not retrace.

    # Independent reference: repeatedly differentiate the ordinary tanh field
    # along a scalar path, then average the same explicitly reproduced draws.
    g = jax.random.normal(key, (8, 2), dtype=x.dtype)
    directions = g / jnp.linalg.norm(g, axis=-1, keepdims=True)

    def reference(parameters):
        w, b, c, d = parameters
        def one(v):
            def f(t):
                return (jnp.tanh(w @ (x + t * v) + b) @ c[0] + d[0])
            for _ in range(4):
                f = jax.grad(f)
            return f(jnp.asarray(0., dtype=x.dtype))
        return (8. / 3.) * jax.vmap(one)(directions).mean()

    np.testing.assert_allclose(actual[0], reference(params), rtol=2e-11, atol=2e-12)
    a = jax.jit(jax.grad(lambda parameters: (estimate(x, parameters, key).sum() - .7)**2))(params)
    b = jax.grad(lambda parameters: (reference(parameters) - .7)**2)(params)
    for left, right in zip(a, b, strict=True):
        assert np.isfinite(np.asarray(left)).all()
        np.testing.assert_allclose(left, right, rtol=2e-10, atol=2e-11)


def test_torch_estimator_all_parameter_gradients_against_nested_ad():
    from omnibias.torch.laplacian import deep_field_polylaplacian
    x, params = case("torch")
    actual = deep_field_polylaplacian(x, layers(params), 2, mode="estimator", n_directions=8, seed=19).sum()
    g = torch.randn(8, 2, dtype=x.dtype, generator=torch.Generator().manual_seed(19))
    directions = g / torch.linalg.norm(g, dim=-1, keepdim=True)
    values = []
    w, b, c, d = params
    for v in directions:
        t = torch.zeros((), dtype=x.dtype, requires_grad=True)
        value = torch.tanh(w @ (x + t * v) + b) @ c[0] + d[0]
        for _ in range(4):
            value = torch.autograd.grad(value, t, create_graph=True)[0]
        values.append(value)
    expected = (8. / 3.) * torch.stack(values).mean()
    torch.testing.assert_close(actual, expected, rtol=2e-11, atol=2e-12)
    left = torch.autograd.grad((actual-.7).square(), params, allow_unused=True, materialize_grads=True)
    right = torch.autograd.grad((expected-.7).square(), params, allow_unused=True, materialize_grads=True)
    for a, b in zip(left, right, strict=True):
        assert bool(torch.isfinite(a).all())
        torch.testing.assert_close(a, b, rtol=2e-10, atol=2e-11)


@pytest.mark.parametrize("backend", ["torch", "jax"])
def test_d5000_laplacian_retains_every_parameter_gradient(backend):
    module = importlib.import_module(f"omnibias.{backend}.laplacian")
    arrays = [np.full((4, 5000), .002), np.full(4, .1), np.full((4, 4), .1),
              np.full(4, .2), np.full((1, 4), .3), np.full(1, .1)]
    specs = ["tanh", "tanh", None]
    if backend == "torch":
        params = tuple(torch.tensor(a, dtype=torch.float64, requires_grad=True) for a in arrays)
        x = torch.linspace(-.1, .1, 10000, dtype=torch.float64).reshape(2, 5000)
        field = [(params[2*i], params[2*i+1], spec) for i, spec in enumerate(specs)]
        value = module.deep_field_laplacian(x, field)
        grads = torch.autograd.grad((value-.7).square().mean(), params, allow_unused=True, materialize_grads=True)
    else:
        params = tuple(jnp.asarray(a) for a in arrays)
        x = jnp.linspace(-.1, .1, 10000).reshape(2, 5000)
        def loss(parameters):
            field = [(parameters[2*i], parameters[2*i+1], spec) for i, spec in enumerate(specs)]
            return ((module.deep_field_laplacian(x, field)-.7)**2).mean()
        grads = jax.jit(jax.grad(loss))(params)
    assert len(grads) == len(params)
    assert all(np.isfinite(numpy(g)).all() for g in grads)
    assert np.max(np.abs(numpy(grads[0]))) > 0
