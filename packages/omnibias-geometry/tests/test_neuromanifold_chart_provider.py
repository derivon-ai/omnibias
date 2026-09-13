# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Supplied chart derivatives preserve live metrics and the fallback contract."""

from dataclasses import replace

import numpy as np
import pytest
from omnibias.geometry import ChartSpec


@pytest.mark.parametrize("shape", [(5, 3), (1, 2), (2,)])
def test_supplied_jacobian_shape_must_match_chart_in_both_backends(shape):
    torch = pytest.importorskip("torch")
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp
    from omnibias.geometry.jax.ops.pullback import pullback_metric as jpullback
    from omnibias.geometry.torch.ops.pullback import pullback_metric as tpullback

    tchart = ChartSpec(
        lambda x: torch.cat((x, x)),
        1,
        2,
        jacobian=lambda x: torch.ones(shape, dtype=x.dtype, device=x.device),
    )
    jchart = ChartSpec(
        lambda x: jnp.concatenate((x, x)), 1, 2, jacobian=lambda x: jnp.ones(shape, dtype=x.dtype)
    )
    with pytest.raises(ValueError, match="Jacobian must have shape"):
        tpullback(torch.ones((3, 1)), tchart)
    with pytest.raises(ValueError, match="Jacobian must have shape"):
        jax.jit(lambda x: jpullback(x, jchart))(jnp.ones((3, 1)))


def test_torch_provider_bypasses_forward_ad_and_preserves_training_graph():
    torch = pytest.importorskip("torch")
    from omnibias.geometry.torch.ops.pullback import metric_spec_from_chart, pullback_metric

    scale = torch.tensor(1.3, dtype=torch.float64, requires_grad=True)
    coords = torch.tensor([[-0.7], [0.2], [0.8]], dtype=torch.float64, requires_grad=True)

    def unused_phi(x):
        raise AssertionError("Euclidean supplied-Jacobian path must not differentiate phi")

    def jacobian(x):
        return torch.stack((scale + 0 * x[0], 2 * x[0])).reshape(2, 1)

    chart = ChartSpec(
        unused_phi, 1, 2, jacobian=jacobian, derivative_provenance="analytic polynomial"
    )
    actual = pullback_metric(coords, chart)
    expected = (scale**2 + 4 * coords[:, 0] ** 2).reshape(-1, 1, 1)
    torch.testing.assert_close(actual, expected)
    dscale, dx = torch.autograd.grad(actual.sum(), (scale, coords), create_graph=True)
    torch.testing.assert_close(dscale, 2 * len(coords) * scale)
    torch.testing.assert_close(dx, 8 * coords)
    second = torch.autograd.grad(dscale, scale)[0]
    torch.testing.assert_close(second, torch.tensor(2 * len(coords), dtype=scale.dtype))
    torch.testing.assert_close(metric_spec_from_chart(chart).g_point(coords[0]), actual[0])


@pytest.mark.parametrize("supplied", [False, True])
def test_torch_ambient_metric_and_forward_ad_fallback_are_live(supplied):
    torch = pytest.importorskip("torch")
    from omnibias.geometry.torch.ops.pullback import pullback_metric

    scale = torch.tensor(1.2, dtype=torch.float64, requires_grad=True)
    coords = torch.tensor([[-0.5], [0.25], [0.75]], dtype=scale.dtype)

    def phi(x):
        return torch.stack((scale * x[0], x[0] ** 2))

    def jacobian(x):
        return torch.stack((scale + 0 * x[0], 2 * x[0])).reshape(2, 1)

    def ambient(y):
        return torch.diag(torch.stack((1 + y[0] ** 2, torch.ones_like(y[0]))))

    chart = ChartSpec(phi, 1, 2, ambient_metric=ambient)
    if supplied:
        chart = replace(chart, jacobian=jacobian, derivative_provenance="analytic polynomial")
    metric = pullback_metric(coords, chart)
    expected = scale**2 + (scale**4 + 4) * coords[:, 0] ** 2
    torch.testing.assert_close(metric[:, 0, 0], expected)
    gradient = torch.autograd.grad(metric.sum(), scale)[0]
    torch.testing.assert_close(gradient, 2 * len(coords) * scale + 4 * scale**3 * (coords**2).sum())


@pytest.mark.parametrize("device", ["cpu", "meta"])
@pytest.mark.parametrize("dtype_name", ["float32", "float64"])
def test_torch_euclidean_ambient_inherits_dtype_and_device(device, dtype_name):
    torch = pytest.importorskip("torch")
    from omnibias.geometry.torch.ops.pullback import euclidean_ambient_metric, pullback_metric

    dtype = getattr(torch, dtype_name)
    coords = torch.ones((3, 1), device=device, dtype=dtype)

    def phi(x):
        return torch.stack((x[0], x[0] ** 2))

    def jacobian(x):
        return torch.stack((torch.ones_like(x[0]), 2 * x[0])).reshape(2, 1)

    chart = ChartSpec(phi, 1, 2, ambient_metric=euclidean_ambient_metric(2), jacobian=jacobian)
    result = pullback_metric(coords, chart)
    assert result.shape == (3, 1, 1)
    assert result.dtype == dtype and result.device == coords.device


def test_jax_provider_bypasses_forward_ad_and_is_jittable_and_live():
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from omnibias.geometry.jax.ops.pullback import metric_spec_from_chart, pullback_metric

    coords = jnp.array([[-0.7], [0.2], [0.8]])

    def build(scale):
        def unused_phi(x):
            raise AssertionError("supplied Euclidean Jacobian must bypass phi")

        def jacobian(x):
            return jnp.stack((scale + 0 * x[0], 2 * x[0])).reshape(2, 1)

        return ChartSpec(
            unused_phi, 1, 2, jacobian=jacobian, derivative_provenance="analytic polynomial"
        )

    def loss(scale):
        return pullback_metric(coords, build(scale)).sum()

    scale = jnp.asarray(1.3)
    value, gradient = jax.jit(jax.value_and_grad(loss))(scale)
    np.testing.assert_allclose(value, 3 * scale**2 + 4 * jnp.square(coords).sum(), rtol=1e-13)
    np.testing.assert_allclose(gradient, 6 * scale, rtol=1e-13)
    np.testing.assert_allclose(jax.grad(jax.grad(loss))(scale), 6, rtol=1e-13)
    np.testing.assert_allclose(
        metric_spec_from_chart(build(scale)).g_point(coords[0]), [[scale**2 + 4 * 0.7**2]]
    )


@pytest.mark.parametrize("supplied", [False, True])
def test_jax_ambient_metric_and_fallback_keep_parameter_derivatives(supplied):
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from omnibias.geometry.jax.ops.pullback import pullback_metric

    coords = jnp.array([[-0.5], [0.25], [0.75]])

    def metric(scale):
        def phi(x):
            return jnp.stack((scale * x[0], x[0] ** 2))

        def jacobian(x):
            return jnp.stack((scale + 0 * x[0], 2 * x[0])).reshape(2, 1)

        def ambient(y):
            return jnp.diag(jnp.stack((1 + y[0] ** 2, jnp.ones_like(y[0]))))

        chart = ChartSpec(
            phi, 1, 2, ambient_metric=ambient, jacobian=jacobian if supplied else None
        )
        return pullback_metric(coords, chart)

    scale = jnp.asarray(1.2)
    actual = jax.jit(metric)(scale)
    expected = scale**2 + (scale**4 + 4) * coords[:, 0] ** 2
    np.testing.assert_allclose(actual[:, 0, 0], expected, rtol=1e-13)
    gradient = jax.grad(lambda s: metric(s).sum())(scale)
    np.testing.assert_allclose(
        gradient, 6 * scale + 4 * scale**3 * jnp.square(coords).sum(), rtol=1e-13
    )
    assert actual.dtype == coords.dtype and actual.device == coords.device
