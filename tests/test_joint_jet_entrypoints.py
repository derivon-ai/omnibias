# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Existing jet entry points preserve the realization graph and dtype."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from omnibias.core.multi_index import num_multi_indices
from omnibias.core.realization import RealizationSpec
from omnibias.jax import jet as jj
from omnibias.jax import jet_mv as jmv
from omnibias.jax import realization as jr
from omnibias.torch import jet as tj
from omnibias.torch import jet_mv as tmv
from omnibias.torch import realization as tr

jax.config.update("jax_enable_x64", True)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("dim", [1, 2])
def test_preseeded_joint_mlp_wrappers_preserve_values_live_gradients_and_dtype(dtype, dim):
    spec = RealizationSpec.dense((2, 2, 1))
    order = 2
    count = num_multi_indices(dim, order)
    x = np.linspace(-0.2, 0.3, count * 2, dtype=dtype).reshape(count, 2)
    p = np.linspace(-0.3, 0.4, count * spec.layout.size, dtype=dtype).reshape(count, -1)
    tx = torch.tensor(x, requires_grad=True)
    tp = torch.tensor(p, requires_grad=True)
    options = {"order": order, "max_coefficients": 1000}
    if dim == 1:
        twrapper, jwrapper = tj.mlp_joint_jet, jj.mlp_joint_jet
    else:
        twrapper, jwrapper = tmv.mlp_joint_jet_mv, jmv.mlp_joint_jet_mv
        options["dim"] = dim
    actual = twrapper(tx, tp, spec, **options)
    expected = tr.realization_jet(tx, tp, spec, dim=dim, order=order, max_coefficients=1000)
    assert actual.dtype == tx.dtype
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    gradients = torch.autograd.grad(actual.square().sum(), (tx, tp), retain_graph=True)
    reference = torch.autograd.grad(expected.square().sum(), (tx, tp))
    for got, want in zip(gradients, reference, strict=True):
        assert torch.isfinite(got).all() and torch.count_nonzero(got)
        torch.testing.assert_close(got, want, rtol=0, atol=0)

    def wrapped(a, b):
        return jwrapper(a, b, spec, **options)

    def direct(a, b):
        return jr.realization_jet(a, b, spec, dim=dim, order=order, max_coefficients=1000)

    jx, jp = jnp.asarray(x), jnp.asarray(p)
    result = jax.jit(wrapped)(jx, jp)
    assert result.dtype == jx.dtype
    np.testing.assert_array_equal(result, jax.jit(direct)(jx, jp))
    jgrads = jax.jit(jax.grad(lambda a, b: jnp.sum(wrapped(a, b) ** 2), argnums=(0, 1)))(jx, jp)
    references = jax.jit(jax.grad(lambda a, b: jnp.sum(direct(a, b) ** 2), argnums=(0, 1)))(jx, jp)
    for got, want in zip(jgrads, references, strict=True):
        assert np.isfinite(got).all() and np.count_nonzero(got)
        np.testing.assert_array_equal(got, want)
    tolerance = 2e-6 if dtype is np.float32 else 2e-12
    np.testing.assert_allclose(result, actual.detach().numpy(), rtol=tolerance, atol=tolerance)
    for wrapper, a, b in ((twrapper, tx, tp), (jwrapper, jx, jp)):
        with pytest.raises(ValueError, match="budget"):
            wrapper(a, b, spec, **{**options, "max_coefficients": 1})


def test_existing_multivariate_parameter_jet_entrypoints_preserve_all_live_directions():
    spec = RealizationSpec.dense((2, 2, 1))
    x = torch.tensor([0.3, -0.2], dtype=torch.float64, requires_grad=True)
    theta = torch.linspace(-0.4, 0.6, spec.layout.size, dtype=torch.float64, requires_grad=True)
    basis = torch.full((spec.layout.size, 2), 0.15, dtype=torch.float64, requires_grad=True)
    input_basis = torch.tensor([[0.2, 0.1], [-0.1, 0.4]], dtype=torch.float64, requires_grad=True)
    args = (x, theta, basis, input_basis)

    def torch_call(fn):
        return fn(x, theta, spec, basis, 2, input_basis=input_basis, max_coefficients=1000)

    actual, expected = torch_call(tmv.parameter_jet), torch_call(tr.parameter_jet)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    gradients = torch.autograd.grad(actual.square().sum(), args, retain_graph=True)
    reference = torch.autograd.grad(expected.square().sum(), args)
    for got, want in zip(gradients, reference, strict=True):
        assert torch.isfinite(got).all() and torch.count_nonzero(got)
        torch.testing.assert_close(got, want, rtol=0, atol=0)

    jargs = tuple(jnp.asarray(arg.detach().numpy()) for arg in args)

    def loss(fn, a, b, directions, input_directions):
        return jnp.sum(
            fn(a, b, spec, directions, 2, input_basis=input_directions, max_coefficients=1000) ** 2
        )

    first = jax.jit(jax.grad(lambda *a: loss(jmv.parameter_jet, *a), argnums=(0, 1, 2, 3)))(*jargs)
    second = jax.jit(jax.grad(lambda *a: loss(jr.parameter_jet, *a), argnums=(0, 1, 2, 3)))(*jargs)
    for got, want in zip(first, second, strict=True):
        assert np.isfinite(got).all() and np.count_nonzero(got)
        np.testing.assert_array_equal(got, want)
    with pytest.raises(ValueError, match="budget"):
        tmv.parameter_jet(x, theta, spec, basis, 2, max_coefficients=1)
    with pytest.raises(ValueError, match="budget"):
        jmv.parameter_jet(jargs[0], jargs[1], spec, jargs[2], 2, max_coefficients=1)
