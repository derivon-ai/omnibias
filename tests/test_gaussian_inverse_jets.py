# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Exact, numerical, and analytic views of the same inverse Taylor series."""

from fractions import Fraction as F

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from omnibias.core.gaussian_jet import GaussianInverseSeries
from omnibias.core.verified.gaussian_analytic import certify_gaussian_inverse_series
from omnibias.core.verified.gaussian_heat import GaussianHeatMap
from omnibias.jax.gaussian_jet import gaussian_heat_inverse_jet as jax_jet
from omnibias.torch.gaussian_jet import gaussian_heat_inverse_jet as torch_jet


@pytest.mark.parametrize("adjoint", ["dense", "low_rank", "auto"])
def test_coupled_eighth_order_forward_jets_match_exact_coefficients(adjoint):
    centers, biases = ((0,0),(1,0),(0,1)), (0,F(1,4),F(1,4))
    model = GaussianHeatMap(centers,biases)
    series = GaussianInverseSeries.from_equal_logits(model,(0,0),((1,2),(F(1,3),F(-1,5)),*((0,0),)*6))
    target = np.array(series.target_coefficients,dtype=np.float64)
    options = {"tolerance": 1e-12, "adjoint": adjoint}
    torch_result = torch_jet(torch.tensor(target), centers, [float(b) for b in biases], **options)
    jax_result = jax.jit(lambda v: jax_jet(v, jnp.array(centers,dtype=v.dtype), jnp.array([float(b) for b in biases]), **options))(jnp.asarray(target))
    expected = np.array(series.coefficients,dtype=np.float64)
    assert bool(torch_result.base_result.converged) and bool(jax_result.base_result.converged)
    np.testing.assert_allclose(torch_result.coefficients.detach().numpy(),expected,rtol=2e-9,atol=2e-11)
    np.testing.assert_allclose(jax_result.coefficients,expected,rtol=2e-9,atol=2e-11)
    np.testing.assert_allclose(torch_result.coefficients.detach().numpy(),jax_result.coefficients,rtol=2e-9,atol=2e-11)


@pytest.mark.parametrize("adjoint", ["dense", "low_rank", "auto"])
def test_forward_jet_loss_keeps_all_parameter_and_path_gradients(adjoint):
    data = (np.array([[[.25,-.125],[.1,.3]],[[.2,.1],[-.1,.2]],[[.025,-.05],[.075,.025]],[[0.,0.],[0.,0.]]]),
            np.array([[0.,0.],[1.,0.],[0.,1.]]),np.array([.125,-.25,.25]),np.array(1.5),np.array(2.))
    torch_args = tuple(torch.tensor(value,dtype=torch.float64,requires_grad=True) for value in data)
    jax_args = tuple(jnp.asarray(value,dtype=jnp.float64) for value in data)
    result = torch_jet(*torch_args,adjoint=adjoint,tolerance=1e-12)
    gradients = torch.autograd.grad(result.coefficients.square().sum(),torch_args)
    def loss(*args):
        return jnp.sum(jax_jet(*args,adjoint=adjoint,tolerance=1e-12).coefficients**2)
    expected = jax.jit(jax.grad(loss,argnums=(0,1,2,3,4)))(*jax_args)
    for actual,reference in zip(gradients,expected,strict=True):
        np.testing.assert_allclose(actual.detach().numpy(),reference,rtol=2e-8,atol=2e-10)
    assert np.linalg.norm(gradients[0].detach().numpy()[1:]) > 0.1
    assert abs(float(gradients[2].sum())) < 1e-10


def test_rational_inverse_polynomial_plus_tail_encloses_true_coupled_solution():
    mp = pytest.importorskip("mpmath")
    model = GaussianHeatMap(((0,0),(1,0),(0,1)),(0,F(1,4),F(1,4)))
    series = GaussianInverseSeries.from_equal_logits(model,(0,0),((1,2),*((0,0),)*7))
    certificate = certify_gaussian_inverse_series(series)
    assert certificate is not None and certificate.parameter_radius is not None
    parameter = certificate.parameter_radius/3
    enclosure = certificate.enclose(parameter)
    with mp.workdps(80):
        t = mp.mpf(parameter.numerator)/parameter.denominator
        def residual(x,y):
            ex,ey = mp.exp(x/2),mp.exp(y/2)
            denominator = 1+ex+ey
            return (x-ex/denominator+mp.mpf(1)/3-t,
                    y-ey/denominator+mp.mpf(1)/3-2*t)
        root = mp.findroot(residual,(0,0))
        for bound,value in zip(enclosure,root,strict=True):
            assert mp.mpf(bound.lo) <= value <= mp.mpf(bound.hi)
        approximate = series.evaluate(parameter)
        norm_error = mp.sqrt(sum((mp.mpf(value.numerator)/value.denominator-exact)**2
                                 for value,exact in zip(approximate,root,strict=True)))
        tail = certificate.tail_bound(parameter)
        assert norm_error <= mp.mpf(tail.numerator)/tail.denominator


def test_loose_numerical_base_exposes_changed_expansion_anchor():
    path = np.array([[.3],[.2],[0.],[0.]])
    options = {"max_iterations": 0, "tolerance": 2., "time": 2.}
    result_t = torch_jet(torch.tensor(path),[[-1.],[1.]],[0.,0.],**options)
    result_j = jax_jet(jnp.asarray(path),[[-1.],[1.]],[0.,0.],**options)
    assert bool(result_t.base_result.converged) and bool(result_j.base_result.converged)
    assert abs(float(result_t.base_target_residual.item())) > 0.01
    np.testing.assert_allclose(result_t.anchor_target.detach().numpy(),result_j.anchor_target,rtol=1e-12,atol=1e-12)
    np.testing.assert_allclose(result_t.coefficients.detach().numpy(),result_j.coefficients,rtol=1e-12,atol=1e-12)
