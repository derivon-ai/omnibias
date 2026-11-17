# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Fixed-age resummation, ordinary auxiliary coefficients and live-parameter AD."""

from fractions import Fraction

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from omnibias.jax.gaussian_constrained_aging import gaussian_heat_inverse_constrained_aging
from omnibias.jax.gaussian_constrained_approximation import GaussianConstrainedApproximationResult
from omnibias.jax.gaussian_constrained_approximation import (
    gaussian_heat_inverse_constrained_approximation as approximate,
)
from omnibias.jax.gaussian_predictor import gaussian_heat_inverse_predictor

jax.config.update('jax_enable_x64',True)


def _args():
    return tuple(jnp.asarray(v,dtype=jnp.float64) for v in (
        [.2,-.15],[[.1,-.2],[.8,.3],[-.2,.6]],[.2,-.1,.3],[[1.,2.]],[.3],.8,8.))


@pytest.mark.parametrize('dtype',[jnp.float32,jnp.float64])
def test_jit_auxiliary_coefficients_match_independent_exact_picard_fixture(dtype):
    exact=[Fraction(0),Fraction(-5,216),Fraction(-245,124416),Fraction(-4655,35831808),
           Fraction(-257555,41278242816),Fraction(-237013,3962711310336),
           Fraction(76827149,2282521714753536),Fraction(24734280845,4601563776943128576),
           Fraction(2483951492605,4543715546604414959616)]
    # Normal costs(1,0,1) are absorbed exactly by these biases at age8.
    result=jax.jit(approximate)(jnp.zeros(2,dtype=dtype),jnp.array([[0.,0.],[1.,1.],[3.,2.]]),
                                jnp.array([1/32,0.,1/32]),jnp.array([[0.,1.]]),jnp.ones(1),time=8)
    tolerance=2e-4 if dtype==jnp.float32 else 2e-12
    np.testing.assert_allclose(result.correction_coefficients[:,0],[float(v) for v in exact],rtol=tolerance,atol=0)
    np.testing.assert_array_equal(result.correction_coefficients[:,1],jnp.zeros(9))
    np.testing.assert_allclose(result.value,[float(Fraction(4,3)+sum(exact)),1.],rtol=tolerance,atol=tolerance)
    assert bool(result.valid) and result.contraction_bound==9/64
    assert result.value.dtype==result.approximation_error_bound.dtype==dtype


@pytest.mark.parametrize('order',[0,1,4,8])
def test_large_original_Q_and_small_q_with_independent_physical_root(order):
    mp=pytest.importorskip('mpmath')
    args=(jnp.zeros(2),jnp.array([[-1.,-1.],[1.,1.]]),jnp.zeros(2),jnp.array([[0.,1.]]),jnp.array([10.]))
    old=gaussian_heat_inverse_constrained_aging(*args,time=4)
    assert not bool(old.valid) and old.aging_parameter==11/8
    result=jax.jit(lambda *p:approximate(*p,time=4,order=order))(*args)
    assert bool(result.valid) and result.contraction_bound==1/8 and result.value[1]==10
    q=Fraction(1,8)
    expected=q if order==0 else (2*q*q if order==1 else (3*q)**(order+1)/(1-3*q))
    assert float(result.approximation_error_bound)==pytest.approx(float(expected),rel=2e-14)
    with mp.workdps(70):
        root=mp.findroot(lambda x:x-mp.tanh((x+10)/8),.9)
        assert abs(root-mp.mpf(float(result.value[0])))<mp.mpf(float(result.approximation_error_bound))


@pytest.mark.parametrize('order',[0,1,4])
def test_all_seven_runtime_parameter_gradients_and_higher_derivatives(order):
    args=_args()
    def loss(*p):
        result=approximate(*p,order=order)
        weights=jnp.arange(order+1,dtype=p[0].dtype)[:,None]
        return jnp.sum(result.value**2)+jnp.sum(result.correction_coefficients*weights)
    grads=jax.jit(jax.grad(loss,argnums=tuple(range(7))))(*args)
    step=2e-5
    for parameter,gradient in enumerate(grads):
        expected=np.empty(args[parameter].shape)
        for index in np.ndindex(expected.shape):
            plus,minus=[np.array(v) for v in args],[np.array(v) for v in args]
            plus[parameter][index]+=step
            minus[parameter][index]-=step
            expected[index]=(float(loss(*plus))-float(loss(*minus)))/(2*step)
        np.testing.assert_allclose(gradient,expected,rtol=4e-7,atol=4e-9)
    def combined(z):
        return loss(*args[:3],z[:2].reshape(1,2),z[2:3],z[3],z[4])
    hessian=jax.jit(jax.hessian(combined))(jnp.concatenate((args[3][0],args[4],jnp.stack(args[5:]))))
    assert np.isfinite(hessian).all()
    np.testing.assert_allclose(hessian,hessian.T,rtol=3e-11,atol=3e-12)
    diagnostic=jax.grad(lambda centers:approximate(args[0],centers,*args[2:],order=order).approximation_error_bound)(args[1])
    np.testing.assert_array_equal(diagnostic,jnp.zeros_like(args[1]))


@pytest.mark.parametrize('order,expected',[(0,-1.),(1,-1.5)])
def test_runtime_age_derivative_differs_from_frozen_auxiliary_derivative(order,expected):
    target=jnp.zeros(2)
    centers=jnp.array([[-1.,-2.],[1.,2.]])
    biases=jnp.array([2.25,.25])
    C,rhs=jnp.array([[0.,1.]]),jnp.ones(1)
    def live(age):
        return approximate(target,centers,biases,C,rhs,time=age,order=order).value[0]
    def frozen(age):
        return gaussian_heat_inverse_predictor(target,jnp.array([[-1.,1.],[1.,1.]]),jnp.zeros(2),time=age,order=order).value[0]
    np.testing.assert_allclose(jax.jit(jax.grad(live))(jnp.asarray(1.)),expected,rtol=2e-13,atol=2e-13)
    np.testing.assert_allclose(jax.grad(frozen)(jnp.asarray(1.)),0.,rtol=0,atol=0)
    assert np.isfinite(jax.jit(jax.grad(jax.grad(live)))(jnp.asarray(1.)))


def test_jit_vmap_batches_live_constraints_affine_and_empty_queries():
    args=_args()
    target=jnp.stack((args[0],2*args[0],args[0]/2))
    result=jax.jit(approximate)(target,*args[1:])
    out_axes=GaussianConstrainedApproximationResult(0,1,0,None,0,None,0)
    mapped=jax.vmap(lambda v:approximate(v,*args[1:]),out_axes=out_axes)(target)
    np.testing.assert_allclose(result.value,mapped.value,rtol=2e-13,atol=2e-13)
    np.testing.assert_allclose(result.correction_coefficients,mapped.correction_coefficients,rtol=3e-12,atol=3e-14)
    constraints=jnp.stack((args[3],args[3]+jnp.array([[.2,-.1]])))
    varied=jax.jit(jax.vmap(lambda C:approximate(*args[:3],C,*args[4:])))(constraints)
    assert bool(varied.valid.all())
    empty=approximate(target[:0],*args[1:])
    assert empty.correction_coefficients.shape==(9,0,2)
    affine=approximate(jnp.zeros(2),jnp.array([[0.,-10.],[0.,10.]]),jnp.zeros(2),jnp.array([[0.,1.]]),jnp.array([10.]))
    assert bool(affine.valid) and affine.contraction_bound==0 and affine.approximation_error_bound==0
    np.testing.assert_array_equal(affine.correction_coefficients,jnp.zeros((9,2)))


def test_partial_affine_center_derivatives_use_transformed_posterior():
    target=jnp.zeros(2)
    centers=jnp.array([[0.,-.2],[0.,.3]])
    def value(a):
        return approximate(target,a,jnp.zeros(2),jnp.array([[0.,1.]]),jnp.array([.1]),time=8,order=4).value[0]
    p=jax.nn.softmax(-((centers[:,1]-.1)**2)/32)
    gradient=jax.jit(jax.grad(value))(centers)
    np.testing.assert_allclose(gradient[:,0],p,rtol=2e-13,atol=2e-13)
    hessian=jax.jit(jax.hessian(value))(centers)
    assert np.isfinite(hessian).all()
    np.testing.assert_allclose(hessian.reshape(4,4),hessian.reshape(4,4).T,rtol=2e-12,atol=2e-13)


def test_order_specific_regimes_and_rank_arithmetic_guards():
    args=(jnp.zeros(2),jnp.array([[-1.,-1.],[1.,1.]]),jnp.zeros(2),jnp.array([[0.,1.]]),jnp.array([10.]))
    assert bool(approximate(*args,order=0).valid) and bool(approximate(*args,order=1).valid)
    assert not bool(approximate(*args,order=2).valid)
    assert not bool(approximate(*args,time=1.5,order=2).valid)
    assert not bool(approximate(*args,time=.5,order=0).valid)
    rank=approximate(*args[:3],jnp.array([[1.,2.],[2.,4.]]),jnp.array([.3,.6]))
    assert not bool(rank.valid) and np.isnan(rank.value).all()
    for centers,time in (([[0.,-1e200],[0.,1e200]],1.),([[0.,0.],[1.,1.]],3e307)):
        result=approximate(jnp.zeros(2),jnp.array(centers),jnp.zeros(2),jnp.array([[0.,1.]]),jnp.ones(1),time=time)
        assert not bool(result.valid) and np.isnan(result.approximation_error_bound)


@pytest.mark.parametrize('order',[-1,True,1.5,'2'])
def test_invalid_order_raises(order):
    with pytest.raises((TypeError,ValueError),match='order'):
        approximate(*_args(),order=order)


def test_no_inverse_factorization_and_no_high_dimensional_projector():
    dimension=257
    target=jnp.full(dimension,.01)
    centers=jnp.zeros((3,dimension)).at[1,0].set(.2).at[2,1].set(-.3)
    C=jnp.zeros((1,dimension)).at[0,2].set(1)
    def function(v):
        return approximate(v,centers,jnp.array([.1,-.2,.3]),C,jnp.array([.2])).value
    counts=[]
    def inspect(jaxpr):
        if hasattr(jaxpr,'jaxpr'):
            inspect(jaxpr.jaxpr)
        elif hasattr(jaxpr,'eqns'):
            for equation in jaxpr.eqns:
                assert equation.primitive.name not in ('lu','cholesky','custom_linear_solve')
                if equation.primitive.name=='qr':
                    counts.append(1)
                for variable in equation.outvars:
                    shape=getattr(variable.aval,'shape',())
                    assert len(shape)<2 or shape[-2:]!=(dimension,dimension)
                for value in equation.params.values():
                    if isinstance(value,(tuple,list)):
                        for item in value:
                            inspect(item)
                    else:
                        inspect(value)
    inspect(jax.make_jaxpr(function)(target))
    assert counts==[1]
    assert np.isfinite(jax.jit(jax.grad(lambda v:jnp.sum(function(v))))(target)).all()
