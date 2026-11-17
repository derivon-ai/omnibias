# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Constrained forward formula oracles, tracing and all-parameter derivatives."""

from fractions import Fraction

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from omnibias.jax.gaussian_aging import gaussian_heat_inverse_aging
from omnibias.jax.gaussian_constrained_aging import GaussianConstrainedAgingResult
from omnibias.jax.gaussian_constrained_aging import gaussian_heat_inverse_constrained_aging as aging
from omnibias.jax.gaussian_constrained_aging import (
    gaussian_heat_inverse_constrained_predictor as predictor,
)

jax.config.update('jax_enable_x64',True)


def _args():
    return tuple(jnp.asarray(v,dtype=jnp.float64) for v in (
        [.2,-.15],[[.1,-.2],[.8,.3],[-.2,.6]],[.2,-.1,.3],[[1.,2.]],[.3],.8,8.))


@pytest.mark.parametrize('dtype',[jnp.float32,jnp.float64])
def test_jit_order_eight_matches_independent_lagrange_coefficients(dtype):
    # h=tanh(t*(h+2)/32): Lagrange coefficient extraction avoids our
    # Gaussian exponential and quotient recurrences.
    exact=[Fraction(0),Fraction(1,16),Fraction(1,512),Fraction(-1,49152),Fraction(-13,1572864),
           Fraction(-51,83886080),Fraction(-113,8053063680),Fraction(10399,5411658792960),Fraction(9119,34634616274944)]
    result=jax.jit(aging)(jnp.zeros(2,dtype=dtype),jnp.array([[-1.,-2.],[1.,2.]]),jnp.zeros(2),jnp.array([[0.,1.]]),jnp.ones(1),time=16)
    tol=2e-5 if dtype==jnp.float32 else 3e-12
    np.testing.assert_allclose(result.correction_coefficients[:,0],[float(c) for c in exact],rtol=tol,atol=0)
    np.testing.assert_array_equal(result.correction_coefficients[:,1],jnp.zeros(9))
    np.testing.assert_allclose(result.value,[float(sum(exact)),1],rtol=tol,atol=tol)
    assert bool(result.valid) and result.contraction_bound==1/32 and result.aging_parameter==3/32
    assert result.value.dtype==result.correction_coefficients.dtype==result.approximation_error_bound.dtype==dtype


@pytest.mark.parametrize('order',[0,1])
def test_independent_original_bias_normal_cost_formula_and_bounds(order):
    args=_args()
    v,a,b,C,d,nu,s=(np.asarray(v) for v in args)
    origin=C.T@np.linalg.solve(C@C.T,d)
    P=np.eye(2)-C.T@np.linalg.solve(C@C.T,C)
    projected=a@P.T+origin
    cost=np.sum((a-projected)**2,axis=-1)
    weights=np.exp(b+projected@v/(2*nu))
    p=weights/weights.sum()
    mean=p@projected
    A=projected-mean
    correction=-np.sum((p*(np.sum(A*A,axis=-1)+cost-cost[0]))[:,None]*A,axis=0)/(4*nu*s)
    result=jax.jit(lambda *p:predictor(*p,order=order))(*args)
    np.testing.assert_allclose(result.value,s*(P@v)+mean+(correction if order else 0),rtol=2e-13,atol=2e-13)
    diameter=max(np.linalg.norm(x-y) for x in projected for y in projected)
    q=diameter**2/(8*nu*s)
    Q=(diameter**2+np.ptp(cost))/(8*nu*s)
    assert float(result.approximation_error_bound)==pytest.approx(diameter*Q/2 if not order else diameter*Q*(Q+q)/2,rel=2e-13)


@pytest.mark.parametrize('function,order',[(predictor,0),(predictor,1),(aging,4)])
def test_seven_input_gradients_and_higher_derivatives_of_finite_formula(function,order):
    args=_args()
    def loss(*p):
        result=function(*p,order=order)
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
    diagnostics=jax.grad(lambda centers:jnp.sum(function(args[0],centers,*args[2:],order=order).approximation_error_bound))(args[1])
    np.testing.assert_array_equal(diagnostics,jnp.zeros_like(args[1]))


def test_jit_vmap_live_constraints_batches_empty_and_full_rank():
    args=_args()
    target=jnp.stack((args[0],args[0]*2,args[0]/2))
    result=jax.jit(aging)(target,*args[1:])
    out_axes=GaussianConstrainedAgingResult(0,1,0,None,None,0,None,0)
    mapped=jax.vmap(lambda query:aging(query,*args[1:]),out_axes=out_axes)(target)
    np.testing.assert_allclose(mapped.value,result.value,rtol=2e-13,atol=2e-13)
    np.testing.assert_allclose(mapped.correction_coefficients,result.correction_coefficients,rtol=3e-12,atol=3e-14)
    constraints=jnp.stack((args[3],args[3]+jnp.array([[.2,-.1]])))
    mapped_constraints=jax.jit(jax.vmap(lambda C:aging(*args[:3],C,*args[4:])))(constraints)
    assert bool(mapped_constraints.valid.all())
    empty=aging(target,*args[1:3],jnp.empty((0,2)),jnp.empty(0),*args[5:])
    plain=gaussian_heat_inverse_aging(target,*args[1:3],*args[5:])
    np.testing.assert_allclose(empty.value,plain.value,rtol=2e-13,atol=2e-13)
    np.testing.assert_allclose(empty.correction_coefficients,plain.correction_coefficients,rtol=3e-12,atol=3e-14)
    full=aging(target,*args[1:3],jnp.array([[1.,2.],[0.,1.]]),jnp.array([3.,1.]),*args[5:])
    assert bool(full.valid.all()) and not bool(full.correction_coefficients.any())
    np.testing.assert_allclose(full.value,np.ones_like(target),rtol=1e-13,atol=1e-13)
    noqueries=aging(target[:0],*args[1:])
    assert noqueries.correction_coefficients.shape==(9,0,2)


def test_Q_cutoff_differs_from_q_and_affine_bypasses_Q():
    args=(jnp.zeros(2),jnp.array([[-1.,-2.],[1.,2.]]),jnp.zeros(2),jnp.array([[0.,1.]]),jnp.ones(1))
    result=predictor(*args)
    refused=aging(*args)
    assert bool(result.valid) and result.contraction_bound==.5 and result.aging_parameter==1.5
    assert not bool(refused.valid) and np.isnan(refused.value).all() and np.isnan(refused.correction_coefficients).all()
    affine=aging(args[0],jnp.array([[0.,-10.],[0.,10.]]),jnp.array([.1,-.2]),*args[3:])
    assert bool(affine.valid) and affine.aging_parameter==5 and affine.contraction_bound==0
    np.testing.assert_array_equal(affine.correction_coefficients,jnp.zeros((9,2)))
    np.testing.assert_array_equal(affine.value,[0.,1.])


def test_affine_transverse_center_derivatives_and_gauges():
    target=jnp.zeros(2)
    centers=jnp.array([[0.,-.2],[0.,.3]])
    def coefficient(a):
        return aging(target,a,jnp.zeros(2),jnp.array([[0.,1.]]),jnp.array([.1]),time=8,order=3).correction_coefficients[1,0]
    gradient=jax.grad(coefficient)(centers)
    assert jnp.max(jnp.abs(gradient[:,0]))>1e-5
    step=1e-5
    expected=(float(coefficient(centers.at[0,0].add(step)))-float(coefficient(centers.at[0,0].add(-step))))/(2*step)
    np.testing.assert_allclose(gradient[0,0],expected,rtol=3e-7,atol=3e-10)
    assert np.isfinite(jax.hessian(coefficient)(centers)).all()
    args=_args()
    base=aging(*args)
    scaled=aging(*args[:3],args[3]*-1e200,args[4]*-1e200,*args[5:])
    np.testing.assert_allclose(base.value,scaled.value,rtol=2e-13,atol=2e-13)
    first=aging(args[0],args[1],jnp.zeros(3),*args[3:])
    second=aging(args[0],args[1],jnp.full(3,1e100),*args[3:])
    np.testing.assert_array_equal(first.value,second.value)


def test_affine_bypass_does_not_accept_overflowed_cost_arithmetic():
    for C,rhs in (([[0.,1.]],[1.]),([[1.,0.],[0.,1.]],[0.,1.])):
        result=aging(jnp.zeros(2),jnp.array([[0.,-1e200],[0.,1e200]]),jnp.zeros(2),jnp.array(C),jnp.array(rhs))
        assert not bool(result.valid) and np.isnan(result.value).all() and np.isnan(result.approximation_error_bound)


@pytest.mark.parametrize('function,order',[(predictor,0),(predictor,1),(aging,4),(aging,8)])
def test_true_original_scalar_inverse_error_below_analytical_bound(function,order):
    mp=pytest.importorskip('mpmath')
    result=function(jnp.zeros(2),jnp.array([[-1.,-2.],[1.,2.]]),jnp.array([2.25,.25]),jnp.array([[0.,1.]]),jnp.ones(1),time=16,order=order)
    with mp.workdps(70):
        root=mp.findroot(lambda x:x-mp.tanh(-1+(x+2)/32),-.75)
        assert abs(mp.mpf(float(result.value[0]))-root)<mp.mpf(float(result.approximation_error_bound))


@pytest.mark.parametrize('bad',['rank','query','age','rcond'])
def test_invalid_numerics_refuse(bad):
    args=list(_args())
    if bad=='rank':
        args[3],args[4]=jnp.array([[1.,2.],[2.,4.]]),jnp.array([.3,.6])
    elif bad=='query':
        args[0]=jnp.array([jnp.nan,0.])
    elif bad=='age':
        args[6]=0.
    else:
        args[3],args[4]=jnp.array([[1.,0.],[1.,1e-12]]),jnp.zeros(2)
    result=aging(*args)
    assert not bool(result.valid) and np.isnan(result.value).all() and np.isnan(result.correction_coefficients).all()


@pytest.mark.parametrize('function,order',[(aging,-1),(aging,True),(predictor,2),(predictor,.5)])
def test_invalid_order_raises(function,order):
    with pytest.raises((TypeError,ValueError)):
        function(*_args(),order=order)


def test_high_dimension_no_dense_projector_or_inverse_linear_solve():
    dimension=257
    target=jnp.full(dimension,.01)
    centers=jnp.zeros((3,dimension)).at[1,0].set(.2).at[2,1].set(-.3)
    C=jnp.zeros((1,dimension)).at[0,2].set(1)
    def function(v):
        return aging(v,centers,jnp.array([.1,-.2,.3]),C,jnp.array([.2])).value
    graph=jax.make_jaxpr(function)(target)
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
    inspect(graph)
    assert counts==[1]
    assert np.isfinite(jax.jit(jax.grad(lambda v:jnp.sum(function(v))))(target)).all()
