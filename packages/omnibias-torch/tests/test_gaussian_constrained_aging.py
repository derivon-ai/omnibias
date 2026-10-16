# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Constrained original-family forward approximations and finite-formula AD."""

from fractions import Fraction
from typing import Any

import numpy as np
import pytest
import torch
from omnibias.torch.gaussian_aging import gaussian_heat_inverse_aging
from omnibias.torch.gaussian_constrained_aging import (
    gaussian_heat_inverse_constrained_aging as aging,
)
from omnibias.torch.gaussian_constrained_aging import (
    gaussian_heat_inverse_constrained_predictor as predictor,
)
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten


@pytest.fixture(scope="module", autouse=True)
def _one_thread():
    previous=torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _args():
    return tuple(torch.tensor(v,dtype=torch.float64,requires_grad=True) for v in (
        [.2,-.15],[[.1,-.2],[.8,.3],[-.2,.6]],[.2,-.1,.3],[[1.,2.]],[.3],.8,8.))


@pytest.mark.parametrize("dtype",[torch.float32,torch.float64])
def test_order_eight_matches_independent_exact_lagrange_coefficients(dtype):
    # h=tanh(t*(h+2)/32). Lagrange: c_n=[u^(n-1)]
    # ((u+2)/(32*(atanh(u)/u)))**n/n, independent of our recurrence.
    exact=[Fraction(0),Fraction(1,16),Fraction(1,512),Fraction(-1,49152),Fraction(-13,1572864),
           Fraction(-51,83886080),Fraction(-113,8053063680),Fraction(10399,5411658792960),Fraction(9119,34634616274944)]
    result=aging(torch.zeros(2,dtype=dtype),[[-1.,-2.],[1.,2.]],[0.,0.],[[0.,1.]],[1.],time=16)
    tol=2e-5 if dtype==torch.float32 else 3e-12
    np.testing.assert_allclose(result.correction_coefficients[:,0],[float(c) for c in exact],rtol=tol,atol=0)
    assert not result.correction_coefficients[:,1].any() and result.valid
    torch.testing.assert_close(result.value,result.value.new_tensor([float(sum(exact)),1]))
    assert result.contraction_bound==1/32 and result.aging_parameter==3/32
    assert result.approximation_error_bound.item()==pytest.approx(float((Fraction(9,32))**9/(1-Fraction(9,32))),rel=tol)


@pytest.mark.parametrize("order",[0,1])
def test_original_bias_and_normal_cost_moment_formula(order):
    args=tuple(v.detach() for v in _args())
    v,a,b,C,d,nu,s=(v.numpy() for v in args)
    origin=C.T@np.linalg.solve(C@C.T,d)
    P=np.eye(2)-C.T@np.linalg.solve(C@C.T,C)
    projected=a@P.T+origin
    normal=a-projected
    cost=np.sum(normal**2,axis=-1)
    weights=np.exp(b+projected@v/(2*nu))
    p=weights/weights.sum()
    mean=p@projected
    A=projected-mean
    correction=-np.sum((p*(np.sum(A*A,axis=-1)+cost-cost[0]))[:,None]*A,axis=0)/(4*nu*s)
    result=predictor(*args,order=order)
    np.testing.assert_allclose(result.value,s*(P@v)+mean+(correction if order else 0),rtol=2e-13,atol=2e-13)
    if order:
        np.testing.assert_allclose(result.correction_coefficients[1],correction,rtol=3e-12,atol=3e-15)
    diameter=max(np.linalg.norm(x-y) for x in projected for y in projected)
    q=diameter**2/(8*nu*s)
    Q=(diameter**2+np.ptp(cost))/(8*nu*s)
    expected=diameter*Q/2 if not order else diameter*Q*(Q+q)/2
    assert result.approximation_error_bound.item()==pytest.approx(expected,rel=2e-13)


@pytest.mark.parametrize("function,order",[(predictor,0),(predictor,1),(aging,4)])
def test_all_seven_inputs_and_coefficients_support_gradcheck_and_gradgrad(function,order):
    args=_args()
    def evaluate(*values):
        result=function(*values,order=order)
        return torch.cat((result.value,result.correction_coefficients.flatten()))
    assert torch.autograd.gradcheck(evaluate,args,atol=3e-6,rtol=3e-5)
    assert torch.autograd.gradgradcheck(evaluate,args,atol=4e-6,rtol=4e-5)
    assert all(not v.requires_grad for v in function(*args,order=order)[2:])


@pytest.mark.parametrize("function,order",[(predictor,0),(predictor,1),(aging,4),(aging,8)])
def test_analytical_tail_contains_independent_original_scalar_root(function,order):
    mp=pytest.importorskip('mpmath')
    result=function(torch.zeros(2,dtype=torch.float64),[[-1.,-2.],[1.,2.]],[2.25,.25],[[0.,1.]],[1.],time=16,order=order)
    with mp.workdps(70):
        root=mp.findroot(lambda x:x-mp.tanh(-1+(x+2)/32),-.75)
        assert abs(mp.mpf(float(result.value[0]))-root)<mp.mpf(float(result.approximation_error_bound))
    assert result.value[1]==1 and result.valid


def test_aging_Q_is_distinct_from_tangent_contraction_and_affine_bypasses_Q():
    args=(torch.zeros(2,dtype=torch.float64),[[-1.,-2.],[1.,2.]],[0.,0.],[[0.,1.]],[1.])
    first=predictor(*args)
    refused=aging(*args)
    assert first.valid and first.contraction_bound==.5 and first.aging_parameter==1.5
    assert not refused.valid and torch.isnan(refused.value).all()
    assert torch.isnan(refused.correction_coefficients).all() and torch.isnan(refused.approximation_error_bound)
    affine=aging(args[0],[[0.,-10.],[0.,10.]],[.1,-.2],args[3],args[4])
    assert affine.valid and affine.contraction_bound==0 and affine.aging_parameter==5
    assert not affine.correction_coefficients.any() and affine.approximation_error_bound==0
    torch.testing.assert_close(affine.value,args[0].new_tensor([0.,1.]))


def test_affine_coefficients_retain_center_perturbation_derivatives():
    # Values vanish at coincident tangent centers, but their transverse
    # derivatives generally do not when normal costs differ.
    target=torch.zeros(2,dtype=torch.float64)
    centers=torch.tensor([[0.,-.2],[0.,.3]],dtype=target.dtype,requires_grad=True)
    def coefficients(a):
        return aging(target,a,[0.,0.],[[0.,1.]],[.1],time=8,order=3).correction_coefficients
    assert not coefficients(centers).any()
    gradient=torch.autograd.grad(coefficients(centers)[1,0],centers)[0]
    assert gradient[:,0].abs().max()>1e-5
    assert torch.autograd.gradcheck(coefficients,(centers,),atol=2e-7,rtol=2e-5)


def test_affine_bypass_does_not_accept_overflowed_cost_arithmetic():
    for C,rhs in (([[0.,1.]],[1.]),([[1.,0.],[0.,1.]],[0.,1.])):
        result=aging(torch.zeros(2,dtype=torch.float64),[[0.,-1e200],[0.,1e200]],[0.,0.],C,rhs)
        assert not result.valid and torch.isnan(result.value).all() and torch.isnan(result.approximation_error_bound)


def test_empty_full_constraints_batches_and_gauges():
    args=tuple(v.detach() for v in _args())
    target=args[0].repeat(2,3,1)
    empty=aging(target,*args[1:3],torch.empty((0,2)),[],*args[5:])
    plain=gaussian_heat_inverse_aging(target,*args[1:3],*args[5:])
    torch.testing.assert_close(empty.value,plain.value)
    torch.testing.assert_close(empty.correction_coefficients,plain.correction_coefficients)
    full=aging(target,*args[1:3],[[1.,2.],[0.,1.]],[3.,1.],*args[5:])
    assert full.valid.all() and not full.correction_coefficients.any()
    torch.testing.assert_close(full.value,torch.ones_like(target))
    noqueries=aging(target[:0],*args[1:])
    assert noqueries.correction_coefficients.shape==(9,0,3,2)
    base=aging(*args)
    shifted=aging(*args[:3],args[3]*-1e200,args[4]*-1e200,*args[5:])
    torch.testing.assert_close(shifted.value,base.value,rtol=2e-13,atol=2e-13)
    a=aging(args[0],args[1],[0.]*3,*args[3:])
    b=aging(args[0],args[1],[1e100]*3,*args[3:])
    torch.testing.assert_close(a.value,b.value,rtol=0,atol=0)


@pytest.mark.parametrize("bad",["rank","query","age","rcond"])
def test_explicit_numeric_refusals(bad):
    args=list(_args())
    options={}
    if bad=='rank':
        args[3],args[4]=[[1.,2.],[2.,4.]],[.3,.6]
    elif bad=='query':
        args[0]=args[0].new_tensor([float('nan'),0.])
    elif bad=='age':
        args[6]=0.
    else:
        args[3],args[4]=[[1.,0.],[1.,1e-12]],[0.,0.]
    result=aging(*args,**options)
    assert not result.valid and torch.isnan(result.value).all() and torch.isnan(result.correction_coefficients).all()


@pytest.mark.parametrize("function,order",[(aging,-1),(aging,True),(predictor,2),(predictor,.5)])
def test_invalid_orders_raise(function,order):
    with pytest.raises((TypeError,ValueError)):
        function(*_args(),order=order)


def test_high_dimension_has_one_QR_and_no_inverse_factorization(monkeypatch):
    dimension=257
    target=torch.full((2,dimension),.01,dtype=torch.float64,requires_grad=True)
    centers=torch.zeros((3,dimension),dtype=target.dtype)
    centers[1,0],centers[2,1]=.2,-.3
    C=torch.zeros((1,dimension),dtype=target.dtype)
    C[0,2]=1
    calls=[]
    original=torch.linalg.qr
    def counted(*args,**kwargs):
        calls.append(1)
        return original(*args,**kwargs)
    def forbidden(*args,**kwargs):
        pytest.fail('forward recurrence must not solve an inverse root or factor its Jacobian')
    monkeypatch.setattr(torch.linalg,'qr',counted)
    for name in ('solve','solve_ex','cholesky','cholesky_ex'):
        monkeypatch.setattr(torch.linalg,name,forbidden)
    class NoSquare(TorchDispatchMode):
        def __torch_dispatch__(self,function:Any,types:Any,args:Any=(),kwargs:Any=None)->Any:
            result=function(*args,**({} if kwargs is None else kwargs))
            leaves,_=tree_flatten(result)
            assert all(not isinstance(v,torch.Tensor) or v.ndim<2 or v.shape[-2:]!=(dimension,dimension) for v in leaves)
            return result
    with NoSquare():
        result=aging(target,centers,[.1,-.2,.3],C,[.2])
        gradient=torch.autograd.grad(result.value.sum(),target)[0]
    assert calls==[1] and result.valid.all() and torch.isfinite(gradient).all()


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable')
def test_cuda_preserves_device_and_gradient():
    args=tuple(v.detach().cuda().requires_grad_() for v in _args())
    result=aging(*args)
    assert result.valid and all(v.device==args[0].device for v in result)
    assert all(torch.isfinite(v).all() for v in torch.autograd.grad(result.value.sum(),args))
