# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Fixed-age resummation, auxiliary coefficients and full runtime gradients."""

from fractions import Fraction

import numpy as np
import pytest
import torch
from omnibias.torch.gaussian_aging import gaussian_heat_inverse_aging
from omnibias.torch.gaussian_constrained_aging import gaussian_heat_inverse_constrained_aging
from omnibias.torch.gaussian_constrained_approximation import (
    gaussian_heat_inverse_constrained_approximation as approximate,
)
from omnibias.torch.gaussian_predictor import gaussian_heat_inverse_predictor


@pytest.fixture(scope="module",autouse=True)
def _one_thread():
    previous=torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _args():
    return tuple(torch.tensor(v,dtype=torch.float64,requires_grad=True) for v in (
        [.2,-.15],[[.1,-.2],[.8,.3],[-.2,.6]],[.2,-.1,.3],[[1.,2.]],[.3],.8,8.))


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_auxiliary_coefficients_match_independent_rational_fixture(dtype):
    # Normal costs (1,0,1) are exactly absorbed by original biases at s=8.
    # Frozen projected biases are then zero; exact whole-series Picard
    # substitution for tangent centers (0,1,3) supplies this independent oracle.
    exact=[Fraction(0),Fraction(-5,216),Fraction(-245,124416),Fraction(-4655,35831808),
           Fraction(-257555,41278242816),Fraction(-237013,3962711310336),
           Fraction(76827149,2282521714753536),Fraction(24734280845,4601563776943128576),
           Fraction(2483951492605,4543715546604414959616)]
    result=approximate(torch.zeros(2,dtype=dtype),[[0.,0.],[1.,1.],[3.,2.]],[1/32,0.,1/32],[[0.,1.]],[1.],time=8)
    tolerance=2e-4 if dtype==torch.float32 else 2e-12
    np.testing.assert_allclose(result.correction_coefficients[:,0],[float(v) for v in exact],rtol=tolerance,atol=0)
    assert not result.correction_coefficients[:,1].any() and result.valid
    torch.testing.assert_close(result.value,result.value.new_tensor([float(Fraction(4,3)+sum(exact)),1.]))
    assert result.contraction_bound==9/64 and result.value.dtype==result.approximation_error_bound.dtype==dtype


@pytest.mark.parametrize('order',[0,1,4,8])
def test_large_original_Q_valid_q_and_original_root_error(order):
    mp=pytest.importorskip('mpmath')
    args=(torch.zeros(2,dtype=torch.float64),[[-1.,-1.],[1.,1.]],[0.,0.],[[0.,1.]],[10.])
    old=gaussian_heat_inverse_constrained_aging(*args,time=4)
    assert not old.valid and old.aging_parameter==11/8
    result=approximate(*args,time=4,order=order)
    assert result.valid and result.contraction_bound==1/8 and result.value[1]==10
    q=Fraction(1,8)
    expected=q if order==0 else (2*q*q if order==1 else (3*q)**(order+1)/(1-3*q))
    assert result.approximation_error_bound.item()==pytest.approx(float(expected),rel=2e-14)
    with mp.workdps(70):
        root=mp.findroot(lambda x:x-mp.tanh((x+10)/8),.9)
        assert abs(root-mp.mpf(float(result.value[0])))<mp.mpf(float(result.approximation_error_bound))


@pytest.mark.parametrize('order',[0,1,4])
def test_seven_input_gradcheck_and_gradgrad_of_finite_formula(order):
    args=_args()
    def evaluate(*p):
        result=approximate(*p,order=order)
        return torch.cat((result.value,result.correction_coefficients.flatten()))
    assert torch.autograd.gradcheck(evaluate,args,atol=3e-6,rtol=3e-5)
    assert torch.autograd.gradgradcheck(evaluate,args,atol=4e-6,rtol=4e-5)
    assert all(not v.requires_grad for v in approximate(*args,order=order)[2:])


@pytest.mark.parametrize('order,expected',[(0,-1.),(1,-1.5)])
def test_runtime_age_derivative_includes_bias_shift_and_differs_from_frozen_family(order,expected):
    target=torch.zeros(2,dtype=torch.float64)
    age=torch.ones((),dtype=target.dtype,requires_grad=True)
    result=approximate(target,[[-1.,-2.],[1.,2.]],[2.25,.25],[[0.,1.]],[1.],time=age,order=order)
    live=torch.autograd.grad(result.value[0],age,create_graph=True)[0]
    torch.testing.assert_close(live,age.new_tensor(expected),rtol=2e-13,atol=2e-13)
    # At s=1 the transformed biases coincide. Freezing them makes every
    # auxiliary coefficient vanish by symmetry, so this different slope is 0.
    frozen=gaussian_heat_inverse_predictor(target,[[-1.,1.],[1.,1.]],[0.,0.],time=age,order=order)
    torch.testing.assert_close(torch.autograd.grad(frozen.value[0],age)[0],age.new_tensor(0.),rtol=0,atol=0)
    assert torch.isfinite(torch.autograd.grad(live,age)[0])


def test_order_specific_q_cutoffs_and_affine_branches():
    args=(torch.zeros(2,dtype=torch.float64),[[-1.,-1.],[1.,1.]],[0.,0.],[[0.,1.]],[10.])
    assert approximate(*args,order=0).valid and approximate(*args,order=1).valid
    assert not approximate(*args,order=2).valid
    assert not approximate(*args,time=1.5,order=2).valid
    assert not approximate(*args,time=.5,order=0).valid
    affine=approximate(args[0],[[0.,-10.],[0.,10.]],[.1,-.2],args[3],args[4])
    assert affine.valid and affine.contraction_bound==0 and affine.approximation_error_bound==0
    assert not affine.correction_coefficients.any()
    torch.testing.assert_close(affine.value,args[0].new_tensor([0.,10.]))
    full=approximate(args[0],args[1],args[2],[[1.,2.],[0.,1.]],[3.,1.])
    assert full.valid and not full.correction_coefficients.any()
    torch.testing.assert_close(full.value,torch.ones_like(args[0]))


def test_partial_affine_case_retains_transverse_center_gradients():
    target=torch.zeros(2,dtype=torch.float64)
    centers=torch.tensor([[0.,-.2],[0.,.3]],dtype=target.dtype,requires_grad=True)
    def value(a):
        return approximate(target,a,[0.,0.],[[0.,1.]],[.1],time=8,order=4).value
    p=torch.softmax(-((centers.detach()[:,1]-.1)**2)/32,dim=0)
    gradient=torch.autograd.grad(value(centers)[0],centers)[0]
    torch.testing.assert_close(gradient[:,0],p,rtol=2e-13,atol=2e-13)
    assert torch.autograd.gradcheck(value,(centers,),atol=2e-7,rtol=2e-5)
    assert torch.autograd.gradgradcheck(value,(centers,),atol=3e-7,rtol=3e-5)


def test_empty_constraints_batches_gauges_and_empty_batch():
    args=tuple(v.detach() for v in _args())
    target=args[0].repeat(2,3,1)
    for order in (0,1,4):
        result=approximate(target,*args[1:3],torch.empty((0,2)),[],*args[5:],order=order)
        reference=(gaussian_heat_inverse_predictor if order<2 else gaussian_heat_inverse_aging)(target,*args[1:3],*args[5:],order=order)
        torch.testing.assert_close(result.value,reference.value)
    none=approximate(target[:0],*args[1:])
    assert none.value.shape==(0,3,2) and none.correction_coefficients.shape==(9,0,3,2)
    first=approximate(args[0],args[1],[0.]*3,*args[3:])
    second=approximate(args[0],args[1],[1e100]*3,*args[3:])
    torch.testing.assert_close(first.value,second.value,rtol=0,atol=0)
    rescaled=approximate(*args[:3],args[3]*-1e200,args[4]*-1e200,*args[5:])
    torch.testing.assert_close(rescaled.value,approximate(*args).value,rtol=2e-13,atol=2e-13)


@pytest.mark.parametrize('order',[-1,True,1.5,'2'])
def test_invalid_order_raises(order):
    with pytest.raises((TypeError,ValueError),match='order'):
        approximate(*_args(),order=order)


def test_rank_bad_inputs_and_arithmetic_refuse():
    args=list(_args())
    args[3],args[4]=[[1.,2.],[2.,4.]],[.3,.6]
    result=approximate(*args)
    assert not result.valid and torch.isnan(result.value).all() and torch.isnan(result.correction_coefficients).all()
    for centers,time in (([[0.,-1e200],[0.,1e200]],1.),([[0.,0.],[1.,1.]],3e307)):
        result=approximate(torch.zeros(2,dtype=torch.float64),centers,[0.,0.],[[0.,1.]],[1.],time=time)
        assert not result.valid and torch.isnan(result.approximation_error_bound)


def test_no_inverse_factorization_after_single_QR(monkeypatch):
    original=torch.linalg.qr
    calls=[]
    def counted(*args,**kwargs):
        calls.append(1)
        return original(*args,**kwargs)
    def forbidden(*args,**kwargs):
        pytest.fail('fixed-age forward approximation must not factor an inverse Jacobian')
    monkeypatch.setattr(torch.linalg,'qr',counted)
    for name in ('solve','solve_ex','cholesky','cholesky_ex'):
        monkeypatch.setattr(torch.linalg,name,forbidden)
    result=approximate(*_args())
    assert result.valid and calls==[1]


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA unavailable')
def test_cuda_gradients_and_device():
    args=tuple(v.detach().cuda().requires_grad_() for v in _args())
    result=approximate(*args)
    assert result.valid and all(v.device==args[0].device for v in result)
    assert all(torch.isfinite(v).all() for v in torch.autograd.grad(result.value.sum(),args))
