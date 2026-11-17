# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Constrained roots, source KKT conditions, seven-input AD and guards."""

from typing import Any

import numpy as np
import pytest
import torch
from omnibias.torch.gaussian_constrained import gaussian_heat_inverse_constrained
from omnibias.torch.gaussian_heat import gaussian_heat_inverse
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten


@pytest.fixture(scope="module", autouse=True)
def _one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _parameters():
    return tuple(torch.tensor(v, dtype=torch.float64, requires_grad=True) for v in (
        [0.2, -0.15], [[0.1, -0.2], [0.8, 0.3], [-0.2, 0.6]], [0.2, -0.1, 0.3], [[1.0, 2.0]], [0.3], 0.8, 1.3))


def _reference(args):
    mp = pytest.importorskip("mpmath")
    arrays = [v.detach().numpy() for v in args]
    with mp.workdps(65):
        target = [mp.mpf(float(v)) for v in arrays[0]]
        centers = [[mp.mpf(float(v)) for v in row] for row in arrays[1]]
        biases = [mp.mpf(float(v)) for v in arrays[2]]
        c = [mp.mpf(float(v)) for v in arrays[3][0]]
        rhs, nu, age = [mp.mpf(float(v)) for v in (arrays[4][0], arrays[5], arrays[6])]
        norm2 = sum(v*v for v in c)
        origin = [rhs*v/norm2 for v in c]
        tangent = [-c[1]/mp.sqrt(norm2), c[0]/mp.sqrt(norm2)]
        def residual(z):
            point = [a+z*b for a,b in zip(origin,tangent,strict=True)]
            weights = [mp.exp(b-sum((x-a)**2 for x,a in zip(point,row,strict=True))/(4*nu*age)) for row,b in zip(centers,biases,strict=True)]
            mean = [sum(w*row[k] for w,row in zip(weights,centers,strict=True))/sum(weights) for k in range(2)]
            return sum(t*(x-m-age*v) for t,x,m,v in zip(tangent,point,mean,target,strict=True))
        z = mp.findroot(residual, 0)
        return np.array([float(a+z*b) for a,b in zip(origin,tangent,strict=True)])


@pytest.mark.parametrize("adjoint", ["auto", "dense", "low_rank"])
def test_root_and_source_kkt_match_independent_high_precision_oracle(adjoint):
    args = _parameters()
    result = gaussian_heat_inverse_constrained(*args, adjoint=adjoint)
    assert result.converged and result.constraints_valid
    np.testing.assert_allclose(result.value.detach(), _reference(args), rtol=3e-13, atol=3e-13)
    target, centers, biases, matrix, rhs, nu, age = args
    logits = biases - (result.value - centers).square().sum(-1)/(4*nu*age)
    mean = torch.softmax(logits, dim=0) @ centers
    gradient = (result.value-mean)/age-target
    tangent = torch.stack((-matrix[0,1], matrix[0,0]))
    torch.testing.assert_close(gradient @ tangent, target.new_tensor(0), rtol=0, atol=2e-14)
    torch.testing.assert_close(matrix @ result.value, rhs, rtol=0, atol=2e-14)


def test_projected_contraction_and_bias_shift_age_rhs_derivatives():
    target = torch.zeros(2, dtype=torch.float64)
    centers = [[-1.0, -2.0], [1.0, 2.0]]
    biases = [2.25, 0.25]
    age = torch.ones((), dtype=target.dtype, requires_grad=True)
    rhs = torch.ones(1, dtype=target.dtype, requires_grad=True)
    result = gaussian_heat_inverse_constrained(target, centers, biases, [[0.0, 1.0]], rhs, time=age)
    assert result.converged and result.contraction_bound == 0.5
    unconstrained = gaussian_heat_inverse(target, centers, biases, time=age)
    assert not unconstrained.converged and unconstrained.contraction_bound == 2.5
    torch.testing.assert_close(result.value, target.new_tensor([0, 1]), rtol=0, atol=1e-14)
    ds, dd = torch.autograd.functional.jacobian(lambda s,d: gaussian_heat_inverse_constrained(
        target, centers, biases, [[0.0, 1.0]], d, time=s).value, (age,rhs))
    torch.testing.assert_close(ds, target.new_tensor([-2,0]), rtol=2e-13, atol=2e-13)
    torch.testing.assert_close(dd, target.new_tensor([[2],[1]]), rtol=2e-13, atol=2e-13)


@pytest.mark.parametrize("adjoint", ["dense", "low_rank"])
def test_seven_parameter_gradcheck_gradgradcheck_and_detached_diagnostics(adjoint):
    args = _parameters()
    def function(*values):
        return gaussian_heat_inverse_constrained(*values, adjoint=adjoint).value
    assert torch.autograd.gradcheck(function, args, rtol=3e-5, atol=3e-6)
    assert torch.autograd.gradgradcheck(function, args, rtol=4e-5, atol=4e-6)
    result = gaussian_heat_inverse_constrained(*args)
    assert all(not value.requires_grad for value in result[1:])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_empty_constraints_full_constraints_batches_and_dtypes(dtype):
    target = torch.full((2,3,2), 0.25, dtype=dtype, requires_grad=True)
    centers = torch.tensor([[0.1,-0.2],[0.2,0.1]], dtype=dtype, requires_grad=True)
    biases = torch.tensor([0.1,-0.1], dtype=dtype, requires_grad=True)
    nu = torch.ones((), dtype=dtype, requires_grad=True)
    age = torch.ones((), dtype=dtype, requires_grad=True)
    plain = gaussian_heat_inverse(target, centers, biases, nu, age)
    empty = gaussian_heat_inverse_constrained(target, centers, biases, torch.empty((0,2)), [], nu, age)
    torch.testing.assert_close(empty.value, plain.value)
    assert empty.feasibility_residual.dtype == dtype and empty.constraints_valid and empty.constraint_reciprocal_condition == 1
    matrix = torch.tensor([[1.,2.],[0.,1.]], dtype=dtype, requires_grad=True)
    rhs = torch.tensor([3.,1.], dtype=dtype, requires_grad=True)
    full = gaussian_heat_inverse_constrained(target, centers, biases, matrix, rhs, nu, age, max_iterations=0)
    torch.testing.assert_close(full.value, torch.ones_like(target))
    assert full.converged.all() and full.contraction_bound == 0 and not full.residual_bound.any()
    gradients = torch.autograd.grad(full.value.sum(), (target,centers,biases,matrix,rhs,nu,age))
    assert all(torch.isfinite(g).all() for g in gradients)
    assert all(not gradients[i].any() for i in (0,1,2,5,6))
    batch = gaussian_heat_inverse_constrained(target[:0], centers, biases, [[0.,1.]], [1.])
    assert batch.value.shape == (0,3,2) and batch.converged.shape == (0,3)


def test_constraint_row_basis_scaling_and_physical_gauges():
    args = tuple(v.detach() for v in _parameters())
    base = gaussian_heat_inverse_constrained(*args)
    for scale in (-1e200, 1e-200):
        changed = gaussian_heat_inverse_constrained(*args[:3], args[3]*scale, args[4]*scale, *args[5:])
        torch.testing.assert_close(changed.value, base.value, rtol=2e-13, atol=2e-13)
    shift = args[0].new_tensor([8.,-4.])
    translated = gaussian_heat_inverse_constrained(args[0], args[1]+shift, args[2]+16, args[3], args[4]+args[3]@shift, *args[5:])
    torch.testing.assert_close(translated.value, base.value+shift, rtol=2e-13, atol=2e-13)
    zero_bias = gaussian_heat_inverse_constrained(args[0], args[1], [0.]*3, *args[3:])
    huge_bias = gaussian_heat_inverse_constrained(args[0], args[1], [1e100]*3, *args[3:])
    torch.testing.assert_close(huge_bias.value, zero_bias.value, rtol=0, atol=0)


def test_multiple_constraint_rows_basis_invariance_and_higher_derivatives():
    target=torch.tensor([.2,-.1,.15],dtype=torch.float64)
    centers=torch.tensor([[0.,.2,.1],[.3,-.1,.2],[-.2,.1,-.1]],dtype=target.dtype)
    matrix=torch.tensor([[1.,2.,0.],[0.,1.,-1.]],dtype=target.dtype,requires_grad=True)
    rhs=torch.tensor([.3,-.2],dtype=target.dtype,requires_grad=True)
    basis=torch.tensor([[2.,1.],[-1.,3.]],dtype=target.dtype)
    def solve(C,d):
        return gaussian_heat_inverse_constrained(target,centers,[.2,-.1,.3],C,d).value
    expected=solve(matrix,rhs)
    torch.testing.assert_close(solve(basis@matrix,basis@rhs),expected,rtol=2e-13,atol=2e-13)
    torch.testing.assert_close(matrix@expected,rhs,rtol=2e-13,atol=2e-13)
    assert torch.autograd.gradcheck(solve,(matrix,rhs),atol=3e-6,rtol=3e-5)
    assert torch.autograd.gradgradcheck(solve,(matrix,rhs),atol=4e-6,rtol=4e-5)


@pytest.mark.parametrize("matrix,rhs", [([[1.,2.],[2.,4.]],[1.,2.]), ([[0.,0.]],[0.]),
                                        ([[1.,0.],[1.,1e-12]],[0.,0.]), ([[float('nan'),1.]],[0.]),
                                        ([[1.,0.]],[float('inf')]), ([[1e-308,0.]],[1e308])])
def test_rank_condition_and_invalid_constraint_refusals(matrix,rhs):
    result = gaussian_heat_inverse_constrained(torch.zeros(2,dtype=torch.float64), [[0.,0.]], [0.], matrix, rhs)
    assert not result.constraints_valid and not result.converged and torch.isnan(result.value).all()
    assert torch.isinf(result.feasibility_residual) and torch.isinf(result.residual_bound)


def test_query_model_failure_exhaustion_and_feasibility_tolerance():
    args = list(_parameters())
    args[0] = torch.stack((args[0], args[0].new_tensor([float('nan'),0.])))
    mixed = gaussian_heat_inverse_constrained(*args)
    assert mixed.converged.tolist() == [True,False] and torch.isnan(mixed.value[1]).all()
    early = gaussian_heat_inverse_constrained(*_parameters(), max_iterations=0)
    assert not early.converged and torch.isfinite(early.residual_bound) and early.residual_bound > 1e-5
    refused = gaussian_heat_inverse_constrained(*_parameters(), feasibility_tolerance=1e-30)
    assert refused.feasibility_residual > 1e-30 and not refused.converged
    args = list(_parameters())
    args[5] = 0.0
    invalid = gaussian_heat_inverse_constrained(*args)
    assert not invalid.converged and invalid.constraints_valid and torch.isinf(invalid.contraction_bound)


def test_high_dimension_projection_and_backward_never_allocate_dense_projector():
    dimension = 257
    target = torch.full((2,dimension), 0.01, dtype=torch.float64, requires_grad=True)
    centers = torch.zeros((3,dimension), dtype=target.dtype)
    centers[1,0], centers[2,1] = 0.2,-0.3
    matrix = torch.zeros((2,dimension), dtype=target.dtype, requires_grad=True)
    with torch.no_grad():
        matrix[0,2], matrix[1,3] = 1,1
    class NoSquare(TorchDispatchMode):
        def __torch_dispatch__(self,function:Any,types:Any,args:Any=(),kwargs:Any=None)->Any:
            result=function(*args,**({} if kwargs is None else kwargs))
            leaves,_=tree_flatten(result)
            assert all(not isinstance(v,torch.Tensor) or v.ndim<2 or v.shape[-2:]!=(dimension,dimension) for v in leaves)
            return result
    with NoSquare():
        result=gaussian_heat_inverse_constrained(target,centers,[0.,.1,-.1],matrix,[.2,-.1],adjoint='low_rank')
        gradients=torch.autograd.grad(result.value.square().sum(),(target,matrix))
    assert result.converged.all() and all(torch.isfinite(g).all() for g in gradients)


@pytest.mark.parametrize("options",[{"constraint_rcond":0},{"constraint_rcond":1},{"constraint_rcond":True},
                                   {"feasibility_tolerance":0},{"max_iterations":True},{"tolerance":float('inf')},{"adjoint":"bad"}])
def test_invalid_controls_raise(options):
    with pytest.raises((TypeError,ValueError)):
        gaussian_heat_inverse_constrained(*_parameters(),**options)


def test_invalid_shapes_and_types_raise():
    args=list(_parameters())
    for matrix,rhs in (([[1.]],[0.]), ([[1.,0.],[0.,1.],[1.,1.]],[0.,0.,0.]), ([[1.,0.]],[])):
        with pytest.raises(ValueError):
            gaussian_heat_inverse_constrained(*args[:3],matrix,rhs,*args[5:])
    for matrix in ([[True,False]],[[1j,0j]]):
        with pytest.raises(TypeError):
            gaussian_heat_inverse_constrained(*args[:3],matrix,[0.],*args[5:])


@pytest.mark.skipif(not torch.cuda.is_available(),reason="CUDA unavailable")
def test_cuda_dtype_device_and_constraint_gradients():
    args=tuple(v.detach().cuda().requires_grad_() for v in _parameters())
    result=gaussian_heat_inverse_constrained(*args)
    assert result.converged and all(v.device==args[0].device for v in result)
    assert all(torch.isfinite(v).all() for v in torch.autograd.grad(result.value.sum(),args))
