# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Constrained inverse oracle, tracing, parameter derivatives and guards."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from omnibias.jax.gaussian_constrained import gaussian_heat_inverse_constrained
from omnibias.jax.gaussian_heat import gaussian_heat_inverse

jax.config.update("jax_enable_x64", True)


def _parameters():
    return tuple(jnp.asarray(v,dtype=jnp.float64) for v in (
        [0.2,-0.15],[[0.1,-0.2],[0.8,0.3],[-0.2,0.6]],[0.2,-0.1,0.3],[[1.,2.]],[0.3],0.8,1.3))


def _reference(args):
    mp=pytest.importorskip("mpmath")
    arrays=[np.asarray(v) for v in args]
    with mp.workdps(65):
        target=[mp.mpf(float(v)) for v in arrays[0]]
        centers=[[mp.mpf(float(v)) for v in row] for row in arrays[1]]
        biases=[mp.mpf(float(v)) for v in arrays[2]]
        c=[mp.mpf(float(v)) for v in arrays[3][0]]
        rhs,nu,age=[mp.mpf(float(v)) for v in (arrays[4][0],arrays[5],arrays[6])]
        norm2=sum(v*v for v in c)
        origin=[rhs*v/norm2 for v in c]
        tangent=[-c[1]/mp.sqrt(norm2),c[0]/mp.sqrt(norm2)]
        def residual(z):
            point=[a+z*b for a,b in zip(origin,tangent,strict=True)]
            weights=[mp.exp(b-sum((x-a)**2 for x,a in zip(point,row,strict=True))/(4*nu*age)) for row,b in zip(centers,biases,strict=True)]
            mean=[sum(w*row[k] for w,row in zip(weights,centers,strict=True))/sum(weights) for k in range(2)]
            return sum(t*(x-m-age*v) for t,x,m,v in zip(tangent,point,mean,target,strict=True))
        z=mp.findroot(residual,0)
        return np.array([float(a+z*b) for a,b in zip(origin,tangent,strict=True)])


@pytest.mark.parametrize("adjoint",["auto","dense","low_rank"])
def test_jit_root_source_kkt_and_independent_high_precision_oracle(adjoint):
    args=_parameters()
    result=jax.jit(lambda *p:gaussian_heat_inverse_constrained(*p,adjoint=adjoint))(*args)
    assert bool(result.converged) and bool(result.constraints_valid)
    np.testing.assert_allclose(result.value,_reference(args),rtol=3e-13,atol=3e-13)
    target,centers,biases,matrix,rhs,nu,age=args
    weights=jax.nn.softmax(biases-jnp.sum((result.value-centers)**2,axis=-1)/(4*nu*age))
    gradient=(result.value-weights@centers)/age-target
    np.testing.assert_allclose(gradient@jnp.array([-matrix[0,1],matrix[0,0]]),0,atol=2e-14)
    np.testing.assert_allclose(matrix@result.value,rhs,atol=2e-14)


def test_decisive_projected_contraction_age_and_rhs_derivatives():
    target=jnp.zeros(2)
    centers=jnp.array([[-1.,-2.],[1.,2.]])
    biases=jnp.array([2.25,.25])
    matrix=jnp.array([[0.,1.]])
    rhs=jnp.ones(1)
    result=jax.jit(gaussian_heat_inverse_constrained)(target,centers,biases,matrix,rhs)
    assert bool(result.converged) and result.contraction_bound==.5
    plain=gaussian_heat_inverse(target,centers,biases)
    assert not bool(plain.converged) and plain.contraction_bound==2.5
    np.testing.assert_allclose(result.value,[0,1],rtol=0,atol=1e-14)
    ds,dd=jax.jit(jax.jacfwd(lambda s,d:gaussian_heat_inverse_constrained(target,centers,biases,matrix,d,time=s).value,argnums=(0,1)))(jnp.asarray(1.),rhs)
    np.testing.assert_allclose(ds,[-2,0],rtol=2e-13,atol=2e-13)
    np.testing.assert_allclose(dd,[[2],[1]],rtol=2e-13,atol=2e-13)


@pytest.mark.parametrize("adjoint",["dense","low_rank"])
def test_seven_input_gradients_match_independent_root_differences_and_hessian(adjoint):
    args=_parameters()
    weights=jnp.array([.7,-.4])
    def loss(*p):
        return weights@gaussian_heat_inverse_constrained(*p,adjoint=adjoint).value
    grads=jax.jit(jax.grad(loss,argnums=tuple(range(7))))(*args)
    step=2e-5
    for parameter,gradient in enumerate(grads):
        expected=np.empty(args[parameter].shape)
        for index in np.ndindex(expected.shape):
            plus,minus=[np.array(v) for v in args],[np.array(v) for v in args]
            plus[parameter][index]+=step
            minus[parameter][index]-=step
            expected[index]=np.asarray(weights)@(_reference(plus)-_reference(minus))/(2*step)
        np.testing.assert_allclose(gradient,expected,rtol=3e-7,atol=3e-9)
    def combined(z):
        return loss(*args[:3],z[:2].reshape(1,2),z[2:3],z[3],z[4])
    hessian=jax.jit(jax.hessian(combined))(jnp.concatenate((args[3][0],args[4],jnp.stack(args[5:]))))
    assert np.isfinite(hessian).all() and abs(float(hessian[2,4]))>1e-5
    np.testing.assert_allclose(hessian,hessian.T,rtol=2e-11,atol=2e-12)
    diag=jax.grad(lambda C:gaussian_heat_inverse_constrained(*args[:3],C,*args[4:]).constraint_reciprocal_condition)(args[3])
    np.testing.assert_array_equal(diag,jnp.zeros_like(args[3]))


@pytest.mark.parametrize("dtype",[jnp.float32,jnp.float64])
def test_vmap_batches_empty_full_constraints_dtypes_and_zero_model_gradients(dtype):
    args=tuple(v.astype(dtype) for v in _parameters())
    target=jnp.stack((args[0],args[0]*2,args[0]/2)).reshape(1,3,2)
    result=jax.jit(gaussian_heat_inverse_constrained)(target,*args[1:])
    mapped=jax.vmap(lambda query:gaussian_heat_inverse_constrained(query,*args[1:]))(target[0])
    np.testing.assert_allclose(mapped.value,result.value[0],rtol=2e-6,atol=2e-6)
    assert result.value.dtype==result.feasibility_residual.dtype==dtype
    empty=gaussian_heat_inverse_constrained(target,*args[1:3],jnp.empty((0,2)),jnp.empty(0),*args[5:])
    plain=gaussian_heat_inverse(target,*args[1:3],*args[5:])
    np.testing.assert_allclose(empty.value,plain.value,rtol=2e-6,atol=2e-6)
    assert empty.feasibility_residual.dtype==dtype and bool(empty.constraints_valid)
    full_args=(*args[:3],jnp.array([[1.,2.],[0.,1.]],dtype=dtype),jnp.array([3.,1.],dtype=dtype),*args[5:])
    full=gaussian_heat_inverse_constrained(*full_args,max_iterations=0)
    assert bool(full.converged) and full.contraction_bound==0 and full.residual_bound==0
    np.testing.assert_allclose(full.value,[1,1],rtol=2e-6,atol=2e-6)
    grads=jax.grad(lambda *p:gaussian_heat_inverse_constrained(*p,max_iterations=0).value.sum(),argnums=tuple(range(7)))(*full_args)
    assert all(np.isfinite(g).all() for g in grads)
    for index in (0,1,2,5,6):
        np.testing.assert_array_equal(grads[index],jnp.zeros_like(full_args[index]))
    no_queries=gaussian_heat_inverse_constrained(target[:0],*args[1:])
    assert no_queries.value.shape==(0,3,2) and no_queries.converged.shape==(0,3)


def test_jitted_vmap_over_live_constraint_parameters():
    args=_parameters()
    matrices=jnp.stack((args[3],args[3]+jnp.array([[.2,-.1]])))
    rhs=jnp.stack((args[4],args[4]+.1))
    result=jax.jit(jax.vmap(lambda C,d:gaussian_heat_inverse_constrained(*args[:3],C,d,*args[5:])))(matrices,rhs)
    assert bool(result.converged.all())
    np.testing.assert_allclose(jnp.einsum('brd,bd->br',matrices,result.value),rhs,rtol=1e-13,atol=1e-13)
    gradients=jax.jit(jax.vmap(jax.jacrev(lambda C,d:gaussian_heat_inverse_constrained(*args[:3],C,d,*args[5:]).value,argnums=(0,1))))(matrices,rhs)
    assert all(np.isfinite(g).all() for g in gradients)


def test_row_scale_and_translation_bias_gauges():
    args=_parameters()
    base=gaussian_heat_inverse_constrained(*args)
    for scale in (-1e200,1e-200):
        value=gaussian_heat_inverse_constrained(*args[:3],args[3]*scale,args[4]*scale,*args[5:]).value
        np.testing.assert_allclose(value,base.value,rtol=2e-13,atol=2e-13)
    shift=jnp.array([8.,-4.])
    changed=gaussian_heat_inverse_constrained(args[0],args[1]+shift,args[2]+16,args[3],args[4]+args[3]@shift,*args[5:])
    np.testing.assert_allclose(changed.value,base.value+shift,rtol=2e-13,atol=2e-13)
    first=gaussian_heat_inverse_constrained(args[0],args[1],jnp.zeros(3),*args[3:])
    second=gaussian_heat_inverse_constrained(args[0],args[1],jnp.full(3,1e100),*args[3:])
    np.testing.assert_array_equal(first.value,second.value)


def test_multiple_constraint_rows_basis_invariance_and_higher_derivatives():
    target=jnp.array([.2,-.1,.15])
    centers=jnp.array([[0.,.2,.1],[.3,-.1,.2],[-.2,.1,-.1]])
    matrix=jnp.array([[1.,2.,0.],[0.,1.,-1.]])
    rhs=jnp.array([.3,-.2])
    basis=jnp.array([[2.,1.],[-1.,3.]])
    def solve(C,d):
        return gaussian_heat_inverse_constrained(target,centers,jnp.array([.2,-.1,.3]),C,d).value
    expected=jax.jit(solve)(matrix,rhs)
    np.testing.assert_allclose(solve(basis@matrix,basis@rhs),expected,rtol=2e-13,atol=2e-13)
    np.testing.assert_allclose(matrix@expected,rhs,rtol=2e-13,atol=2e-13)
    hessian=jax.jit(jax.hessian(lambda C,d:jnp.sum(solve(C,d)**2),argnums=(0,1)))(matrix,rhs)
    assert all(np.isfinite(v).all() for v in jax.tree_util.tree_leaves(hessian))


@pytest.mark.parametrize("matrix,rhs",[([[1.,2.],[2.,4.]],[1.,2.]),([[0.,0.]],[0.]),
                                       ([[1.,0.],[1.,1e-12]],[0.,0.]),([[float('nan'),1.]],[0.]),
                                       ([[1.,0.]],[float('inf')]),([[1e-308,0.]],[1e308])])
def test_constraint_refusals(matrix,rhs):
    result=jax.jit(gaussian_heat_inverse_constrained)(jnp.zeros(2),jnp.array([[0.,0.]]),jnp.zeros(1),jnp.array(matrix),jnp.array(rhs))
    assert not bool(result.constraints_valid) and not bool(result.converged) and bool(jnp.isnan(result.value).all())
    assert bool(jnp.isinf(result.feasibility_residual)) and bool(jnp.isinf(result.residual_bound))


def test_query_model_failure_and_exhausted_inverse():
    args=list(_parameters())
    args[0]=jnp.stack((args[0],jnp.array([jnp.nan,0.])))
    result=gaussian_heat_inverse_constrained(*args)
    np.testing.assert_array_equal(result.converged,[True,False])
    assert np.isnan(result.value[1]).all()
    early=gaussian_heat_inverse_constrained(*_parameters(),max_iterations=0)
    assert not bool(early.converged) and bool(jnp.isfinite(early.residual_bound)) and early.residual_bound>1e-5
    args=list(_parameters())
    args[5]=0.
    invalid=gaussian_heat_inverse_constrained(*args)
    assert not bool(invalid.converged) and bool(invalid.constraints_valid) and bool(jnp.isinf(invalid.contraction_bound))


def test_low_rank_high_dimension_has_no_dense_projector_in_gradient():
    dimension=257
    target=jnp.full(dimension,.01)
    centers=jnp.zeros((3,dimension)).at[1,0].set(.2).at[2,1].set(-.3)
    matrix=jnp.zeros((2,dimension)).at[0,2].set(1).at[1,3].set(1)
    biases=jnp.array([0.,.1,-.1])
    gradient=jax.grad(lambda v,C:gaussian_heat_inverse_constrained(v,centers,biases,C,jnp.array([.2,-.1]),adjoint='low_rank').value.sum(),argnums=(0,1))
    graph=jax.make_jaxpr(gradient)(target,matrix)
    def inspect(jaxpr):
        if hasattr(jaxpr,'jaxpr'):
            inspect(jaxpr.jaxpr)
        elif hasattr(jaxpr,'eqns'):
            for equation in jaxpr.eqns:
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
    assert all(np.isfinite(g).all() for g in jax.jit(gradient)(target,matrix))


@pytest.mark.parametrize("options",[{"constraint_rcond":0},{"constraint_rcond":1},{"constraint_rcond":True},
                                   {"feasibility_tolerance":0},{"max_iterations":True},{"tolerance":float('inf')},{"adjoint":"bad"}])
def test_invalid_options_raise(options):
    with pytest.raises((TypeError,ValueError)):
        gaussian_heat_inverse_constrained(*_parameters(),**options)


def test_invalid_shapes_and_types():
    args=_parameters()
    for matrix,rhs in (([[1.]],[0.]),([[1.,0.],[0.,1.],[1.,1.]],[0.,0.,0.]),([[1.,0.]],[])):
        with pytest.raises(ValueError):
            gaussian_heat_inverse_constrained(*args[:3],matrix,rhs,*args[5:])
    for matrix in ([[True,False]],[[1j,0j]]):
        with pytest.raises(TypeError):
            gaussian_heat_inverse_constrained(*args[:3],matrix,[0.],*args[5:])
