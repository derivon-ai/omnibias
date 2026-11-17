# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Independent geometric oracles, differentiation and visible refusal states."""

from fractions import Fraction

import numpy as np
import pytest
import torch
from omnibias.torch.affine_geometry import affine_project, linearized_equality_step


@pytest.fixture(scope="module", autouse=True)
def _one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _rational_oracle():
    # A=[[1,2,0],[0,1,1]], b=[2,-1], x=[3,-1,2].
    # Solve the two normal equations by an exact 2x2 determinant formula.
    matrix = [[Fraction(v) for v in row] for row in [[1, 2, 0], [0, 1, 1]]]
    query = [Fraction(v) for v in [3, -1, 2]]
    rhs = [Fraction(2), Fraction(-1)]
    residual = [sum(a * v for a, v in zip(row, query, strict=True)) - b
                for row, b in zip(matrix, rhs, strict=True)]
    g00, g01, g11 = Fraction(5), Fraction(2), Fraction(2)
    det = g00 * g11 - g01 * g01
    lam = [(g11 * residual[0] - g01 * residual[1]) / det,
           (g00 * residual[1] - g01 * residual[0]) / det]
    correction = [-sum(lam[i] * matrix[i][j] for i in range(2)) for j in range(3)]
    projected = [v + d for v, d in zip(query, correction, strict=True)]
    distance = sum(d * d for d in correction) / 2
    return np.array(projected, dtype=float), np.array(correction, dtype=float), float(distance)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_exact_rational_oracle_feasibility_idempotence_and_orthogonality(dtype):
    query = torch.tensor([3, -1, 2], dtype=dtype)
    matrix = query.new_tensor([[1, 2, 0], [0, 1, 1]])
    rhs = query.new_tensor([2, -1])
    result = affine_project(query, matrix, rhs)
    tolerance = 3e-6 if dtype == torch.float32 else 2e-14
    assert result.valid and 0 < result.rcond <= 1
    for got, expected in zip(result[:3], _rational_oracle(), strict=True):
        np.testing.assert_allclose(got, expected, rtol=tolerance, atol=tolerance)
    torch.testing.assert_close(matrix @ result.projected, rhs, rtol=0, atol=tolerance)
    torch.testing.assert_close(affine_project(result.projected, matrix, rhs).projected,
                               result.projected, rtol=0, atol=tolerance)
    # (2,-1,1) spans the exact null space of A.
    torch.testing.assert_close(result.correction @ query.new_tensor([2, -1, 1]),
                               query.new_zeros(()), rtol=0, atol=tolerance)


def test_projection_distance_and_parameter_first_second_derivatives():
    args = tuple(torch.tensor(value, dtype=torch.float64, requires_grad=True) for value in (
        [0.4, -0.7, 1.2], [[1.1, 2.3, 0.4], [0.2, -0.5, 1.3]], [0.1, -0.2]))

    def outputs(*values):
        r = affine_project(*values)
        return torch.cat((r.projected, r.correction, r.half_squared_distance[None]))

    assert torch.autograd.gradcheck(outputs, args, atol=2e-6, rtol=2e-5)
    assert torch.autograd.gradgradcheck(outputs, args, atol=3e-6, rtol=3e-5)
    x, matrix, rhs = args
    result = affine_project(*args)
    gradient = torch.autograd.grad(result.half_squared_distance, x, create_graph=True)[0]
    torch.testing.assert_close(gradient, -result.correction, atol=2e-14, rtol=2e-14)
    jacobian = torch.autograd.functional.jacobian(lambda q: affine_project(q, matrix, rhs).projected, x)
    hessian = torch.autograd.functional.hessian(lambda q: affine_project(q, matrix, rhs).half_squared_distance, x)
    torch.testing.assert_close(jacobian + hessian, torch.eye(3, dtype=x.dtype), atol=2e-14, rtol=2e-14)
    torch.testing.assert_close(jacobian @ jacobian, jacobian, atol=2e-14, rtol=2e-14)
    torch.testing.assert_close(jacobian @ matrix.mT, torch.zeros_like(matrix.mT), atol=2e-14, rtol=0)
    assert not result.rcond.requires_grad and not result.valid.requires_grad


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_batches_vmap_empty_square_and_row_transformations(dtype):
    x = torch.arange(18, dtype=dtype).reshape(2, 3, 3) / 7
    matrix = x.new_tensor([[1.1, 2.3, 0.4], [0.2, -0.5, 1.3]])
    rhs = x.new_tensor([0.1, -0.2])
    result = affine_project(x, matrix, rhs)
    mapped = torch.vmap(lambda q: affine_project(q, matrix, rhs))(x.reshape(-1, 3))
    tolerance = 5e-6 if dtype == torch.float32 else 3e-14
    torch.testing.assert_close(result.projected, mapped.projected.reshape_as(x), rtol=tolerance, atol=tolerance)
    for transform in (x.new_tensor([[1e-8, 0], [0, -1e8]]), x.new_tensor([[2, -1], [1, 3]])):
        changed = affine_project(x, transform @ matrix, transform @ rhs)
        assert changed.valid.all()
        torch.testing.assert_close(changed.projected, result.projected, rtol=tolerance, atol=tolerance)
    empty = affine_project(x, x.new_empty((0, 3)), x.new_empty((0,)))
    assert empty.valid.all() and empty.rcond == 1
    torch.testing.assert_close(empty.projected, x)
    torch.testing.assert_close(empty.correction, torch.zeros_like(x))
    square = affine_project(x, x.new_tensor([[1, 2, 0], [0, 1, 1], [1, 0, 1]]), x.new_tensor([2, -1, 4]))
    assert square.valid.all()
    torch.testing.assert_close(square.projected, x.new_tensor([4, -1, 0]).expand_as(x), rtol=tolerance, atol=tolerance)
    single = x[0, 0].clone().requires_grad_()
    jacobian = torch.autograd.functional.jacobian(lambda q: affine_project(q, torch.eye(3, dtype=dtype), rhs.new_ones(3)).projected, single)
    torch.testing.assert_close(jacobian, torch.zeros_like(jacobian), rtol=0, atol=0)


@pytest.mark.parametrize("bad_matrix,bad_rhs", [
    ([[1, 2, 3], [2, 4, 6]], [1, 2]),
    ([[0, 0, 0], [1, 0, 0]], [1, 2]),
    ([[float("nan"), 1, 0], [0, 1, 0]], [1, 2]),
    ([[1, 0, 0], [0, 1, 0]], [float("inf"), 2]),
])
def test_invalid_constraints_refuse_with_safe_masked_backward(bad_matrix, bad_rhs):
    args = tuple(torch.tensor(v, dtype=torch.float64, requires_grad=True)
                 for v in ([1, 2, 3], bad_matrix, bad_rhs))
    result = affine_project(*args)
    assert not result.valid and torch.isnan(result.projected).all()
    assert torch.isnan(result.correction).all() and torch.isnan(result.half_squared_distance)
    gradients = torch.autograd.grad(torch.nan_to_num(result.projected).sum(), args)
    assert all(torch.isfinite(g).all() for g in gradients)


def test_per_query_refusal_rank_cutoff_and_shape_errors():
    x = torch.tensor([[1., 2.], [float("nan"), 2.]], dtype=torch.float32, requires_grad=True)
    result = affine_project(x, [[1, 0]], [0])
    assert result.valid.tolist() == [True, False]
    gradient = torch.autograd.grad(torch.nan_to_num(result.projected).sum(), x)[0]
    torch.testing.assert_close(gradient, torch.tensor([[0., 1.], [0., 0.]]))
    # Cond(A) around 10^4 is supported in float32; sqrt(eps) would reject it.
    accepted = affine_project(torch.zeros(2), [[1., 0.], [1., 1e-4]], [0., 0.])
    assert accepted.valid and accepted.rcond > 2 * torch.finfo(torch.float32).eps
    assert not affine_project(torch.zeros(2), [[1., 0.], [1., 1e-4]], [0., 0.], rank_rcond=1e-3).valid
    with pytest.raises(ValueError, match="shape"):
        affine_project(torch.zeros(2), torch.ones(3, 2), torch.zeros(3))
    with pytest.raises(ValueError, match="shape"):
        affine_project(torch.zeros(2), torch.ones(1, 2), torch.zeros(1, 1))
    with pytest.raises(ValueError, match="rank_rcond"):
        affine_project(torch.zeros(2), [[1., 0.]], [0.], rank_rcond=0)
    with pytest.raises(TypeError, match="float32"):
        affine_project(torch.zeros(2, dtype=torch.int64), [[1, 0]], [0])


def test_linearized_circle_step_is_not_nonlinear_projection_and_keeps_iteration_derivative():
    x = torch.tensor([2., 0.], dtype=torch.float64, requires_grad=True)

    def step(q):
        return linearized_equality_step(q, (q.square().sum() - 1)[None], (2 * q)[None])

    result = step(x)
    assert result.valid
    torch.testing.assert_close(result.value, x.new_tensor([1.25, 0]), atol=2e-15, rtol=0)
    assert result.value.square().sum() != 1
    torch.testing.assert_close(result.local_half_squared_distance, x.new_tensor(0.28125))
    jacobian = torch.autograd.functional.jacobian(lambda q: step(q).value, x)
    torch.testing.assert_close(jacobian, x.new_tensor([[.375, 0], [0, .625]]), atol=2e-14, rtol=0)
    batched = linearized_equality_step(x.expand(2, 2), x.new_tensor([[3], [1]]), x.new_tensor([[4, 0]]))
    torch.testing.assert_close(batched.value, x.new_tensor([[1.25, 0], [1.75, 0]]))
    with pytest.raises(ValueError, match="residual"):
        linearized_equality_step(x, x.new_tensor(3.), x.new_tensor([[4, 0]]))
def test_compiled_affine_projection_preserves_parameter_gradients():
    """The native tensor path can be captured without a host rank branch."""
    import torch
    from omnibias.torch.affine_geometry import affine_project

    x = torch.tensor([[0.3, -0.2, 0.7]], dtype=torch.float64, requires_grad=True)
    a = torch.tensor([[1.2, -0.4, 0.3]], dtype=torch.float64, requires_grad=True)
    b = torch.tensor([0.1], dtype=torch.float64, requires_grad=True)

    def loss(query, matrix, rhs):
        result = affine_project(query, matrix, rhs)
        return result.half_squared_distance.sum() + result.projected.square().sum()

    compiled = torch.compile(loss, backend="eager", fullgraph=True)
    ordinary = loss(x, a, b)
    captured = compiled(x, a, b)
    torch.testing.assert_close(captured, ordinary)
    expected = torch.autograd.grad(ordinary, (x, a, b))
    actual = torch.autograd.grad(captured, (x, a, b))
    for observed, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(observed, reference)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_saturated_sigmoid_second_derivative_has_false_floating_zeros(dtype):
    from omnibias.core.affine_geometry import exact_affine_geometry

    locations = torch.tensor([[-1000.0], [0.0], [1000.0]], dtype=dtype)
    sigmoid = torch.sigmoid(locations[:, 0])
    second = sigmoid * (1 - sigmoid) * (1 - 2 * sigmoid)
    # Mathematically the finite sigmoid second derivative vanishes only at
    # zero. Saturation creates two additional floating zeros off that locus.
    assert bool((second == 0).all())
    exact = exact_affine_geometry([[1]], [0])
    assert [exact.contains([value]) for value in (-1000, 0, 1000)] == [False, True, False]
    result = affine_project(locations, torch.ones((1, 1), dtype=dtype), torch.zeros(1, dtype=dtype))
    torch.testing.assert_close(result.half_squared_distance, torch.tensor([500000.0, 0.0, 500000.0], dtype=dtype))


def test_local_nonlinear_loss_hessian_is_normal_projector_at_feasibility():
    point = torch.tensor([0.6, 0.8], dtype=torch.float64)

    def local_loss(x):
        residual = (x.square().sum() - 1).reshape(1)
        jacobian = (2 * x).reshape(1, 2)
        return linearized_equality_step(x, residual, jacobian).local_half_squared_distance

    hessian = torch.autograd.functional.hessian(local_loss, point)
    torch.testing.assert_close(hessian, torch.outer(point, point), atol=2e-14, rtol=2e-14)
