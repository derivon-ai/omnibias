# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Fixed Chebyshev artifact evaluation, query derivatives and domain guards."""

from __future__ import annotations

from copy import copy
from fractions import Fraction as F
from typing import Any

import numpy as np
import pytest
import torch
from omnibias.core.verified.gaussian_chebyshev import approximate_gaussian_inverse
from omnibias.core.verified.gaussian_heat import GaussianHeatMap
from omnibias.core.verified.gaussian_tube import certify_gaussian_inverse_tube
from omnibias.torch.gaussian_chebyshev import evaluate_gaussian_inverse_approximation


@pytest.fixture(scope="module")
def approximation() -> Any:
    pytest.importorskip("mpmath")
    model = GaussianHeatMap(((3, -2), (4, -2), (3, -1)), (0, F(1, 8), F(1, 8)), time=2)
    tube = certify_gaussian_inverse_tube(model)
    assert tube is not None
    result = approximate_gaussian_inverse(tube, (-F(1, 6), -F(1, 6)), (F(1, 5), -F(1, 7)),
                                          half_width=F(1, 4), degree=8)
    assert result is not None
    return result


def _affine(width: F = F(2)) -> Any:
    model = GaussianHeatMap(((2, -1),), (3,), time=F(3, 2))
    tube = certify_gaussian_inverse_tube(model)
    assert tube is not None
    result = approximate_gaussian_inverse(tube, (F(1, 3), -F(1, 4)), (2, -3), half_width=width, degree=1)
    assert result is not None
    return result


def _exact(approximation: Any, parameter: float) -> list[float]:
    value = F.from_float(parameter)
    scaled = value / approximation.half_width
    basis = [F(1)]
    if len(approximation.coefficients) > 1:
        basis.append(scaled)
    for _ in approximation.coefficients[2:]:
        basis.append(2 * scaled * basis[-1] - basis[-2])
    return [float(approximation.tube.model.time * (base + value * direction) + center
                  + sum(F.from_float(row[axis]) * polynomial
                        for row, polynomial in zip(approximation.coefficients, basis, strict=True)))
            for axis, (base, direction, center) in enumerate(zip(approximation.base_target, approximation.direction,
                                                               approximation.tube.reference_center, strict=True))]


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_coupled_real_artifact_matches_independent_exact_polynomial(approximation: Any, dtype: torch.dtype) -> None:
    query = torch.tensor([[-0.25, -0.12], [0.0, 0.25]], dtype=dtype)
    result = evaluate_gaussian_inverse_approximation(query, approximation)
    expected = query.new_tensor([_exact(approximation, float(value)) for value in query.flatten()]).reshape(2, 2, 2)
    assert result.value.shape == (2, 2, 2) and result.valid.shape == query.shape
    assert result.value.dtype == dtype and result.valid.dtype == torch.bool and result.valid.all()
    tolerance = 2e-6 if dtype == torch.float32 else 2e-14
    torch.testing.assert_close(result.value, expected, rtol=tolerance, atol=tolerance)


def test_parameter_gradcheck_and_gradgradcheck(approximation: Any) -> None:
    query = torch.tensor([-0.13, 0.07], dtype=torch.float64, requires_grad=True)

    def function(value: torch.Tensor) -> torch.Tensor:
        return evaluate_gaussian_inverse_approximation(value, approximation).value

    assert torch.autograd.gradcheck(function, (query,))
    assert torch.autograd.gradgradcheck(function, (query,))


def test_first_four_parameter_derivatives_match_independent_chebyshev_derivatives(approximation: Any) -> None:
    query = torch.tensor(0.07, dtype=torch.float64, requires_grad=True)
    weights = query.new_tensor([0.7, -0.2])
    value = (evaluate_gaussian_inverse_approximation(query, approximation).value * weights).sum()
    coefficients = np.asarray(approximation.coefficients) @ weights.detach().numpy()
    for order in range(1, 5):
        value = torch.autograd.grad(value, query, create_graph=True)[0]
        expected = np.polynomial.chebyshev.chebval(float(query.detach()) / float(approximation.half_width),
                    np.polynomial.chebyshev.chebder(coefficients, m=order)) / float(approximation.half_width) ** order
        if order == 1:
            expected += float(approximation.tube.model.time) * np.dot(weights.detach().numpy(),
                        [float(component) for component in approximation.direction])
        torch.testing.assert_close(value, query.new_tensor(expected), rtol=3e-12, atol=2e-12)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_affine_artifact_and_closed_endpoints(dtype: torch.dtype) -> None:
    approximation = _affine()
    query = torch.tensor([-2.0, 0.0, 2.0], dtype=dtype, requires_grad=True)
    result = evaluate_gaussian_inverse_approximation(query, approximation)
    expected = 1.5 * (query.new_tensor([1 / 3, -0.25]) + query[:, None] * query.new_tensor([2.0, -3.0]))
    expected = expected + query.new_tensor([2.0, -1.0])
    assert result.valid.all()
    torch.testing.assert_close(result.value, expected)
    gradient = torch.autograd.grad(result.value.sum(), query)[0]
    torch.testing.assert_close(gradient, torch.full_like(query, -1.5))


def test_constant_chebyshev_coefficient_has_no_half_weight() -> None:
    # Deliberately altered metadata checks the evaluator's polynomial convention;
    # this copy is not a proof artifact and its original certificate is unused.
    approximation = copy(_affine())
    object.__setattr__(approximation, "coefficients", ((0.75, -0.125),))
    query = torch.tensor(0.0, dtype=torch.float64)
    result = evaluate_gaussian_inverse_approximation(query, approximation)
    torch.testing.assert_close(result.value, query.new_tensor([3.25, -1.5]), rtol=0, atol=0)
    assert result.valid


def test_outside_nonfinite_empty_and_scalar_queries(approximation: Any) -> None:
    query = torch.tensor([0.0, 0.2501, -0.2501, float("nan"), float("inf"), -float("inf")], dtype=torch.float64)
    result = evaluate_gaussian_inverse_approximation(query, approximation)
    assert result.valid.tolist() == [True, False, False, False, False, False]
    assert torch.isnan(result.value[1:]).all() and torch.isfinite(result.value[0]).all()
    scalar = evaluate_gaussian_inverse_approximation(query[0], approximation)
    assert scalar.value.shape == (2,) and scalar.valid.shape == ()
    empty = evaluate_gaussian_inverse_approximation(query[:0].reshape(0, 2), approximation)
    assert empty.value.shape == (0, 2, 2) and empty.valid.shape == (0, 2)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_rounded_up_endpoint_does_not_accept_a_query_outside_exact_domain(dtype: torch.dtype) -> None:
    approximation = _affine(F(1, 10))
    outside = torch.tensor(0.1, dtype=dtype)
    inside = torch.nextafter(outside, torch.zeros_like(outside))
    result = evaluate_gaussian_inverse_approximation(torch.stack((inside, outside, -outside)), approximation)
    assert result.valid.tolist() == [True, False, False]


@pytest.mark.parametrize("field,value", [("half_width", F(10**400)), ("half_width", F(1, 10**400)),
                                        ("base_target", (F(10**400), F(0))),
                                        ("direction", (F(0), F(10**400))),
                                        ("coefficients", ((float("inf"), 0.0),))])
def test_nonrepresentable_metadata_returns_invalid(field: str, value: Any) -> None:
    approximation = copy(_affine())
    object.__setattr__(approximation, field, value)
    result = evaluate_gaussian_inverse_approximation(torch.zeros(2, dtype=torch.float64), approximation)
    assert not result.valid.any() and torch.isnan(result.value).all()


def test_float32_conversion_and_result_overflow_are_explicit() -> None:
    approximation = copy(_affine())
    object.__setattr__(approximation, "coefficients", ((1e100, 0.0),))
    result = evaluate_gaussian_inverse_approximation(torch.zeros(2, dtype=torch.float32), approximation)
    assert not result.valid.any() and torch.isnan(result.value).all()
    object.__setattr__(approximation, "coefficients", ((1e308, 0.0), (1e308, 0.0)))
    result = evaluate_gaussian_inverse_approximation(torch.tensor(2.0, dtype=torch.float64), approximation)
    assert not result.valid and torch.isnan(result.value).all()


@pytest.mark.parametrize("query", [torch.tensor(1), torch.tensor(True), torch.tensor(1j),
                                   torch.tensor(0.0, dtype=torch.float16), [0.0]])
def test_invalid_query_types_are_rejected(query: Any) -> None:
    with pytest.raises(TypeError, match="float32 or float64"):
        evaluate_gaussian_inverse_approximation(query, _affine())


def test_wrong_artifact_type_is_rejected() -> None:
    with pytest.raises(TypeError, match="GaussianInverseApproximation"):
        evaluate_gaussian_inverse_approximation(torch.tensor(0.0), None)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_cuda_device_and_parameter_gradient(approximation: Any) -> None:
    query = torch.tensor([0.03, -0.1], dtype=torch.float64, device="cuda", requires_grad=True)
    result = evaluate_gaussian_inverse_approximation(query, approximation)
    assert result.value.device == query.device and result.valid.device == query.device
    assert torch.isfinite(torch.autograd.grad(result.value.sum(), query)[0]).all()
