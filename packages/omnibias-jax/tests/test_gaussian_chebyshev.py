# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Fixed Chebyshev artifacts under JIT, vmap and higher query derivatives."""

from __future__ import annotations

from copy import copy
from fractions import Fraction as F
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from omnibias.core.verified.gaussian_chebyshev import approximate_gaussian_inverse
from omnibias.core.verified.gaussian_heat import GaussianHeatMap
from omnibias.core.verified.gaussian_tube import certify_gaussian_inverse_tube
from omnibias.jax.gaussian_chebyshev import evaluate_gaussian_inverse_approximation

jax.config.update("jax_enable_x64", True)


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


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_coupled_real_artifact_matches_independent_exact_polynomial(approximation: Any, dtype: Any) -> None:
    query = jnp.asarray([[-0.25, -0.12], [0.0, 0.25]], dtype=dtype)
    result = evaluate_gaussian_inverse_approximation(query, approximation)
    expected = np.asarray([_exact(approximation, float(value)) for value in np.asarray(query).flatten()]).reshape(2, 2, 2)
    assert result.value.shape == (2, 2, 2) and result.valid.shape == query.shape
    assert result.value.dtype == dtype and result.valid.dtype == jnp.bool_ and bool(result.valid.all())
    tolerance = 2e-6 if dtype == jnp.float32 else 2e-14
    np.testing.assert_allclose(result.value, expected, rtol=tolerance, atol=tolerance)


def test_jit_vmap_and_varying_runtime_queries(approximation: Any) -> None:
    query = jnp.asarray([[-0.2, -0.1], [0.0, 0.2]], dtype=jnp.float64)

    def evaluate(value: Any) -> Any:
        return evaluate_gaussian_inverse_approximation(value, approximation)

    eager = evaluate(query)
    compiled = jax.jit(evaluate)(query)
    mapped = jax.jit(jax.vmap(evaluate))(query)
    np.testing.assert_allclose(compiled.value, eager.value, rtol=2e-15, atol=2e-15)
    np.testing.assert_allclose(mapped.value, eager.value, rtol=2e-15, atol=2e-15)
    np.testing.assert_array_equal(compiled.valid, eager.valid)
    assert not np.array_equal(jax.jit(evaluate)(query + 0.01).value, compiled.value)


def test_first_four_parameter_derivatives_match_independent_chebyshev_derivatives(approximation: Any) -> None:
    query = jnp.asarray(0.07, dtype=jnp.float64)
    weights = jnp.asarray([0.7, -0.2], dtype=query.dtype)

    def function(value: Any) -> Any:
        return jnp.dot(evaluate_gaussian_inverse_approximation(value, approximation).value, weights)

    coefficients = np.asarray(approximation.coefficients) @ np.asarray(weights)
    for order in range(1, 5):
        function = jax.grad(function)
        actual = jax.jit(function)(query)
        expected = np.polynomial.chebyshev.chebval(float(query) / float(approximation.half_width),
                    np.polynomial.chebyshev.chebder(coefficients, m=order)) / float(approximation.half_width) ** order
        if order == 1:
            expected += float(approximation.tube.model.time) * np.dot(np.asarray(weights),
                        [float(component) for component in approximation.direction])
        np.testing.assert_allclose(actual, expected, rtol=3e-12, atol=2e-12)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_affine_artifact_and_closed_endpoints(dtype: Any) -> None:
    approximation = _affine()
    query = jnp.asarray([-2.0, 0.0, 2.0], dtype=dtype)
    result = evaluate_gaussian_inverse_approximation(query, approximation)
    expected = 1.5 * (jnp.asarray([1 / 3, -0.25], dtype=dtype) + query[:, None] * jnp.asarray([2.0, -3.0], dtype=dtype))
    expected = expected + jnp.asarray([2.0, -1.0], dtype=dtype)
    assert bool(result.valid.all())
    np.testing.assert_allclose(result.value, expected, rtol=2e-7, atol=2e-7)
    gradient = jax.grad(lambda value: evaluate_gaussian_inverse_approximation(value, approximation).value.sum())(query)
    np.testing.assert_allclose(gradient, jnp.full_like(query, -1.5))


def test_constant_chebyshev_coefficient_has_no_half_weight() -> None:
    # Altered test metadata is not a proof artifact; only its numerical
    # coefficient convention is tested and its original certificate is unused.
    approximation = copy(_affine())
    object.__setattr__(approximation, "coefficients", ((0.75, -0.125),))
    result = evaluate_gaussian_inverse_approximation(jnp.asarray(0.0, dtype=jnp.float64), approximation)
    np.testing.assert_array_equal(result.value, [3.25, -1.5])
    assert bool(result.valid)


def test_outside_nonfinite_empty_and_scalar_queries(approximation: Any) -> None:
    query = jnp.asarray([0.0, 0.2501, -0.2501, jnp.nan, jnp.inf, -jnp.inf], dtype=jnp.float64)
    result = jax.jit(lambda value: evaluate_gaussian_inverse_approximation(value, approximation))(query)
    np.testing.assert_array_equal(result.valid, [True, False, False, False, False, False])
    assert bool(jnp.isnan(result.value[1:]).all()) and bool(jnp.isfinite(result.value[0]).all())
    scalar = evaluate_gaussian_inverse_approximation(query[0], approximation)
    assert scalar.value.shape == (2,) and scalar.valid.shape == ()
    empty = evaluate_gaussian_inverse_approximation(query[:0].reshape(0, 2), approximation)
    assert empty.value.shape == (0, 2, 2) and empty.valid.shape == (0, 2)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_rounded_up_endpoint_does_not_accept_a_query_outside_exact_domain(dtype: Any) -> None:
    approximation = _affine(F(1, 10))
    outside = jnp.asarray(0.1, dtype=dtype)
    inside = jnp.nextafter(outside, jnp.zeros_like(outside))
    result = jax.jit(lambda value: evaluate_gaussian_inverse_approximation(value, approximation))(
        jnp.stack((inside, outside, -outside)))
    np.testing.assert_array_equal(result.valid, [True, False, False])


@pytest.mark.parametrize("field,value", [("half_width", F(10**400)), ("half_width", F(1, 10**400)),
                                        ("base_target", (F(10**400), F(0))),
                                        ("direction", (F(0), F(10**400))),
                                        ("coefficients", ((float("inf"), 0.0),))])
def test_nonrepresentable_metadata_returns_invalid(field: str, value: Any) -> None:
    approximation = copy(_affine())
    object.__setattr__(approximation, field, value)
    result = evaluate_gaussian_inverse_approximation(jnp.zeros(2, dtype=jnp.float64), approximation)
    assert not bool(result.valid.any()) and bool(jnp.isnan(result.value).all())


def test_float32_conversion_and_result_overflow_are_explicit() -> None:
    approximation = copy(_affine())
    object.__setattr__(approximation, "coefficients", ((1e100, 0.0),))
    with np.errstate(over="ignore"):
        result = evaluate_gaussian_inverse_approximation(jnp.zeros(2, dtype=jnp.float32), approximation)
    assert not bool(result.valid.any()) and bool(jnp.isnan(result.value).all())
    object.__setattr__(approximation, "coefficients", ((1e308, 0.0), (1e308, 0.0)))
    result = evaluate_gaussian_inverse_approximation(jnp.asarray(2.0, dtype=jnp.float64), approximation)
    assert not bool(result.valid) and bool(jnp.isnan(result.value).all())


@pytest.mark.parametrize("query", [jnp.asarray(1), jnp.asarray(True), jnp.asarray(1j),
                                   jnp.asarray(0.0, dtype=jnp.float16)])
def test_invalid_query_types_are_rejected(query: Any) -> None:
    with pytest.raises(TypeError, match="float32 or float64"):
        evaluate_gaussian_inverse_approximation(query, _affine())


def test_wrong_artifact_type_is_rejected() -> None:
    with pytest.raises(TypeError, match="GaussianInverseApproximation"):
        evaluate_gaussian_inverse_approximation(jnp.asarray(0.0), None)
