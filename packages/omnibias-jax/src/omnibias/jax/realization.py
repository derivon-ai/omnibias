# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Live input/weight realization jets and the six activation operator roles.

All coefficient arrays use the shared multi-index ordering and contain Taylor
coefficients, not raw derivatives. Only static shapes and descriptions are
inspected on the host; returned arrays keep their parameter graphs intact.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

from omnibias.core.multi_index import (
    index_position,
    multi_indices,
    multiply_table,
    num_multi_indices,
)
from omnibias.core.realization import LayerSpec, ObservationSpec, ParameterLayout, RealizationSpec
from omnibias.core.spec import ActivationSpec
from omnibias.jax.activations import get_activation
from omnibias.jax.jet import _sigma_tower
from omnibias.jax.jet_mv import compose_jet_mv

import jax
import jax.numpy as jnp
from jax import Array


def affine_joint_jet_mv(
    input_jet: Array, weight_jet: Array, bias_jet: Array | None = None,
    *, dim: int, order: int,
) -> Array:
    """Cauchy product for a layer whose input, weights and bias all vary."""
    count = num_multi_indices(dim, order)
    if input_jet.shape[0] != count or weight_jet.shape[0] != count:
        raise ValueError("input and weight jets must have the specified coefficient count")
    if weight_jet.ndim != 3 or input_jet.shape[-1] != weight_jet.shape[-1]:
        raise ValueError("weight jet must be (M, out, in) with matching input width")
    sample = jnp.tensordot(input_jet[0], weight_jet[0], axes=([-1], [-1]))
    rows = [jnp.zeros_like(sample) for _ in range(count)]
    for out, a, b in multiply_table(dim, order):
        rows[out] = rows[out] + jnp.tensordot(input_jet[a], weight_jet[b], axes=([-1], [-1]))
    if bias_jet is not None:
        if bias_jet.shape != (count, weight_jet.shape[1]):
            raise ValueError("bias jet must be (M, out)")
        rows = [row + bias_jet[i] for i, row in enumerate(rows)]
    return jnp.stack(rows, axis=0)


def affine_joint_jet(input_jet: Array, weight_jet: Array, bias_jet: Array | None = None) -> Array:
    """Directional restriction of :func:`affine_joint_jet_mv`."""
    return affine_joint_jet_mv(input_jet, weight_jet, bias_jet, dim=1, order=input_jet.shape[0] - 1)


def _seed(value: Array, basis: Array | None, dim: int, order: int) -> Array:
    indices = multi_indices(dim, order)
    if basis is not None and basis.shape != value.shape + (dim,):
        raise ValueError("basis must have shape value.shape + (jet_dimension,)")
    rows = []
    for index in indices:
        degree = sum(index)
        if degree == 0:
            rows.append(value)
        elif degree == 1 and basis is not None:
            rows.append(basis[..., index.index(1)])
        else:
            rows.append(jnp.zeros_like(value))
    return jnp.stack(rows, axis=0)


def _block(jet: Array, layout: ParameterLayout, name: str) -> Array:
    block = layout.block(name)
    return jet[:, block.slice].reshape((jet.shape[0],) + block.shape)


def _activation_jet(jet: Array, name: str | ActivationSpec[Array], dim: int, order: int, offset: int = 0) -> Array:
    spec = get_activation(name) if isinstance(name, str) else name
    tower = _sigma_tower(spec, jet[0], order + offset)[offset:]
    return compose_jet_mv(jet, tower, dim, order)


def _antiderivative_jet(jet: Array, name: str | ActivationSpec[Array], dim: int, order: int) -> Array:
    spec = get_activation(name) if isinstance(name, str) else name
    if spec.integral is None:
        raise NotImplementedError(f"activation {name!r} has no antiderivative kernel")
    rows = [spec.integral(jet[0])]
    if order:
        derivatives = _sigma_tower(spec, jet[0], order - 1)
        rows.extend(derivatives[k] for k in range(order))
    return compose_jet_mv(jet, jnp.stack(rows, axis=0), dim, order)


def _shift(jet: Array, shift: Array) -> Array:
    # The coefficient axis remains first; endpoint shifts broadcast over batches.
    return jnp.stack([jet[i] + shift[i] for i in range(jet.shape[0])], axis=0)


def _multiply(a: Array, b: Array, dim: int, order: int) -> Array:
    rows = [jnp.zeros_like(a[0] * b[0]) for _ in range(a.shape[0])]
    for out, i, j in multiply_table(dim, order):
        rows[out] = rows[out] + a[i] * b[j]
    return jnp.stack(rows, axis=0)


def operator_parameter_jet(
    jet: Array, biases: Array, signs: Array, activation: str | ActivationSpec[Array],
    op: str, *, dim: int, order: int, derivative_order: int = 0,
    normalize_integral: bool = False, integral_small_width: float = 0.0,
    use_small_width_taylor: bool = False,
) -> Array:
    """All six OperatorBlock roles in the original bias/sign coordinates.

    Bias/sign jets have shape ``(M, channels, K)``. Integral biases store
    center/raw-softplus-width. The optional midpoint branch reproduces the
    existing block's explicitly approximate small-width policy, including its
    branch derivatives; zero threshold selects the antiderivative window.
    """
    count = num_multi_indices(dim, order)
    if biases.ndim != 3 or biases.shape[0] != count or signs.shape != biases.shape:
        raise ValueError("bias/sign jets must have matching (M, channels, K) shapes")
    if jet.shape[0] != count or jet.shape[-1] != biases.shape[1]:
        raise ValueError("input jet coefficient count/channels must match bias jets")
    if op in {"grad", "laplacian", "derivative"}:
        offset = {"grad": 1, "laplacian": 2, "derivative": derivative_order}[op]
        if offset < 1:
            raise ValueError("derivative_order must be positive")
        return _activation_jet(_shift(jet, biases.mean(axis=-1)), activation, dim, order, offset)
    if op not in {"identity", "band", "integral"}:
        raise ValueError(f"unknown operator role {op!r}")
    arity = biases.shape[-1]
    if arity != (1 if op == "identity" else 2):
        raise ValueError("operator bias arity does not match the role")
    endpoints = biases
    width: Array | None = None
    if op == "integral":
        center = biases[..., 0]
        width = _activation_jet(biases[..., 1], "softplus", dim, order)
        endpoints = jnp.stack((center - 0.5 * width, center + 0.5 * width), axis=-1)
    pieces = []
    for k in range(arity):
        shifted = _shift(jet, endpoints[..., k])
        value = (_antiderivative_jet(shifted, activation, dim, order) if op == "integral"
                 else _activation_jet(shifted, activation, dim, order))
        pieces.append(_multiply(value, signs[..., k], dim, order))
    result = pieces[0]
    for piece in pieces[1:]:
        result = result + piece
    if width is not None:
        if normalize_integral:
            # Match the existing forward denominator floor without evaluating 1/0.
            floor = _seed(jnp.full_like(width[0], 1e-12), None, dim, order)
            denominator = jnp.where(width[0] > 1e-12, width, floor)
            reciprocal = jnp.stack([((-1) ** n) * math.factorial(n) / denominator[0] ** (n + 1)
                                      for n in range(order + 1)], axis=0)
            result = _multiply(result, compose_jet_mv(denominator, reciprocal, dim, order), dim, order)
        if use_small_width_taylor and integral_small_width > 0:
            midpoint = _activation_jet(_shift(jet, biases[..., 0]), activation, dim, order)
            approx = midpoint if normalize_integral else _multiply(midpoint, width, dim, order)
            result = jnp.where(width[0] < integral_small_width, approx, result)
    return result


def operator_jet(
    jet: Array, layer: LayerSpec, *, dim: int, order: int,
    windows: tuple[Array, Array] | None = None,
) -> Array:
    """Apply one static activation-level role, including live window endpoints."""
    name = layer.activation
    if name is None:
        return jet
    if layer.op not in {"band", "integral"}:
        return _activation_jet(jet, name, dim, order, layer.activation_order)
    if windows is None:
        raise ValueError("window roles require endpoint jets")
    lo, hi = windows
    if layer.window_parameterization == "center_softplus_width":
        width = _activation_jet(hi, "softplus", dim, order)
        lo, hi = lo - 0.5 * width, lo + 0.5 * width
    lower, upper = _shift(jet, lo), _shift(jet, hi)
    if layer.op == "band":
        return _activation_jet(upper, name, dim, order) - _activation_jet(lower, name, dim, order)
    return _antiderivative_jet(upper, name, dim, order) - _antiderivative_jet(lower, name, dim, order)


def realization_jet(
    input_jet: Array, parameter_jet: Array, spec: RealizationSpec,
    *, dim: int, order: int, max_coefficients: int = 1_000_000,
) -> Array:
    """Propagate joint input/parameter jets through a described dense network."""
    count = num_multi_indices(dim, order)
    if parameter_jet.shape != (count, spec.layout.size):
        raise ValueError("parameter jet does not match the layout/coefficient count")
    if input_jet.shape[0] != count or input_jet.shape[-1] != spec.input_dim:
        raise ValueError("input jet does not match the realization")
    batch = math.prod(input_jet.shape[1:-1])
    peak = max(layer.out_features for layer in spec.layers)
    needed = count * (spec.layout.size + batch * (spec.input_dim + peak))
    if max_coefficients < 1 or needed > max_coefficients:
        raise ValueError(f"joint jet needs {needed} coefficients; budget is {max_coefficients}")
    jet = input_jet
    for i, layer in enumerate(spec.layers):
        weight = _block(parameter_jet, spec.layout, layer.weight_name or f"W{i}")
        bias = None if layer.bias_name is None else _block(parameter_jet, spec.layout, layer.bias_name)
        jet = affine_joint_jet_mv(jet, weight, bias, dim=dim, order=order)
        windows = None
        if layer.window_names is not None:
            windows = (_block(parameter_jet, spec.layout, layer.window_names[0]),
                       _block(parameter_jet, spec.layout, layer.window_names[1]))
        jet = operator_jet(jet, layer, dim=dim, order=order, windows=windows)
    return jet


def parameter_jet(
    x: Array, theta: Array, spec: RealizationSpec, basis: Array, order: int,
    *, input_basis: Array | None = None, max_coefficients: int = 1_000_000,
) -> Array:
    """Live Taylor jet along explicit P-by-q parameter directions.

    ``input_basis`` is ``x.shape + (q,)`` (a ``(D,q)`` basis also broadcasts
    across input batches). Use zero parameter directions for pure input jets.
    The basis is explicit so a full parameter tensor is never allocated implicitly.
    """
    if theta.ndim != 1 or theta.shape[0] != spec.layout.size:
        raise ValueError("theta must be the flat parameter vector described by layout")
    if basis.ndim != 2 or basis.shape[0] != theta.shape[0] or basis.shape[1] < 1:
        raise ValueError("basis must have shape (P, q), q >= 1")
    dim = basis.shape[1]
    count = num_multi_indices(dim, order)
    if count * (theta.size + x.size) > max_coefficients:
        raise ValueError("joint-jet coefficient budget exceeded before seeding")
    if input_basis is not None:
        input_basis = jnp.broadcast_to(input_basis, x.shape + (dim,))
    return realization_jet(_seed(x, input_basis, dim, order), _seed(theta, basis, dim, order),
                           spec, dim=dim, order=order, max_coefficients=max_coefficients)


def evaluate(x: Array, theta: Array, spec: RealizationSpec) -> Array:
    """Evaluate the exact described architecture with live parameters."""
    if theta.ndim != 1 or theta.shape[0] != spec.layout.size:
        raise ValueError("theta does not match realization layout")
    needed = spec.layout.size + math.prod(x.shape[:-1]) * (spec.input_dim + max(layer.out_features for layer in spec.layers))
    return realization_jet(x[None, ...], theta[None, ...], spec, dim=1, order=0,
                           max_coefficients=needed)[0]


def observe(
    theta: Array, spec: RealizationSpec, observation: ObservationSpec,
    *, points: Array | None = None, weights: Array | None = None,
    residual: Callable[[Array, Callable[[Array], Array]], Array] | None = None,
    coefficient_map: Callable[[Array], Array] | None = None,
) -> Array:
    """Observe values/derivatives/quadrature, or an explicitly supplied operator.

    These are observations only. A declared domain or coefficient-map label is
    not independently checked as a global equality proof by this numerical API.
    """
    if observation.kind == "coefficients":
        if coefficient_map is None:
            raise ValueError("coefficient observations require an explicit coefficient map")
        return coefficient_map(theta).reshape(-1)
    coords = jnp.asarray(observation.points, dtype=theta.dtype) if points is None else points
    if coords.ndim != 2 or coords.shape[1] != spec.input_dim:
        raise ValueError("observation points must be (N, input_dim)")
    if observation.kind == "residuals":
        if residual is None:
            raise ValueError("residual observations require an explicit residual operator")
        return residual(coords, lambda x: evaluate(x, theta, spec)).reshape(-1)
    if observation.kind == "derivatives":
        indices = observation.derivative_indices
        if not indices or any(len(a) != spec.input_dim for a in indices):
            raise ValueError("derivative observations require input multi-indices")
        order = max(map(sum, indices))
        directions = jnp.zeros((theta.shape[0], spec.input_dim), dtype=theta.dtype)
        ibasis = jnp.eye(spec.input_dim, dtype=theta.dtype)
        jet = parameter_jet(coords, theta, spec, directions, order, input_basis=ibasis)
        positions = index_position(spec.input_dim, order)
        return jnp.stack([jet[positions[a]] * math.prod(math.factorial(i) for i in a) for a in indices], axis=0).reshape(-1)
    values = evaluate(coords, theta, spec)
    if observation.kind == "integrals":
        if observation.integral_kind == "activation_window":
            if not any(layer.op == "integral" for layer in spec.layers):
                raise ValueError("activation_window observations require an integral-role realization")
            if weights is not None or observation.weights:
                raise ValueError("activation windows do not apply quadrature weights")
            return values.reshape(-1)
        # Domain/measure observations are the declared weighted numerical rule.
        w = jnp.asarray(observation.weights, dtype=theta.dtype) if weights is None else weights
        if w.shape != (coords.shape[0],):
            raise ValueError("quadrature observations require one weight per point")
        return (values * w[:, None]).sum(axis=0).reshape(-1)
    return values.reshape(-1)


@dataclass(frozen=True)
class Realization:
    spec: RealizationSpec

    def evaluate(self, x: Array, theta: Array) -> Array:
        return evaluate(x, theta, self.spec)

    def jvp(self, x: Array, theta: Array, direction: Array) -> Array:
        """Closed-form directional parameter jet, with live coefficients."""
        return parameter_jet(x, theta, self.spec, direction.reshape(-1, 1), 1)[1]

    def vjp(self, x: Array, theta: Array, cotangent: Array) -> Array:
        """First-order backend reverse-mode adjoint of the live evaluation."""
        _, pullback = jax.vjp(lambda p: evaluate(x, p, self.spec), theta)
        return cast(Array, pullback(cotangent)[0])


__all__ = ["Realization", "affine_joint_jet", "affine_joint_jet_mv", "evaluate", "observe",
           "operator_jet", "operator_parameter_jet", "parameter_jet", "realization_jet"]
