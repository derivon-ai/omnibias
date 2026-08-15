# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Weak-form VPINN assembly (JAX twin of torch; theory 02-04)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import jax.numpy as jnp
from jax import Array, grad
from omnibias.fields._core.quadrature import QuadratureSpec
from omnibias.fields.weak._core import (
    TestFunctionSpace,
    WeakForm,
    boundary_bound,
    eval_test,
    exact_moment,
    poly_eval,
)

PathName = Literal["exact", "quadrature"]


@dataclass(frozen=True)
class ResidualTerm:
    name: str
    path: PathName
    value: Array


def weak_residual(
    field: Callable[[Array], Array] | tuple[float, ...],
    space: TestFunctionSpace,
    *,
    operator: WeakForm,
    source: Callable[[Array], Array] | tuple[float, ...] | None = None,
    quadrature: QuadratureSpec | None = None,
) -> tuple[Array, tuple[ResidualTerm, ...]]:
    if space.window is None:
        raise ValueError("weak_residual needs a box window")
    src = operator.source if source is None else source
    values: list[Array] = []
    terms: list[ResidualTerm] = []
    for i in range(space.size):
        stiffness, stiff_path = _stiffness_term(field, space, i, operator.diffusion, quadrature)
        load, load_path = _load_term(src, space, i, quadrature)
        r = stiffness - load
        values.append(r)
        terms.append(ResidualTerm(f"stiffness[{i}]", stiff_path, stiffness))
        terms.append(ResidualTerm(f"load[{i}]", load_path, load))
    return jnp.stack(values), tuple(terms)


def _stiffness_term(
    field: Callable[[Array], Array] | tuple[float, ...],
    space: TestFunctionSpace,
    index: int,
    diffusion: tuple[float, ...],
    quadrature: QuadratureSpec | None,
) -> tuple[Array, PathName]:
    if isinstance(field, tuple):
        du = tuple(float(k) * float(field[k]) for k in range(1, len(field)))
        prod = _poly_mul(diffusion, du if du else (0.0,))
        acc = 0.0
        bumped = TestFunctionSpace(
            bank=space.bank,
            orders=tuple(space.order_at(index) + 1 for _ in range(space.size)),
            base=space.base,
            window=space.window,
        )
        alpha = space.scale_at(index)
        for j, c in enumerate(prod):
            acc += float(c) * alpha * exact_moment(bumped, j, index)
        return jnp.asarray(acc, dtype=jnp.float64), "exact"
    if quadrature is None:
        raise ValueError("callable field needs a QuadratureSpec for the coefficient factor")
    xs = jnp.asarray(quadrature.nodes[:, 0], dtype=jnp.float64)
    ws = jnp.asarray(quadrature.weights, dtype=jnp.float64)

    def _scalar_at(x_s: Array) -> Array:
        return field(x_s.reshape((1, 1))).reshape(())

    du_vals = jnp.stack([grad(_scalar_at)(x) for x in xs])
    a = jnp.asarray([poly_eval(diffusion, float(x)) for x in xs.tolist()], dtype=jnp.float64)
    vp = jnp.asarray(
        [eval_test(space, index, float(x), deriv=1) for x in xs.tolist()],
        dtype=jnp.float64,
    )
    return jnp.sum(ws * a * du_vals * vp), "quadrature"


def _load_term(
    source: Callable[[Array], Array] | tuple[float, ...] | None,
    space: TestFunctionSpace,
    index: int,
    quadrature: QuadratureSpec | None,
) -> tuple[Array, PathName]:
    if source is None:
        return jnp.asarray(0.0, dtype=jnp.float64), "exact"
    if isinstance(source, tuple):
        acc = 0.0
        for j, c in enumerate(source):
            acc += float(c) * exact_moment(space, j, index)
        return jnp.asarray(acc, dtype=jnp.float64), "exact"
    if quadrature is None:
        raise ValueError("callable source needs a QuadratureSpec")
    xs = jnp.asarray(quadrature.nodes[:, 0], dtype=jnp.float64)
    ws = jnp.asarray(quadrature.weights, dtype=jnp.float64)
    f = source(xs.reshape((-1, 1))).reshape(-1)
    v = jnp.asarray(
        [eval_test(space, index, float(x), deriv=0) for x in xs.tolist()],
        dtype=jnp.float64,
    )
    return jnp.sum(ws * f * v), "quadrature"


def _poly_mul(a: tuple[float, ...], b: tuple[float, ...]) -> tuple[float, ...]:
    out = [0.0] * (len(a) + len(b) - 1)
    for i, ai in enumerate(a):
        for j, bj in enumerate(b):
            out[i + j] += float(ai) * float(bj)
    return tuple(out)


def weak_loss(
    field: Callable[[Array], Array] | tuple[float, ...],
    space: TestFunctionSpace,
    *,
    operator: WeakForm,
    source: Callable[[Array], Array] | tuple[float, ...] | None = None,
    quadrature: QuadratureSpec | None = None,
    include_boundary_bound: bool = True,
    deriv_bound: float = 1.0,
) -> Array:
    residual, _terms = weak_residual(
        field, space, operator=operator, source=source, quadrature=quadrature
    )
    loss = jnp.sum(residual * residual)
    if include_boundary_bound:
        bound = boundary_bound(space, deriv_bound=deriv_bound)
        loss = loss + jnp.asarray(float(bound.hi), dtype=loss.dtype)
    return loss


__all__ = [
    "ResidualTerm",
    "weak_loss",
    "weak_residual",
]
