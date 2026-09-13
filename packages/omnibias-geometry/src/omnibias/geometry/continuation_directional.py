# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Explicit directional residual callbacks for the finite continuation solvers.

No derivative is estimated by differences. Dense correctors materialize only
requested derivative tensors, subject to a declared coefficient budget.
"""

from __future__ import annotations

from collections.abc import Callable
from itertools import product

import numpy as np

from .continuation import Array, ImplicitFamily, Map

First = Callable[[Array, Array], Array]
Second = Callable[[Array, Array, Array], Array]
Third = Callable[[Array, Array, Array, Array], Array]


def directional_residual_family(
    value: Map,
    first: First,
    *,
    state_dimension: int,
    second: Second | None = None,
    third: Third | None = None,
    name: str = "directional residual family",
    max_coefficients: int = 1_000_000,
) -> ImplicitFamily:
    """Adapt D F[v], D²F[u,v], D³F[u,v,w] without hidden autodiff.

    Points and directions have length ``state_dimension+1``; the last axis is
    the continuation parameter. Each callback returns ``state_dimension``
    residual coordinates. Derivatives are unnormalized, rather than Taylor
    coefficients. The callbacks may contract closed-form parameter/input jets.

    The existing Newton and event correctors need dense tensors. Their total
    possible coefficient count is checked before allocating a basis or calling
    the model. The adapter does not imply a matrix-free event solver or validate
    the mathematical correctness of user-supplied derivatives.
    """
    if (
        not isinstance(state_dimension, int)
        or isinstance(state_dimension, bool)
        or state_dimension < 1
        or not isinstance(max_coefficients, int)
        or isinstance(max_coefficients, bool)
        or max_coefficients < 1
    ):
        raise ValueError("positive integer state dimension and coefficient budget required")
    if third is not None and second is None:
        raise ValueError("third derivatives require a second derivative callback")
    n, p = state_dimension, state_dimension + 1
    orders = (1,) + ((2,) if second is not None else ()) + ((3,) if third is not None else ())
    if sum(n * p**order for order in orders) > max_coefficients:
        raise ValueError("dense continuation derivative coefficient budget exceeded")
    basis = np.eye(p, dtype=float)

    def point(y: Array) -> Array:
        out = np.asarray(y, dtype=float)
        if out.shape != (p,) or not np.isfinite(out).all():
            raise ValueError("finite state-plus-parameter vector required")
        return out

    def residual(raw: Array) -> Array:
        out = np.asarray(raw, dtype=float)
        if out.shape != (n,) or not np.isfinite(out).all():
            raise ValueError("callback must return a finite residual vector")
        return out

    def evaluate(y: Array) -> Array:
        return residual(value(point(y)))

    def jacobian(y: Array) -> Array:
        center = point(y)
        return np.stack([residual(first(center, direction)) for direction in basis], axis=1)

    def hessian(y: Array) -> Array:
        assert second is not None
        center = point(y)
        out = np.empty((n, p, p), dtype=float)
        for i, j in product(range(p), repeat=2):
            out[:, i, j] = residual(second(center, basis[i], basis[j]))
        return out

    def third_tensor(y: Array) -> Array:
        assert third is not None
        center = point(y)
        out = np.empty((n, p, p, p), dtype=float)
        for i, j, k in product(range(p), repeat=3):
            out[:, i, j, k] = residual(third(center, basis[i], basis[j], basis[k]))
        return out

    return ImplicitFamily(
        evaluate,
        jacobian,
        hessian if second is not None else None,
        third_tensor if third is not None else None,
        name,
    )


__all__ = ["directional_residual_family"]
