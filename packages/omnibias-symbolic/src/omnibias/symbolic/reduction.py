# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Explicit effective-model candidates and a finite parameter reduction grammar.

Fixing, coalescence and power scaling construct parameter paths. They do not
assert arbitrary asymptotic identities. Numerical reports describe their
evaluation grid; uniform error certification is a separate consumer.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]
Model = Callable[[Array, Array], Array]
Reconstruction = Callable[[float, Array], Array]


@dataclass(frozen=True)
class ParameterTerm:
    """One reconstructed parameter: offset + scale*eps**power*eta[source].

    ``source=None`` replaces eta[source] by one (fixed parameter). Sharing
    a source implements coalescence; positive powers remove amplitudes.
    Negative powers require a user-supplied desingularized reduced model.
    """

    source: int | None
    scale: float = 1.0
    power: int = 0
    offset: float = 0.0

    def __post_init__(self) -> None:
        if self.source is not None and (
            not isinstance(self.source, int) or isinstance(self.source, bool) or self.source < 0
        ):
            raise ValueError("source must be a nonnegative index")
        if not isinstance(self.power, int) or isinstance(self.power, bool):
            raise ValueError("integer power required")
        if not np.isfinite(self.scale + self.offset):
            raise ValueError("finite scale and offset required")


@dataclass(frozen=True)
class ParameterReduction:
    terms: tuple[ParameterTerm, ...]
    reduced_dimension: int

    def __post_init__(self) -> None:
        if (
            not self.terms
            or self.reduced_dimension < 0
            or any(t.source is not None and t.source >= self.reduced_dimension for t in self.terms)
        ):
            raise ValueError("terms and reduced dimension do not agree")

    def __call__(self, epsilon: float, eta: Array) -> Array:
        v = np.asarray(eta, dtype=float)
        if (
            v.shape != (self.reduced_dimension,)
            or not np.all(np.isfinite(v))
            or not np.isfinite(epsilon)
            or epsilon < 0
        ):
            raise ValueError("finite reduced parameters and nonnegative epsilon required")
        if epsilon == 0 and any(t.power < 0 for t in self.terms):
            raise ValueError("singular path needs an explicit reduced model")
        return np.asarray(
            [
                t.offset + t.scale * epsilon**t.power * (1.0 if t.source is None else v[t.source])
                for t in self.terms
            ]
        )


@dataclass(frozen=True)
class ReductionCandidate:
    full_model: Model
    reduced_model: Model
    reconstruct: Reconstruction
    name: str = "explicit effective model"


@dataclass(frozen=True)
class ReductionReport:
    epsilons: Array
    max_errors: Array
    observed_orders: Array
    scope: str = "evaluated inputs only"
    certified: bool = False


def reduction_candidate(
    full_model: Model,
    transform: ParameterReduction,
    *,
    reduced_model: Model | None = None,
    name: str = "parameter limit",
) -> ReductionCandidate:
    """Construct an analytic-at-zero limit, or accept an explicit reduced model."""
    if reduced_model is None:
        if any(t.power < 0 for t in transform.terms):
            raise ValueError("singular scaling requires explicit reduced_model")

        def reduced(eta: Array, inputs: Array) -> Array:
            return full_model(transform(0.0, eta), inputs)

        reduced_model = reduced
    return ReductionCandidate(full_model, reduced_model, transform, name)


def evaluate_reduction(
    candidate: ReductionCandidate, epsilons: Sequence[float], reduced_params: Array, inputs: Array
) -> ReductionReport:
    eps = np.asarray(epsilons, dtype=float)
    if (
        eps.ndim != 1
        or eps.size == 0
        or not np.all(np.isfinite(eps))
        or np.any(eps <= 0)
        or np.any(np.diff(eps) >= 0)
    ):
        raise ValueError("strictly decreasing positive finite epsilon ladder required")
    target = np.asarray(candidate.reduced_model(reduced_params, inputs), dtype=float)
    errors = []
    for e in eps:
        actual = np.asarray(
            candidate.full_model(candidate.reconstruct(float(e), reduced_params), inputs),
            dtype=float,
        )
        if (
            actual.shape != target.shape
            or actual.size == 0
            or not np.all(np.isfinite(actual))
            or not np.all(np.isfinite(target))
        ):
            raise ValueError("full and reduced models must return finite matching observations")
        errors.append(float(np.max(np.abs(actual - target))))
    err = np.asarray(errors)
    orders = np.full(max(0, len(eps) - 1), np.nan)
    for i in range(len(orders)):
        if err[i] > 0 and err[i + 1] > 0:
            orders[i] = np.log(err[i] / err[i + 1]) / np.log(eps[i] / eps[i + 1])
    return ReductionReport(eps, err, orders)


def boundary_direction(jacobian: Array, *, rtol: float = 1e-10) -> Array:
    """Least visible *nonzero* right-singular direction on a regular stratum.

    Zero directions are parameter redundancy, not a model-manifold boundary.
    This diagnostic proposes a direction; it does not perform a limiting proof.
    """
    j = np.asarray(jacobian, dtype=float)
    if j.ndim != 2 or min(j.shape) == 0 or not np.all(np.isfinite(j)) or rtol < 0:
        raise ValueError("finite nonempty Jacobian required")
    _, s, vh = np.linalg.svd(j, full_matrices=False)
    rank = int(np.count_nonzero(s > rtol * s[0]))
    if rank == 0:
        raise ValueError("no visible direction")
    direction: Array = np.array(vh[rank - 1], dtype=np.float64, copy=True)
    return direction


__all__ = [
    "ParameterReduction",
    "ParameterTerm",
    "ReductionCandidate",
    "ReductionReport",
    "boundary_direction",
    "evaluate_reduction",
    "reduction_candidate",
]
