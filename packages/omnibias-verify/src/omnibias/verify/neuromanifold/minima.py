# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Whole-box hyperdual minima on slices and exact affine quotient families."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from fractions import Fraction
from typing import Literal

from omnibias.core.proof.certificate import Cert, interval_certificate
from omnibias.core.verified.eig_operator import interval_ldlt_pivots
from omnibias.core.verified.interval import Interval
from omnibias.geometry.neuromanifold import RegularQuotientChart
from omnibias.verify._core.newton import krawczyk_image, krawczyk_unique
from omnibias.verify._core.param_loss import HyperDual, _sigmoid, _tanh

Box = tuple[Interval, ...]
Objective = Callable[[tuple[HyperDual, ...]], HyperDual]
_ZERO = Interval.point(0.0)
_ONE = Interval.point(1.0)

# Public adapters reuse the existing interval-hyperdual activation propagation.
dual_sigmoid = _sigmoid
dual_tanh = _tanh


@dataclass(frozen=True)
class IntervalObjective:
    """An expression evaluated in interval hyperdual arithmetic over the whole box.

    source_id identifies the supplied expression. For callbacks, semantic
    identity of that expression and its intended scientific model remains an
    explicit dependency; center Taylor coefficients alone are never accepted.
    """

    expression: Objective
    source_id: str
    parameter_names: tuple[str, ...]

    def __post_init__(self) -> None:
        if (
            not self.source_id
            or not self.parameter_names
            or len(set(self.parameter_names)) != len(self.parameter_names)
        ):
            raise ValueError("source identity and unique parameter names are required")

    def _evaluate(self, box: Box, first: int = -1, second: int = -1) -> HyperDual:
        if len(box) != len(self.parameter_names):
            raise ValueError("box does not match parameter ordering")
        values = tuple(
            HyperDual(x, _ONE if i == first else _ZERO, _ONE if i == second else _ZERO, _ZERO)
            for i, x in enumerate(box)
        )
        result = self.expression(values)
        if not isinstance(result, HyperDual):
            raise TypeError("objective must preserve interval hyperdual values")
        return result

    def value(self, box: Box) -> Interval:
        return self._evaluate(box).val

    def gradient(self, box: Box) -> list[Interval]:
        return [self._evaluate(box, i).d1 for i in range(len(box))]

    def hessian(self, box: Box) -> list[list[Interval]]:
        n = len(box)
        result = [[_ZERO for _ in range(n)] for _ in range(n)]
        for i in range(n):
            for j in range(i, n):
                result[i][j] = result[j][i] = self._evaluate(box, i, j).d12
        return result


@dataclass(frozen=True)
class MinimumCertificate:
    status: Literal["proved", "inconclusive"]
    claim: Literal["slice_minimum", "quotient_minimum", "morse_bott_minimum"]
    stationary_box: Box | None
    search_box: Box
    hessian: tuple[tuple[Interval, ...], ...]
    pivots: tuple[Interval, ...] | None
    certificate: Cert
    dependencies: tuple[str, ...]


def certify_slice_minimum(objective: IntervalObjective, box: Box) -> MinimumCertificate:
    """Enclose the actual stationary point, and prove whole-box positive curvature.

    This is a minimum in the supplied coordinates. No symmetry or neighborhood
    coverage is inferred from an arbitrary restricted subspace.
    """
    if not box or any(x.lo >= x.hi for x in box):
        raise ValueError("nonempty box with strict positive widths required")
    hessian = objective.hessian(box)
    pivots = interval_ldlt_pivots(hessian)
    image = krawczyk_image(objective.gradient, objective.hessian, box)
    unique = image is not None and krawczyk_unique(image, box)
    positive = pivots is not None and all(p.lo > 0 for p in pivots)
    proved = unique and positive
    root = (
        tuple(Interval(max(x.lo, y.lo), min(x.hi, y.hi)) for x, y in zip(box, image, strict=True))
        if proved and image is not None
        else None
    )
    meta = {
        "kind": "slice_minimum",
        "source_id": objective.source_id,
        "parameter_names": list(objective.parameter_names),
        "search_box": [[x.lo, x.hi] for x in box],
        "stationary_box": None if root is None else [[x.lo, x.hi] for x in root],
        "hessian": [[[x.lo, x.hi] for x in row] for row in hessian],
        "krawczyk_image": None if image is None else [[x.lo, x.hi] for x in image],
        "gradient_at_center": [
            [x.lo, x.hi] for x in objective.gradient(tuple(Interval.point(x.mid) for x in box))
        ],
        "status": "proved" if proved else "inconclusive",
    }
    certificate = interval_certificate(
        "stationary inclusion and uniformly positive reduced Hessian",
        objective.value(box),
        meta=meta,
    )
    return MinimumCertificate(
        "proved" if proved else "inconclusive",
        "slice_minimum",
        root,
        box,
        tuple(tuple(row) for row in hessian),
        None if pivots is None else tuple(pivots),
        certificate,
        (
            "supplied hyperdual expression describes the intended objective",
            "callback preserves sound smooth hyperdual arithmetic and derivative seeds",
            "Krawczyk existence theorem",
            "interval LDL positive-definiteness theorem",
        ),
    )


def verify_affine_chart(chart: RegularQuotientChart) -> bool:
    """Replay all source factorization, right-inverse, and complete-kernel relations."""
    a, p, r, kernels = chart.source, chart.independent_columns, chart.projection, chart.kernel
    if not a or not a[0] or not p:
        return False
    n, d = len(a[0]), len(p)
    if (
        len(set(p)) != d
        or any(j < 0 or j >= n for j in p)
        or len(r) != d
        or any(len(row) != n for row in a)
        or any(len(row) != n for row in r)
    ):
        return False
    for i in range(d):
        for j in range(d):
            if r[i][p[j]] != Fraction(int(i == j)):
                return False
    for row in a:
        for j in range(n):
            if sum((row[p[k]] * r[k][j] for k in range(d)), Fraction()) != row[j]:
                return False
    # Independence of selected image columns, in exact rational elimination.
    from omnibias.geometry.neuromanifold.geometry import _rref

    if len(_rref([[row[j] for j in p] for row in a])[1]) != d:
        return False
    free = [j for j in range(n) if j not in p]
    expected = []
    for j in free:
        vector = [Fraction(0)] * n
        vector[j] = Fraction(1)
        for i, pivot in enumerate(p):
            vector[pivot] = -r[i][j]
        expected.append(tuple(vector))
    return tuple(expected) == kernels


def certify_quotient_minimum(
    chart: RegularQuotientChart,
    reduced_objective: IntervalObjective,
    box: Box,
    *,
    morse_bott: bool = False,
) -> MinimumCertificate:
    """Certify the objective DEFINED by L(theta)=ell(chart.projection @ theta).

    Exact affine factorization gives invariance, a global slice, constant rank,
    and the complete critical family E*q_star + ker(A) over the certified reduced
    box. Other critical points outside that box are not excluded. This API never assumes
    an independently supplied full loss factors through a sampled Jacobian.
    With morse_bott=True this explicit critical family earns that stronger claim.
    """
    if not verify_affine_chart(chart) or len(box) != chart.dim:
        raise ValueError("invalid exact affine quotient chart or coordinate dimension")
    result = certify_slice_minimum(reduced_objective, box)
    claim: Literal["slice_minimum", "quotient_minimum", "morse_bott_minimum"] = (
        "morse_bott_minimum" if morse_bott else "quotient_minimum"
    )
    certificate = interval_certificate(
        "minimum of ell(R theta) on an exact affine quotient",
        reduced_objective.value(box),
        meta={
            "kind": claim,
            "source": [[str(v) for v in row] for row in chart.source],
            "projection": [[str(v) for v in row] for row in chart.projection],
            "kernel": [[str(v) for v in row] for row in chart.kernel],
            "independent_columns": list(chart.independent_columns),
            "observation_scope": chart.observation_scope,
            "minimum_scope": "preimage of the certified reduced search box under projection",
            "reduced_certificate": result.certificate,
            "status": result.status,
        },
    )
    return MinimumCertificate(
        result.status,
        claim,
        result.stationary_box,
        result.search_box,
        result.hessian,
        result.pivots,
        certificate,
        result.dependencies,
    )


__all__ = [
    "IntervalObjective",
    "MinimumCertificate",
    "certify_quotient_minimum",
    "certify_slice_minimum",
    "dual_sigmoid",
    "dual_tanh",
    "verify_affine_chart",
]
