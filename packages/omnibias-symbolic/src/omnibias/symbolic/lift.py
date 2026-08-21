# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Soft residual / SINDy hit → exact-``Q`` identity, or ``None``.

Rounding a float coefficient is not a certificate. The snapped relation is
accepted only when the residual is identically zero over the supplied exact
design. A miss stays ``None`` — there is no “almost identity.”
"""

from __future__ import annotations

from collections.abc import Sequence
from fractions import Fraction
from typing import Any

from omnibias.core.proof.discovery import DiscoveredEquation
from omnibias.core.proof.lift import as_fraction, residual_identically_zero
from omnibias.symbolic.discovery import SparseEquation

Number = int | Fraction | float

_EQUATION_KINDS = frozenset(
    {
        "recurrence",
        "ore",
        "polynomial_identity",
        "pde_span",
        "map_witness",
        "graph_witness",
        "conservation",
        "sos_template",
        "forbidden_minor",
    }
)


def snap_sparse_equation(
    equation: SparseEquation,
    design: Sequence[Sequence[Number]],
    target: Sequence[Number],
    *,
    denom_bound: int = 32,
    kind: str = "pde_span",
) -> DiscoveredEquation | None:
    """Round ``equation`` to bounded-denominator rationals; accept iff residual is 0."""

    snapped = [
        as_fraction(float(coef), denom_bound=denom_bound) for coef in equation.coefficients
    ]
    intercept = as_fraction(float(equation.intercept), denom_bound=denom_bound)
    if not residual_identically_zero(
        design,
        snapped,
        target,
        intercept=intercept,
        denom_bound=denom_bound,
    ):
        return None
    names = list(equation.term_names)
    pieces: list[str] = []
    coeffs: list[str] = []
    if intercept != 0:
        pieces.append(str(intercept))
        coeffs.append(str(intercept))
    for name, coef in zip(names, snapped, strict=False):
        coeffs.append(str(coef))
        if coef == 0:
            continue
        pieces.append(f"({coef})*{name}")
    pretty = " + ".join(pieces) if pieces else "0"
    equation_kind = kind if kind in _EQUATION_KINDS else "pde_span"
    return DiscoveredEquation(
        kind=equation_kind,  # type: ignore[arg-type]
        pretty=f"target = {pretty}",
        coefficients=tuple(coeffs),
    )


def planted_heat_rational(
    *,
    n: int = 8,
    diffusivity: Fraction = Fraction(1, 8),
) -> tuple[list[list[Fraction]], list[Fraction], list[str]]:
    """Integer-grid heat samples: ``u=x^2``, ``u_t = D u_xx`` with ``u_xx=2``."""

    design: list[list[Fraction]] = []
    target: list[Fraction] = []
    for x in range(n):
        design.append([Fraction(x * x), Fraction(2 * x), Fraction(2)])
        target.append(diffusivity * 2)
    return design, target, ["u", "u_x", "u_xx"]


def sparse_from_coeffs(
    coefficients: Sequence[float],
    names: Sequence[str],
    *,
    intercept: float = 0.0,
) -> SparseEquation:
    """Build a :class:`SparseEquation` from explicit float coefficients."""

    import numpy as np

    coeff = np.asarray(list(coefficients), dtype=float)
    return SparseEquation(
        term_names=tuple(names),
        coefficients=coeff,
        intercept=float(intercept),
        alpha=0.0,
        threshold=0.0,
        active_mask=np.abs(coeff) > 0,
    )


def snap_payload(equation: DiscoveredEquation | None) -> dict[str, Any] | None:
    return None if equation is None else equation.as_dict()


__all__ = [
    "planted_heat_rational",
    "snap_payload",
    "snap_sparse_equation",
    "sparse_from_coeffs",
]
