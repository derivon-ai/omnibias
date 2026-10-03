# SPDX-License-Identifier: AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial
# Copyright (C) 2026 Derivon
"""Track C3: derive one concrete ``hCauchy`` input for ``Hilbert16LNCell.lean``."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

from omnibias.core.verified.interval import Interval
from omnibias.core.verified.log_noetherian import log_chart_derivative_bound
from omnibias.dynamics.ln_passage import canonical_L, quadratic_relation_residual

__all__ = [
    "DerivedLogChartCauchyReport",
    "derive_quadratic_ln_cell_cauchy_bound",
]


@dataclass(frozen=True)
class DerivedLogChartCauchyReport:
    lambda1: float
    sep: float
    L: float
    delta: Fraction
    sup_bound: Interval
    second_derivative_bound: Interval
    quadratic_residual: float
    analytic_hypothesis_derived: bool

    def to_payload(self) -> dict[str, object]:
        return {
            "lambda1": self.lambda1,
            "sep": self.sep,
            "L": self.L,
            "delta": [self.delta.numerator, self.delta.denominator],
            "sup_bound": [self.sup_bound.lo, self.sup_bound.hi],
            "second_derivative_bound": [self.second_derivative_bound.lo, self.second_derivative_bound.hi],
            "quadratic_residual": self.quadratic_residual,
            "analytic_hypothesis_derived": self.analytic_hypothesis_derived,
            "theorem_prover_verified": False,
            "continuum_pde_claim": False,
        }


def derive_quadratic_ln_cell_cauchy_bound(
    *,
    lambda1: float = -3.0,
    sep: float = 0.25,
    sup_hi: float = 1.0,
    delta: Fraction = Fraction(1, 8),
) -> DerivedLogChartCauchyReport:
    """Derive a Cauchy bound for order two on the declared quadratic LN model."""
    L = canonical_L(lambda1, sep)
    residual = quadratic_relation_residual(L, lambda1, sep)
    if abs(residual) > 1e-12:
        raise ValueError("quadratic LN model relation failed replay")
    sup_bound = Interval(0.0, sup_hi)
    second = log_chart_derivative_bound(sup_bound, delta, 2)
    return DerivedLogChartCauchyReport(
        lambda1=lambda1,
        sep=sep,
        L=L,
        delta=delta,
        sup_bound=sup_bound,
        second_derivative_bound=second,
        quadratic_residual=residual,
        analytic_hypothesis_derived=second.hi > 0.0,
    )
