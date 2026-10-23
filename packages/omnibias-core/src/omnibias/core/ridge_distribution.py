# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Exact constant-coefficient calculus for planar distributional ridges.

The profile basis is ``T_j(z) = z_+**j/j!`` for j>=1, ``T_0=H``, and
``T_(-k-1)=delta^(k)``. Distributionally, ``D T_j = T_(j-1)`` for EVERY
integer j. This makes a fixed linear PDE's action a finite exact shift and
matrix multiplication. A Dirac is never represented by an ordinary value.

For nonzero w, ``delta(w.x+b)`` is surface measure divided by ``||w||``.
Coefficients here use that pullback convention; no unit-normal assumption is
made. No products of distributions, boundary conditions or source inversion
are supplied. This is classical distribution calculus, not a novelty claim.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from operator import index as integer_index
from typing import Any

from omnibias.core.ridge_system import (
    RidgeBudget,
    RidgeSystem,
    RidgeSyzygy,
    Vector,
    _direction,
    _exact,
    _integer,
)


def _profile_index(value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError("profile indices must be integers")
    try:
        return integer_index(value)
    except TypeError as exc:
        raise ValueError("profile indices must be integers") from exc


@dataclass(frozen=True)
class RidgeDistribution:
    r"""``sum_j terms[j] * T_j(w.x+b)`` with exact vector coefficients.

    ``terms`` stores ``(profile_index, coefficient_vector)`` pairs. Duplicate
    indices are combined; zero vectors are removed. An empty tuple is a zero
    field with the declared output dimension. For regular profiles, the
    coefficient of T_j is the jump of the j-th profile derivative at zero.

    Metadata is rational and immutable. Finite-budget exact arithmetic proves
    coefficient identities, not floating-point enclosure or PDE completeness.
    Negative indices must be evaluated through weak pairings, not point values.
    """

    direction: Vector
    terms: tuple[tuple[int, Vector], ...]
    outputs: int = 1
    bias: Fraction = Fraction(0)
    budget: RidgeBudget = RidgeBudget()

    def __post_init__(self) -> None:
        if not isinstance(self.budget, RidgeBudget):
            raise TypeError("budget must be a RidgeBudget")
        limits = self.budget
        outputs = _integer(self.outputs, "outputs", 1)
        limits._entries(len(self.direction) + (len(self.terms) + 1) * outputs)
        direction = _direction(self.direction)
        for value in direction:
            limits._coefficient(value)
        bias = limits._coefficient(_exact(self.bias))
        combined: dict[int, list[Fraction]] = {}
        for raw_index, raw_row in self.terms:
            index = _profile_index(raw_index)
            if index > limits.max_degree or -index - 1 > limits.max_order:
                raise ValueError("distribution profile exceeds the order budget")
            if len(raw_row) != outputs:
                raise ValueError("coefficient vector must match outputs")
            row = combined.setdefault(index, [Fraction(0)] * outputs)
            for component, raw_value in enumerate(raw_row):
                value = limits._coefficient(_exact(raw_value))
                row[component] = limits._coefficient(row[component] + value)
        terms = tuple((index, tuple(row)) for index, row in sorted(combined.items()) if any(row))
        object.__setattr__(self, "direction", direction)
        object.__setattr__(self, "terms", terms)
        object.__setattr__(self, "outputs", outputs)
        object.__setattr__(self, "bias", bias)

    @classmethod
    def from_syzygy(
        cls, pack: RidgeSyzygy, profile_index: int = -1, *,
        bias: int | Fraction = 0, budget: RidgeBudget | None = None,
    ) -> RidgeDistribution:
        """Evaluate a derivative pack on a distributional profile T_index.

        This is the weak counterpart of the smooth ``ridge_solution``.
        A kernel pack gives the same PDE identity, now including singular
        derivatives. Directly constructed RidgeSyzygy metadata need not be
        in any PDE kernel; check the result with ``apply(system)``.
        """
        if not isinstance(pack, RidgeSyzygy):
            raise TypeError("pack must be a RidgeSyzygy")
        limits = budget if budget is not None else RidgeBudget()
        if not isinstance(limits, RidgeBudget):
            raise TypeError("budget must be a RidgeBudget")
        limits._entries(len(pack.direction) + len(pack.coefficients) * pack.outputs)
        index = _profile_index(profile_index)
        return cls(pack.direction, tuple((index-order, row)
                                         for order, row in enumerate(pack.coefficients)),
                   outputs=pack.outputs, bias=_exact(bias),
                   budget=limits)

    @property
    def regular_terms(self) -> tuple[tuple[int, Vector], ...]:
        """Heaviside and truncated-power terms (not singular measures)."""
        return tuple(term for term in self.terms if term[0] >= 0)

    @property
    def singular_terms(self) -> tuple[tuple[int, Vector], ...]:
        """Dirac/Dirac-derivative terms, using the pullback convention."""
        return tuple(term for term in self.terms if term[0] < 0)

    def apply(self, system: RidgeSystem) -> RidgeDistribution:
        r"""Apply a restricted linear PDE system, including singular sources.

        ``A_k D_z^k (c_j T_j) = A_k c_j T_(j-k)``. Direction and input
        dimension must match exactly. The result retains the same plane and
        has one output per equation. This does not form any products such as
        delta squared and does not silently discard singular derivatives.
        """
        if not isinstance(system, RidgeSystem):
            raise TypeError("system must be a RidgeSystem")
        if system.direction != self.direction or system.outputs != self.outputs:
            raise ValueError("system direction and input outputs must match the distribution")
        limits = self.budget
        limits._work(len(self.terms) * len(system.matrices) * system.equations * self.outputs)
        limits._entries(len(self.terms) * len(system.matrices) * system.equations)
        combined: dict[int, list[Fraction]] = {}
        for index, vector in self.terms:
            for order, matrix in enumerate(system.matrices):
                for equation, row in enumerate(matrix):
                    value = Fraction(0)
                    for a, c in zip(row, vector, strict=True):
                        value = limits._coefficient(value + limits._coefficient(a * c))
                    if value:
                        target = combined.setdefault(index - order, [Fraction(0)] * system.equations)
                        target[equation] = limits._coefficient(target[equation] + value)
        return RidgeDistribution(
            self.direction,
            tuple((index, tuple(row)) for index, row in combined.items()),
            outputs=system.equations,
            bias=self.bias,
            budget=limits,
        )


def gaussian_pairing_orders(field: RidgeDistribution) -> tuple[int, int]:
    """Largest regular degree and Dirac derivative order; -1 means absent.

    For Y~N(m,v), weak pairing obeys F_0=Phi(m/sqrt(v)),
    F_1=m*F_0+v*p_Y(0), and F_j=(m*F_(j-1)+v*F_(j-2))/j.
    Negative profiles pair as d_m^k p_Y(0), using probabilists' Hermites
    from the shared coefficient definition. Backend twins implement this
    recurrence with runtime m,v, preserving their parameter gradients.
    """
    if not isinstance(field, RidgeDistribution):
        raise TypeError("field must be a RidgeDistribution")
    regular = max((index for index, _ in field.regular_terms), default=-1)
    singular = max((-index - 1 for index, _ in field.singular_terms), default=-1)
    return regular, singular


@dataclass(frozen=True)
class GaussianClusterCollapse:
    """Finite weak jet and an exact rational ANALYTIC error bound.

    The bound is uniform in Gaussian probe mean and applies when the
    projected probe standard deviation is >= ``min_probe_std``. It does
    not enclose backend rounding and does not bound pointwise source error.
    A signed derivative distribution is not a probability density.
    """

    field: RidgeDistribution
    min_probe_std: Fraction
    weak_error_bound: Fraction


def collapse_dirac_cluster(
    direction: Sequence[int | Fraction],
    offsets: Sequence[int | Fraction],
    strengths: Sequence[int | Fraction],
    *,
    max_order: int = 3,
    min_probe_std: int | Fraction,
    bias: int | Fraction = 0,
    budget: RidgeBudget | None = None,
) -> GaussianClusterCollapse:
    r"""Approximate sum a_i delta(w.x+b+s_i) by confluent normal derivatives.

    Coefficient k is ``sum(a_i*s_i**k)/k!`` multiplying delta^(k). The
    original planes share w; their scalar shifts s_i are measured in that
    normal coordinate, not Euclidean distance unless w is a unit normal.

    Set r=max_order+1. For Gaussian projected std >= sigma>0, Taylor's
    theorem and a Fourier Gaussian moment bound give

        error <= C_r/(2 sigma**(r+1)) * sum |a_i| |s_i|**r / r!,

    where C_(2q)=(2q-1)!! and C_(2q+1)=(2q+1)!!. For odd r, Cauchy-Schwarz
    bounds the Fourier moment between its adjacent even moments. Replacing
    1/sqrt(2pi) by 1/2 makes the bound rational and conservative. This is
    a bound on real-arithmetic weak measurements, not a floating certificate.
    It preserves measured fields within that budget, not recoverability of
    individual near-colliding source locations or positivity of the source.
    """
    limits = budget if budget is not None else RidgeBudget()
    if not isinstance(limits, RidgeBudget):
        raise TypeError("budget must be a RidgeBudget")
    order = _integer(max_order, "max_order")
    if order > limits.max_order:
        raise ValueError("cluster order exceeds the resource budget")
    if not offsets or len(offsets) != len(strengths):
        raise ValueError("nonempty matching offsets and strengths are required")
    limits._entries(len(direction) + len(offsets) * (order + 2))
    limits._work(len(offsets) * (order + 2) * (order + 2))
    sigma = limits._coefficient(_exact(min_probe_std))
    if sigma <= 0:
        raise ValueError("min_probe_std must be positive")
    shifts = tuple(limits._coefficient(_exact(value)) for value in offsets)
    weights = tuple(limits._coefficient(_exact(value)) for value in strengths)
    moments = [Fraction(0)] * (order + 1)
    remainder_moment = Fraction(0)
    for shift, weight in zip(shifts, weights, strict=True):
        term = weight
        for k in range(order + 1):
            moments[k] = limits._coefficient(moments[k] + term)
            term = limits._coefficient(term * shift / (k + 1))
        remainder_moment = limits._coefficient(remainder_moment + abs(term))
    r = order + 1
    denominator = Fraction(2)
    for _ in range(r + 1):
        denominator = limits._coefficient(denominator * sigma)
    constant = Fraction(1)
    for odd in range(1, r + 1, 2):
        constant = limits._coefficient(constant * odd)
    bound = limits._coefficient(remainder_moment * constant / denominator)
    field = RidgeDistribution(tuple(_exact(value) for value in direction),
                              tuple((-k - 1, (value,)) for k, value in enumerate(moments)),
                              bias=_exact(bias), budget=limits)
    return GaussianClusterCollapse(field, sigma, bound)


__all__ = ["GaussianClusterCollapse", "RidgeDistribution", "collapse_dirac_cluster",
           "gaussian_pairing_orders"]
