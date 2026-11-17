# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Exact desingularized series for the universal critical Gaussian inverse."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

from omnibias.core.gaussian_jet import _Arithmetic
from omnibias.core.ridge_system import RidgeBudget

Rational = int | Fraction
Polynomial = tuple[Fraction, ...]


def _order(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("order must be an integer, excluding bool")
    if value < 0:
        raise ValueError("order must be nonnegative")
    return value



@dataclass(frozen=True, init=False)
class GaussianCriticalSeries:
    r"""Ordinary Taylor polynomial of the normalized critical branch chi.

    The analytic branch chi satisfies chi(0)=0, chi'(0)=1 and
    chi(t)-tanh(chi(t))=t³/3. Equivalently chi(t)=t*z(t²), where z(0)=1.
    ``coefficients[n]`` is the coefficient of t**n, including every explicit
    even zero. ``z_coefficients[k]`` is the coefficient of u**k in z(u), with
    k=0..floor((order-1)/2); it is empty when order=0. No unrequested higher
    coefficient is computed for an even order.

    This universal exact artifact has no Gaussian model parameters, numerical
    root or remainder bound. A finite polynomial is not asserted to solve the
    inverse equation exactly away from zero.
    """

    coefficients: Polynomial
    z_coefficients: Polynomial
    budget: RidgeBudget

    def __init__(self) -> None:
        raise TypeError("use gaussian_critical_series")

    @property
    def order(self) -> int:
        return len(self.coefficients)-1

    def evaluate(self, parameter: Rational) -> Fraction:
        """Evaluate the finite polynomial exactly; does not add a tail bound."""
        arithmetic = _Arithmetic(self.budget)
        value = arithmetic.exact(parameter)
        arithmetic._entries(8)
        arithmetic._work(4*(self.order+1)+8)
        result = self.coefficients[-1]
        for coefficient in reversed(self.coefficients[:-1]):
            result = arithmetic.add(coefficient, arithmetic.mul(value, result))
        return result


def _build_critical_series(order: int, limits: RidgeBudget, arithmetic: _Arithmetic) -> GaussianCriticalSeries:
    """Build with a caller's cumulative exact meter, without resetting it."""
    degree = _order(order)
    if not isinstance(limits, RidgeBudget):
        raise TypeError("budget must be a RidgeBudget")
    if arithmetic.limits != limits:
        raise ValueError("shared arithmetic and series budgets must agree")
    if degree > limits.max_order:
        raise ValueError("critical inverse series order exceeds the resource budget")
    z_count = (degree+1)//2
    highest = max(0, z_count-1)
    arithmetic._entries(32*(highest+1)+8*(degree+1)+64)
    # Each step uses two convolution sums with at most five operations per
    # previously known coefficient pair. Reserve the whole quadratic work
    # before creating the three streamed coefficient arrays.
    arithmetic._work(5*highest*(highest-1)//2 + 8*highest + 32*(degree+1) + 64)
    z: list[Fraction] = []
    if z_count:
        z, w, square = [Fraction(1)], [Fraction(1)], [Fraction(1)]
        for n in range(1, z_count):
            h = arithmetic.sum(tuple(arithmetic.mul(w[k], w[n-k]) for k in range(1, n)))
            c = arithmetic.sum(tuple(arithmetic.mul(arithmetic.mul(Fraction(2*k+1), z[k]), square[n-k])
                                     for k in range(1, n)))
            forcing = Fraction(2, 3) if n == 1 else Fraction(0)
            numerator = arithmetic.add(arithmetic.add(forcing, -h), -c)
            z.append(arithmetic._coefficient(numerator/(2*n+3)))
            w.append(arithmetic.add(z[-1], -Fraction(1, 3) if n == 1 else Fraction(0)))
            square.append(arithmetic.add(arithmetic.mul(Fraction(2), w[-1]), h))
    coefficients = [Fraction(0)]*(degree+1)
    for k, value in enumerate(z):
        coefficients[2*k+1] = value
    result = object.__new__(GaussianCriticalSeries)
    object.__setattr__(result, "coefficients", tuple(coefficients))
    object.__setattr__(result, "z_coefficients", tuple(z))
    object.__setattr__(result, "budget", limits)
    return result


def gaussian_critical_series(order: int, *, budget: RidgeBudget | None = None) -> GaussianCriticalSeries:
    r"""Construct exact coefficients of chi(t), where chi-tanh(chi)=t³/3.

    Write chi=t*z(u), u=t², with z(0)=1. Differentiating the inverse
    equation and using tanh(chi)=chi-t³/3 gives the nonsingular identity
    (z+2*u*z')*(z-u/3)**2=1. Set w_n=z_n-delta(n,1)/3 and p=w*w. At each
    n>=1, h_n=sum_{k=1}^{n-1} w_k*w_(n-k) and
    c_n=sum_{k=1}^{n-1} (2*k+1)*z_k*p_(n-k) use only earlier coefficients.
    The next coefficient is z_n=((2/3)*delta(n,1)-h_n-c_n)/(2*n+3),
    followed by w_n and p_n=2*w_n+h_n. Only three coefficient arrays are
    retained; no factorials, power tables, matrix solve or transcendental
    evaluation is required.

    The differentiated identity retains the original normalized branch:
    y=chi-t³/3 obeys y'=chi'*(1-y²), y(0)=0. Thus atanh(y)-chi has zero
    derivative and zero constant term, proving y=tanh(chi) as formal series.
    The analytic desingularized branch is fixed by z(0)=1.

    Orders include zero and even values; bool, negative and noninteger orders
    are invalid. Intermediate rational bits, retained storage and aggregate
    arithmetic are bounded by the supplied budget. The streaming recurrence
    uses O(K²) arithmetic and O(K+order) storage, K=floor((order-1)/2) for positive order.
    Resource failure raises ValueError. This is a finite series, not a tail or
    model-binding certificate, and makes no timing or universal PDE claim.
    """
    degree = _order(order)
    limits = RidgeBudget() if budget is None else budget
    if not isinstance(limits, RidgeBudget):
        raise TypeError("budget must be a RidgeBudget")
    return _build_critical_series(degree, limits, _Arithmetic(limits))


__all__ = ["GaussianCriticalSeries", "gaussian_critical_series"]
