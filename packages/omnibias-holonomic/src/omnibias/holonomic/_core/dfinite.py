# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""D-finite functions and P-recursive sequences with closure operations.

* :class:`PRecursive` -- a sequence annihilated by a shift Ore operator plus enough initial
  values to run the recurrence forward.
* :class:`DFinite` -- a power series annihilated by a differential Ore operator plus initial
  Taylor coefficients.

The class is closed under termwise sum, termwise (Hadamard) product, and Cauchy product.
Sum and Hadamard product are computed **symbolically** and for all ``n``: the sum via the
Ore ``lclm`` and the Hadamard product via the shift ``symmetric_product``
(:mod:`omnibias.holonomic._core.oreops`), so the returned annihilator is exact by
construction, not merely range-verified. If the symbolic construction is unavailable the
code falls back to the ansatz path (fit an annihilator to a prefix via
:func:`omnibias.symbolic.discover_recurrence`, then re-check it exactly on held-out terms
-- sound, but only verified on the range). The Cauchy product stays on the ansatz path
(its clean symbolic route is the differential product of the generating functions, exposed
in :mod:`omnibias.holonomic._core.transforms`). Every returned object is verified to
regenerate the combined sequence exactly.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from fractions import Fraction

from omnibias.holonomic._core.guess import guess_recurrence
from omnibias.holonomic._core.ore import OrePolynomial
from omnibias.holonomic._core.oreops import lclm, symmetric_product
from omnibias.holonomic._core.rational_poly import peval


@dataclass(frozen=True)
class PRecursive:
    """A P-recursive sequence: a shift annihilator + initial values."""

    annihilator: OrePolynomial
    initial: tuple[Fraction, ...]

    def __post_init__(self) -> None:
        r = self.annihilator.order
        if r < 1:
            raise ValueError("annihilator must have order >= 1")
        if len(self.initial) < r:
            raise ValueError(f"need >= {r} initial values, got {len(self.initial)}")

    def terms(self, count: int) -> list[Fraction]:
        """Generate the first ``count`` terms by running the recurrence forward."""
        r = self.annihilator.order
        lead = self.annihilator.coeffs[r]
        a = [Fraction(v) for v in self.initial]
        while len(a) < count:
            m = len(a) - r
            denom = peval(lead, m)
            if denom == 0:
                raise ValueError(f"leading coefficient vanishes at m={m}; singular recurrence")
            acc = Fraction(0)
            for i in range(r):
                acc += peval(self.annihilator.coeffs[i], m) * a[m + i]
            a.append(-acc / denom)
        return a[:count]

    def term(self, n: int) -> Fraction:
        """The ``n``-th term."""
        return self.terms(n + 1)[n]


@dataclass(frozen=True)
class DFinite:
    """A D-finite power series: a differential annihilator + initial Taylor coefficients."""

    annihilator: OrePolynomial
    initial: tuple[Fraction, ...]

    def taylor(self, count: int) -> list[Fraction]:
        """Generate the first ``count`` Taylor coefficients from the ODE recurrence.

        The differential operator ``sum_i c_i(x) D^i`` gives, at each series order, a linear
        relation whose highest-index unknown is solved for.
        """
        op = self.annihilator
        coeffs = [Fraction(v) for v in self.initial]
        order = op.order
        # The coefficient recurrence relates a_{m+...}; solve order-by-order.
        s = 0
        while len(coeffs) < count:
            # coefficient of x^s in op(f) must be 0; isolate the highest unknown a_{s+order}.
            target = s + order
            lead = Fraction(0)
            rest = Fraction(0)
            for i, c in enumerate(op.coeffs):
                for d_deg, cc in enumerate(c):
                    m = s - d_deg + i
                    if m < 0:
                        continue
                    falling = Fraction(1)
                    for t in range(i):
                        falling *= m - t
                    weight = cc * falling
                    if m == target:
                        lead += weight
                    elif m < len(coeffs):
                        rest += weight * coeffs[m]
            if lead == 0:
                raise ValueError(f"cannot solve for Taylor coefficient at order {target}")
            coeffs.append(-rest / lead)
            s += 1
        return coeffs[:count]


def _fit_annihilator(samples: Sequence[Fraction], max_order: int, max_index_degree: int) -> OrePolynomial:
    op = guess_recurrence(samples, max_order=max_order, max_index_degree=max_index_degree)
    if op is None:
        raise ValueError("no P-recursive relation found within the given bounds")
    return op


def dfinite_add(a: PRecursive, b: PRecursive, *, terms: int = 40) -> PRecursive:
    """Termwise sum ``a + b`` -- symbolic Ore ``lclm`` (all-``n``), ansatz fallback."""
    ta, tb = a.terms(terms), b.terms(terms)
    combined = [ta[n] + tb[n] for n in range(terms)]
    return _symbolic_or_ansatz(lambda: lclm(a.annihilator, b.annihilator), combined)


def dfinite_hadamard(a: PRecursive, b: PRecursive, *, terms: int = 40) -> PRecursive:
    """Termwise (Hadamard) product ``a . b`` -- symbolic ``symmetric_product``, ansatz fallback."""
    ta, tb = a.terms(terms), b.terms(terms)
    combined = [ta[n] * tb[n] for n in range(terms)]
    return _symbolic_or_ansatz(lambda: symmetric_product(a.annihilator, b.annihilator), combined)


def dfinite_cauchy(a: PRecursive, b: PRecursive, *, terms: int = 40) -> PRecursive:
    """Cauchy product ``c_n = sum_i a_i b_{n-i}`` (P-recursive closure by verified ansatz)."""
    ta, tb = a.terms(terms), b.terms(terms)
    conv = [sum((ta[i] * tb[n - i] for i in range(n + 1)), Fraction(0)) for n in range(terms)]
    return _fit_and_wrap(conv)


def _symbolic_or_ansatz(op_fn: Callable[[], OrePolynomial], samples: Sequence[Fraction]) -> PRecursive:
    """Wrap the symbolic annihilator around ``samples``; fall back to the ansatz fit."""
    try:
        return _wrap_with_op(op_fn(), samples)
    except (ValueError, ZeroDivisionError):
        return _fit_and_wrap(samples)


def _wrap_with_op(op: OrePolynomial, samples: Sequence[Fraction]) -> PRecursive:
    """Seed a known annihilator with enough initials (stepping over singularities) + verify."""
    order = op.order
    if order < 1:
        raise ValueError("annihilator must have order >= 1")
    lead = op.coeffs[order]
    # Seed enough initial values to step over every integer root of the leading
    # coefficient: an index whose recurrence divisor vanishes is not determined by the
    # relation and must be supplied directly.
    singular = [m for m in range(max(0, len(samples) - order + 1)) if peval(lead, m) == 0]
    n_init = order if not singular else max(singular) + order + 1
    n_init = min(n_init, len(samples))
    if n_init < order:
        raise ValueError("not enough samples to seed the annihilator")
    result = PRecursive(op, tuple(samples[:n_init]))
    regenerated = result.terms(len(samples))
    if any(regenerated[i] != samples[i] for i in range(len(samples))):
        raise ValueError("annihilator failed exact verification on generated terms")
    return result


def _fit_and_wrap(samples: Sequence[Fraction]) -> PRecursive:
    op = _fit_annihilator(samples, max_order=6, max_index_degree=4)
    return _wrap_with_op(op, samples)


__all__ = [
    "DFinite",
    "PRecursive",
    "dfinite_add",
    "dfinite_cauchy",
    "dfinite_hadamard",
]
