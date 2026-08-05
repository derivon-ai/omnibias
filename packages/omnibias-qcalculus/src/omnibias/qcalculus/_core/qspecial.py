# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""q-special functions: the two q-exponentials and q-deformed Bernoulli / Euler numbers.

* **q-exponentials** (numerical): ``e_q(z) = sum_n z^n / [n]_q!`` (radius ``1/(1-q)`` for
  ``0 < q < 1``) and the entire ``E_q(z) = sum_n q^{n(n-1)/2} z^n / [n]_q!``. They are
  reciprocal, ``e_q(z) E_q(-z) = 1``.
* **q-Bernoulli / q-Euler numbers** (exact): the q-deformations of the classical defining
  recurrences with the Gaussian binomial in place of the ordinary one --

  .. math::

      \sum_{k=0}^{n} \binom{n+1}{k}_q B_k(q) = 0, \qquad
      \sum_{k=0}^{m} \binom{2m}{2k}_q E_{2k}(q) = 0,

  with ``B_0(q) = E_0(q) = 1``. Both reduce to the classical Bernoulli / (secant) Euler
  numbers as ``q -> 1`` (verified in the tests). Exact :class:`~fractions.Fraction`.
"""

from __future__ import annotations

from fractions import Fraction

from omnibias.qcalculus._core.qnumbers import q_binomial, q_bracket

Rational = Fraction | int


def _frac(v: Rational) -> Fraction:
    return v if isinstance(v, Fraction) else Fraction(v)


def _q_bracket_float(n: int, q: float) -> float:
    """Float ``[n]_q`` accumulated as a geometric sum (stable, no huge Fractions)."""
    total = 0.0
    power = 1.0
    for _ in range(n):
        total += power
        power *= q
    return total


def q_exp(z: float, q: float, *, terms: int = 128) -> float:
    r"""The q-exponential ``e_q(z) = sum_{n>=0} z^n / [n]_q!`` (numerical, ``0 < q < 1``).

    Converges for ``|z| < 1/(1-q)``. ``terms`` truncates the series; evaluated in float via
    the recurrence ``t_n = t_{n-1} z / [n]_q`` (no exact-Fraction factorial blow-up).
    """
    if not 0.0 < q < 1.0:
        raise ValueError(f"q_exp needs 0 < q < 1, got q={q}")
    if terms < 1:
        raise ValueError(f"terms must be >= 1, got {terms}")
    total = 0.0
    term = 1.0  # z^n / [n]_q!
    for n in range(terms):
        total += term
        term *= z / _q_bracket_float(n + 1, q)
    return total


def q_exp_big(z: float, q: float, *, terms: int = 128) -> float:
    r"""The entire q-exponential ``E_q(z) = sum_{n>=0} q^{n(n-1)/2} z^n / [n]_q!`` (numerical).

    Uses ``t_n = t_{n-1} (q^{n-1} z) / [n]_q`` since ``q^{C(n,2)}/q^{C(n-1,2)} = q^{n-1}``.
    """
    if not 0.0 < q < 1.0:
        raise ValueError(f"q_exp_big needs 0 < q < 1, got q={q}")
    if terms < 1:
        raise ValueError(f"terms must be >= 1, got {terms}")
    total = 0.0
    term = 1.0  # q^{C(n,2)} z^n / [n]_q!
    for n in range(terms):
        total += term
        term *= (q**n * z) / _q_bracket_float(n + 1, q)
    return total


def q_bernoulli(n: int, q: Rational) -> Fraction:
    r"""The q-Bernoulli number ``B_n(q)`` (exact; ``-> B_n`` as ``q -> 1``).

    From ``sum_{k=0}^{n} [n+1 choose k]_q B_k(q) = 0`` (``B_0 = 1``), i.e.
    ``B_n(q) = -(1/[n+1]_q) sum_{k=0}^{n-1} [n+1 choose k]_q B_k(q)``.
    """
    if n < 0:
        raise ValueError(f"q_bernoulli needs n >= 0, got {n}")
    bern = [Fraction(1)]  # B_0
    for m in range(1, n + 1):
        s = sum((q_binomial(m + 1, k, q) * bern[k] for k in range(m)), Fraction(0))
        bern.append(-s / q_bracket(m + 1, q))
    return bern[n]


def q_euler(n: int, q: Rational) -> Fraction:
    r"""The q-Euler (secant) number ``E_n(q)`` (exact; ``0`` for odd ``n``; ``-> E_n`` as ``q -> 1``).

    From ``sum_{k=0}^{m} [2m choose 2k]_q E_{2k}(q) = 0`` (``E_0 = 1``), i.e.
    ``E_{2m}(q) = -sum_{k=0}^{m-1} [2m choose 2k]_q E_{2k}(q)``.
    """
    if n < 0:
        raise ValueError(f"q_euler needs n >= 0, got {n}")
    if n % 2 == 1:
        return Fraction(0)
    evens = [Fraction(1)]  # E_0
    for m in range(1, n // 2 + 1):
        s = sum((q_binomial(2 * m, 2 * k, q) * evens[k] for k in range(m)), Fraction(0))
        evens.append(-s)
    return evens[n // 2]


__all__ = [
    "q_bernoulli",
    "q_euler",
    "q_exp",
    "q_exp_big",
]
