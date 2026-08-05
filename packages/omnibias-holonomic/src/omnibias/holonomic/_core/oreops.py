# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Ore-operator Euclidean domain: right division, GCRD, LCLM, symmetric product.

These are the *symbolic* backbone of holonomic closure. Over the field ``Q(x)`` an Ore
algebra is a (left and right) Euclidean domain, so operators have a right division
``A = Q B + R`` (``ord R < ord B``), a greatest common right divisor, and a least common
left multiple. For annihilators:

* ``lclm(A, B)`` annihilates ``f + g`` whenever ``A f = 0`` and ``B g = 0`` -- the
  **sum** closure, proven for *all* ``x`` (not merely a verified range);
* ``symmetric_product(A, B)`` annihilates the product ``f g`` (Leibniz on the
  differential algebra) or the termwise product ``a_n b_n`` (multiplicative shift on the
  shift algebra) -- the **product / Hadamard** closure.

Coefficients live in ``Q(x)`` during the algorithm (:mod:`.ratfunc`); the returned
:class:`~.ore.OrePolynomial` has its denominators cleared and its content removed
(left-multiplication by a non-zero polynomial preserves the annihilator, so the result is
still a valid -- and minimal-order -- annihilator). Everything is exact.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

from omnibias.holonomic._core.ore import OreAlgebra, OrePolynomial
from omnibias.holonomic._core.ratfunc import (
    RatFunc,
    rf_add,
    rf_div,
    rf_from_poly,
    rf_from_rational,
    rf_is_zero,
    rf_mul,
    rf_neg,
    rf_normalize,
    rf_shift,
    rf_sub,
    rf_zero,
)
from omnibias.holonomic._core.rational_poly import (
    Poly,
    is_zero,
    pderiv,
    pdivmod,
    pgcd,
    pmul,
    psub,
)
from omnibias.holonomic._core.relations import find_poly_relation

RFOp = list[RatFunc]

_ONE: Poly = (Fraction(1),)


# --------------------------------------------------------------------------- #
# Rational-function operator helpers (index = power of the operator d).
# --------------------------------------------------------------------------- #
def _plcm(a: Poly, b: Poly) -> Poly:
    if is_zero(a) or is_zero(b):
        return ()
    g = pgcd(a, b)
    q, _ = pdivmod(pmul(a, b), g)
    return q


def _sigma_rf(algebra: OreAlgebra, rf: RatFunc) -> RatFunc:
    if algebra.name == "shift":
        return rf_shift(rf, 1)
    if algebra.name == "differential":
        return rf
    raise ValueError(f"unsupported Ore algebra for symbolic ops: {algebra.name!r}")


def _delta_rf(algebra: OreAlgebra, rf: RatFunc) -> RatFunc:
    if algebra.name == "shift":
        return rf_zero()
    if algebra.name == "differential":
        num, den = rf
        n2 = psub(pmul(pderiv(num), den), pmul(num, pderiv(den)))
        d2 = pmul(den, den)
        return rf_normalize(n2, d2)
    raise ValueError(f"unsupported Ore algebra for symbolic ops: {algebra.name!r}")


def _rf_trim(op: RFOp) -> RFOp:
    out = list(op)
    while out and rf_is_zero(out[-1]):
        out.pop()
    return out


def _rf_order(op: RFOp) -> int:
    t = _rf_trim(op)
    return len(t) - 1


def _rf_all_zero(op: RFOp) -> bool:
    return all(rf_is_zero(c) for c in op)


def _rf_sub_ops(a: RFOp, b: RFOp) -> RFOp:
    n = max(len(a), len(b))
    out: RFOp = []
    for i in range(n):
        x = a[i] if i < len(a) else rf_zero()
        y = b[i] if i < len(b) else rf_zero()
        out.append(rf_sub(x, y))
    return _rf_trim(out)


def _d_power_times_rf(algebra: OreAlgebra, i: int, rf: RatFunc) -> RFOp:
    """``d^i * rf`` as a coefficient list (index l = coefficient of ``d^l``)."""
    result: RFOp = [rf]
    for _ in range(i):
        nxt: RFOp = [rf_zero()] * (len(result) + 1)
        for l_idx, r in enumerate(result):
            if rf_is_zero(r):
                continue
            nxt[l_idx + 1] = rf_add(nxt[l_idx + 1], _sigma_rf(algebra, r))
            nxt[l_idx] = rf_add(nxt[l_idx], _delta_rf(algebra, r))
        result = nxt
    return result


def _mul_rf(algebra: OreAlgebra, a: RFOp, b: RFOp) -> RFOp:
    acc: RFOp = []

    def add_into(idx: int, val: RatFunc) -> None:
        while len(acc) <= idx:
            acc.append(rf_zero())
        acc[idx] = rf_add(acc[idx], val)

    for i, ai in enumerate(a):
        if rf_is_zero(ai):
            continue
        for j, bj in enumerate(b):
            if rf_is_zero(bj):
                continue
            theta = _d_power_times_rf(algebra, i, bj)
            for l_idx, coeff in enumerate(theta):
                if rf_is_zero(coeff):
                    continue
                add_into(l_idx + j, rf_mul(ai, coeff))
    return _rf_trim(acc)


def _divmod_rf(algebra: OreAlgebra, a: RFOp, b: RFOp) -> tuple[RFOp, RFOp]:
    ordb = _rf_order(b)
    if ordb < 0:
        raise ZeroDivisionError("Ore division by the zero operator")
    blead = b[ordb]
    quotient: RFOp = []
    remainder = _rf_trim(list(a))
    while _rf_order(remainder) >= ordb and not _rf_all_zero(remainder):
        ordr = _rf_order(remainder)
        shift = ordr - ordb
        sig = blead
        for _ in range(shift):
            sig = _sigma_rf(algebra, sig)
        c = rf_div(remainder[ordr], sig)
        while len(quotient) <= shift:
            quotient.append(rf_zero())
        quotient[shift] = rf_add(quotient[shift], c)
        term: RFOp = [rf_zero()] * shift + [c]
        remainder = _rf_sub_ops(remainder, _mul_rf(algebra, term, b))
    return _rf_trim(quotient), _rf_trim(remainder)


def _op_to_rf(op: OrePolynomial) -> RFOp:
    return _rf_trim([rf_from_poly(c) for c in op.coeffs])


def _content_reduce(coeffs: list[Poly]) -> list[Poly]:
    nz = [c for c in coeffs if not is_zero(c)]
    if not nz:
        return coeffs
    g = nz[0]
    for c in nz[1:]:
        g = pgcd(g, c)
        if len(g) <= 1:
            break
    if len(g) > 1:
        coeffs = [() if is_zero(c) else pdivmod(c, g)[0] for c in coeffs]
    return coeffs


def _rf_to_op(algebra: OreAlgebra, op: RFOp) -> OrePolynomial:
    op = _rf_trim(op)
    if not op:
        return algebra.operator([[]])
    lcm: Poly = _ONE
    for num, den in op:
        if not is_zero(num):
            lcm = _plcm(lcm, den)
    polys: list[Poly] = []
    for num, den in op:
        if is_zero(num):
            polys.append(())
        else:
            factor, _ = pdivmod(lcm, den)
            polys.append(pmul(num, factor))
    return algebra.operator(_content_reduce(polys))


def _op_from_polys(algebra: OreAlgebra, polys: list[Poly]) -> OrePolynomial:
    return algebra.operator(_content_reduce(polys))


# --------------------------------------------------------------------------- #
# Public API.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class OreDivision:
    """Right division data: ``left_multiplier * A = quotient * B + remainder``.

    Because right division lives over ``Q(x)`` the quotient/remainder can carry
    denominators; clearing them multiplies the dividend on the left by the polynomial
    :attr:`left_multiplier` (a scalar, order-0 operator), so the returned operators have
    polynomial coefficients and the exact identity above holds.
    """

    quotient: OrePolynomial
    remainder: OrePolynomial
    left_multiplier: Poly


def ore_divmod(a: OrePolynomial, b: OrePolynomial) -> OreDivision:
    """Right-divide ``a`` by ``b``: :class:`OreDivision` with ``lambda a = q b + r``."""
    if a.algebra.name != b.algebra.name:
        raise ValueError("operators must share an Ore algebra")
    algebra = a.algebra
    q_rf, r_rf = _divmod_rf(algebra, _op_to_rf(a), _op_to_rf(b))
    lcm: Poly = _ONE
    for num, den in list(q_rf) + list(r_rf):
        if not is_zero(num):
            lcm = _plcm(lcm, den)

    def _clear(op: RFOp) -> list[Poly]:
        out: list[Poly] = []
        for num, den in op:
            if is_zero(num):
                out.append(())
            else:
                factor, _ = pdivmod(lcm, den)
                out.append(pmul(num, factor))
        return out

    return OreDivision(
        quotient=algebra.operator(_clear(q_rf) or [[]]),
        remainder=algebra.operator(_clear(r_rf) or [[]]),
        left_multiplier=lcm,
    )


def gcrd(a: OrePolynomial, b: OrePolynomial) -> OrePolynomial:
    """Greatest common right divisor of ``a`` and ``b`` (up to a left factor)."""
    if a.algebra.name != b.algebra.name:
        raise ValueError("operators must share an Ore algebra")
    algebra = a.algebra
    x = _op_to_rf(a)
    y = _op_to_rf(b)
    if _rf_order(x) < _rf_order(y):
        x, y = y, x
    while not _rf_all_zero(y):
        _, r = _divmod_rf(algebra, x, y)
        x, y = y, _rf_trim(r)
    return _rf_to_op(algebra, x)


def lclm(a: OrePolynomial, b: OrePolynomial) -> OrePolynomial:
    """Least common left multiple: the minimal operator right-divisible by both.

    Computed by reducing ``d^0, d^1, ...`` modulo ``a`` and modulo ``b`` and finding the
    smallest polynomial combination whose two remainders both vanish (exact null space).
    Annihilates ``f + g`` for ``a f = 0``, ``b g = 0``.
    """
    if a.algebra.name != b.algebra.name:
        raise ValueError("operators must share an Ore algebra")
    algebra = a.algebra
    pa, pb = a.order, b.order
    if pa < 0 or pb < 0:
        raise ValueError("lclm needs non-zero operators")
    a_rf, b_rf = _op_to_rf(a), _op_to_rf(b)
    columns: list[list[RatFunc]] = []
    for k in range(pa + pb + 1):
        monomial: RFOp = [rf_zero()] * k + [rf_from_rational(1)]
        _, ra = _divmod_rf(algebra, monomial, a_rf)
        _, rb = _divmod_rf(algebra, monomial, b_rf)
        vec = [ra[i] if i < len(ra) else rf_zero() for i in range(pa)]
        vec += [rb[i] if i < len(rb) else rf_zero() for i in range(pb)]
        columns.append(vec)
    rel = find_poly_relation(columns, max_degree=8)
    if rel is None:
        raise ValueError("no LCLM found within the degree bound")
    return _op_from_polys(algebra, rel)


def _tensor_reductions(op: OrePolynomial) -> list[RatFunc]:
    """``r_t = -op_t / op_p`` so that ``x^{(p)} = sum_t r_t x^{(t)}`` (p = order)."""
    p = op.order
    lead = rf_from_poly(op.coeffs[p])
    out: list[RatFunc] = []
    for t in range(p):
        ct = op.coeffs[t] if t < len(op.coeffs) else ()
        out.append(rf_neg(rf_div(rf_from_poly(ct), lead)))
    return out


def _apply_d_tensor(
    algebra: OreAlgebra, vec: list[RatFunc], ra: list[RatFunc], rb: list[RatFunc], pa: int, pb: int
) -> list[RatFunc]:
    entries: dict[tuple[int, int], RatFunc] = {}

    def add(key: tuple[int, int], val: RatFunc) -> None:
        if rf_is_zero(val):
            return
        entries[key] = rf_add(entries.get(key, rf_zero()), val)

    for i in range(pa):
        for j in range(pb):
            c = vec[i * pb + j]
            if rf_is_zero(c):
                continue
            if algebra.name == "differential":
                add((i, j), _delta_rf(algebra, c))
                add((i + 1, j), c)
                add((i, j + 1), c)
            elif algebra.name == "shift":
                add((i + 1, j + 1), _sigma_rf(algebra, c))
            else:
                raise ValueError(f"unsupported algebra {algebra.name!r}")
    # Reduce the i == pa boundary, then the j == pb boundary.
    after_i: dict[tuple[int, int], RatFunc] = {}
    for (i, j), c in entries.items():
        if i == pa:
            for t in range(pa):
                key = (t, j)
                after_i[key] = rf_add(after_i.get(key, rf_zero()), rf_mul(c, ra[t]))
        else:
            after_i[(i, j)] = rf_add(after_i.get((i, j), rf_zero()), c)
    out: dict[tuple[int, int], RatFunc] = {}
    for (i, j), c in after_i.items():
        if j == pb:
            for s in range(pb):
                key = (i, s)
                out[key] = rf_add(out.get(key, rf_zero()), rf_mul(c, rb[s]))
        else:
            out[(i, j)] = rf_add(out.get((i, j), rf_zero()), c)
    result = [rf_zero()] * (pa * pb)
    for (i, j), c in out.items():
        result[i * pb + j] = c
    return result


def symmetric_product(a: OrePolynomial, b: OrePolynomial) -> OrePolynomial:
    """Annihilator of the product ``f g`` (differential) / ``a_n b_n`` (shift).

    Builds the derivative-closed tensor basis ``{x^{(i)} y^{(j)} : i < ord a, j < ord b}``,
    applies ``d`` to the product (Leibniz for the differential algebra, the multiplicative
    ``S(fg) = (Sf)(Sg)`` for the shift algebra), and finds the minimal polynomial relation
    among the successive images -- an operator of order ``<= (ord a)(ord b)``.
    """
    if a.algebra.name != b.algebra.name:
        raise ValueError("operators must share an Ore algebra")
    algebra = a.algebra
    pa, pb = a.order, b.order
    if pa < 1 or pb < 1:
        raise ValueError("symmetric_product needs order >= 1 operators")
    ra = _tensor_reductions(a)
    rb = _tensor_reductions(b)
    vec = [rf_zero()] * (pa * pb)
    vec[0] = rf_from_rational(1)
    columns = [vec]
    cur = vec
    for _ in range(pa * pb):
        cur = _apply_d_tensor(algebra, cur, ra, rb, pa, pb)
        columns.append(cur)
    rel = find_poly_relation(columns, max_degree=8)
    if rel is None:
        raise ValueError("no symmetric-product annihilator found within the degree bound")
    return _op_from_polys(algebra, rel)


__all__ = [
    "OreDivision",
    "gcrd",
    "lclm",
    "ore_divmod",
    "symmetric_product",
]
