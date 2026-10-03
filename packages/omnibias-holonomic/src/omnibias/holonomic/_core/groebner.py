# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Exact-``Q`` Groebner bases over :class:`~omnibias.holonomic._core.poly_n.PolyN`.

Buchberger's algorithm, both classical pair-skipping criteria (coprime leading
monomials; the LCM chain criterion), a genuine cofactor witness for ideal
membership (``f == sum(cofactor_i * generator_i) + remainder`` exactly, so a
caller never has to trust ``is_member`` without replaying the identity), and a
Rabinowitsch radical-membership test built on top of it.

Buchberger's algorithm is doubly exponential in the worst case. Every entry
point here takes a :class:`GroebnerBudget` and raises
:class:`GroebnerBudgetExceeded` -- loudly, not silently truncating -- once the
declared pair count, polynomial count, term count, or degree is exceeded. A
budget refusal is an engineering limit, not a mathematical statement that no
Groebner basis exists.

Search (which pairs to reduce, in which order) is entirely separate from the
witness that gets sealed: :func:`ideal_member` and :func:`radical_member`
return an explicit cofactor combination that :func:`verify_ideal_membership`
replays by plain polynomial arithmetic, without rerunning Buchberger.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import Literal

from omnibias.holonomic._core.poly_n import Monomial, PolyN

MonomialOrderName = Literal["lex", "grlex", "degrevlex"]

_ORDER_NAMES: frozenset[str] = frozenset({"lex", "grlex", "degrevlex"})


class GroebnerBudgetExceeded(RuntimeError):
    """A Buchberger computation stopped; this is an engineering limit, not a
    mathematical statement that the ideal has no finite Groebner basis."""


@dataclass(frozen=True)
class GroebnerBudget:
    """Hard limits a Buchberger run refuses to exceed."""

    max_pairs: int = 20_000
    max_polynomials: int = 2_000
    max_terms: int = 20_000
    max_degree: int = 64

    def __post_init__(self) -> None:
        values = (self.max_pairs, self.max_polynomials, self.max_terms, self.max_degree)
        if any(type(v) is not int or v < 1 for v in values):
            raise ValueError("Groebner budgets must be positive integers")


DEFAULT_GROEBNER_BUDGET = GroebnerBudget()


def _monomial_key(order: MonomialOrderName, monomial: Monomial) -> tuple[int, ...]:
    """An ascending Python-sortable key: ``key(a) < key(b)`` iff ``a`` is smaller."""
    if order == "lex":
        return monomial
    if order == "grlex":
        return (sum(monomial), *monomial)
    if order == "degrevlex":
        # Same total degree: a >_degrevlex b iff the rightmost nonzero entry of
        # a-b is negative. Negating the reversed tuple turns that into a plain
        # ascending lexicographic comparison.
        return (sum(monomial), *(-e for e in reversed(monomial)))
    raise ValueError(f"unknown monomial order: {order!r}")


def _divides(a: Monomial, b: Monomial) -> bool:
    """``True`` iff monomial ``a`` divides monomial ``b`` (componentwise ``<=``)."""
    return all(x <= y for x, y in zip(a, b, strict=True))


def _coprime(a: Monomial, b: Monomial) -> bool:
    return all(x == 0 or y == 0 for x, y in zip(a, b, strict=True))


class MonomialOrder:
    """A fixed monomial order (``lex``, ``grlex``, or ``degrevlex``)."""

    __slots__ = ("name",)

    def __init__(self, name: MonomialOrderName = "degrevlex") -> None:
        if name not in _ORDER_NAMES:
            raise ValueError(f"unknown monomial order: {name!r}")
        self.name = name

    def key(self, monomial: Monomial) -> tuple[int, ...]:
        return _monomial_key(self.name, monomial)

    def compare(self, a: Monomial, b: Monomial) -> int:
        ka, kb = self.key(a), self.key(b)
        return -1 if ka < kb else (1 if ka > kb else 0)

    def leading_monomial(self, poly: PolyN) -> Monomial | None:
        if poly.is_zero():
            return None
        return max(poly.terms, key=self.key)

    def leading_term_data(self, poly: PolyN) -> tuple[Monomial, Fraction]:
        lm = self.leading_monomial(poly)
        if lm is None:
            raise ValueError("the zero polynomial has no leading term")
        return lm, poly.terms[lm]

    def leading_term(self, poly: PolyN) -> PolyN:
        lm, lc = self.leading_term_data(poly)
        return PolyN(poly.nvars, {lm: lc})


def _check_poly_budget(poly: PolyN, budget: GroebnerBudget) -> None:
    if len(poly.terms) > budget.max_terms:
        raise GroebnerBudgetExceeded(
            f"a polynomial exceeded {budget.max_terms} terms during Buchberger"
        )
    degree = max((sum(mon) for mon in poly.terms), default=0)
    if degree > budget.max_degree:
        raise GroebnerBudgetExceeded(
            f"a polynomial exceeded degree {budget.max_degree} during Buchberger"
        )


def _check_basis_size(size: int, budget: GroebnerBudget) -> None:
    if size > budget.max_polynomials:
        raise GroebnerBudgetExceeded(
            f"the basis exceeded {budget.max_polynomials} polynomials during Buchberger"
        )


def _divide_plain(
    f: PolyN,
    divisors: Sequence[PolyN],
    order: MonomialOrder,
) -> tuple[tuple[PolyN, ...], PolyN]:
    """Multivariate division: ``f == sum(quotients[i]*divisors[i]) + remainder``."""
    nvars = f.nvars
    quotients = [PolyN.zero(nvars) for _ in divisors]
    leads = [order.leading_monomial(d) for d in divisors]
    lead_coeffs = [
        (d.terms[lm] if lm is not None else None) for d, lm in zip(divisors, leads, strict=True)
    ]
    p = f
    remainder = PolyN.zero(nvars)
    while not p.is_zero():
        lm_p, lc_p = order.leading_term_data(p)
        divided = False
        for idx, (d, lm_d, lc_d) in enumerate(zip(divisors, leads, lead_coeffs, strict=True)):
            if lm_d is None or not _divides(lm_d, lm_p):
                continue
            factor_mon = tuple(a - b for a, b in zip(lm_p, lm_d, strict=True))
            factor = PolyN(nvars, {factor_mon: lc_p / lc_d})
            quotients[idx] = quotients[idx] + factor
            p = p - factor * d
            divided = True
            break
        if not divided:
            lead = PolyN(nvars, {lm_p: lc_p})
            remainder = remainder + lead
            p = p - lead
    return tuple(quotients), remainder


@dataclass(frozen=True)
class _TrackedPoly:
    """A polynomial together with its exact combination in the original generators."""

    poly: PolyN
    combo: tuple[PolyN, ...]


def _s_poly_tracked(a: _TrackedPoly, b: _TrackedPoly, order: MonomialOrder) -> _TrackedPoly:
    lm_a, lc_a = order.leading_term_data(a.poly)
    lm_b, lc_b = order.leading_term_data(b.poly)
    lcm_mon = tuple(max(x, y) for x, y in zip(lm_a, lm_b, strict=True))
    nvars = a.poly.nvars
    fa = PolyN(nvars, {tuple(m - x for m, x in zip(lcm_mon, lm_a, strict=True)): Fraction(1) / lc_a})
    fb = PolyN(nvars, {tuple(m - x for m, x in zip(lcm_mon, lm_b, strict=True)): Fraction(1) / lc_b})
    poly = fa * a.poly - fb * b.poly
    combo = tuple(fa * ca - fb * cb for ca, cb in zip(a.combo, b.combo, strict=True))
    return _TrackedPoly(poly, combo)


def _reduce_tracked(
    target: _TrackedPoly,
    basis: Sequence[_TrackedPoly],
    order: MonomialOrder,
) -> _TrackedPoly:
    """Reduce ``target`` modulo ``basis``; ``target`` must already satisfy
    ``target.poly == sum(target.combo[i] * original[i])``."""
    if not basis:
        return target
    quotients, remainder = _divide_plain(target.poly, [b.poly for b in basis], order)
    nvars = target.poly.nvars
    n_gen = len(target.combo)
    subtracted = [PolyN.zero(nvars) for _ in range(n_gen)]
    for idx, q in enumerate(quotients):
        if q.is_zero():
            continue
        for j in range(n_gen):
            comb_j = basis[idx].combo[j]
            if not comb_j.is_zero():
                subtracted[j] = subtracted[j] + q * comb_j
    combo = tuple(tc - sc for tc, sc in zip(target.combo, subtracted, strict=True))
    return _TrackedPoly(remainder, combo)


def _chain_criterion_skips(
    basis: Sequence[_TrackedPoly],
    i: int,
    j: int,
    lcm_ij: Monomial,
    pending: Sequence[tuple[int, int]],
    order: MonomialOrder,
) -> bool:
    """Buchberger's second (chain / LCM) criterion."""
    pending_set = set(pending)
    for k in range(len(basis)):
        if k in (i, j):
            continue
        lm_k = order.leading_monomial(basis[k].poly)
        if lm_k is None or not _divides(lm_k, lcm_ij):
            continue
        pik = (min(i, k), max(i, k))
        pjk = (min(j, k), max(j, k))
        if pik not in pending_set and pjk not in pending_set:
            return True
    return False


def _buchberger_tracked(
    generators: Sequence[PolyN],
    order: MonomialOrder,
    budget: GroebnerBudget,
) -> list[_TrackedPoly]:
    if not generators:
        return []
    nvars = generators[0].nvars
    if any(g.nvars != nvars for g in generators):
        raise ValueError("all generators must share the same number of variables")
    for g in generators:
        _check_poly_budget(g, budget)
    n_gen = len(generators)
    basis: list[_TrackedPoly] = []
    for i, g in enumerate(generators):
        if g.is_zero():
            continue
        combo = tuple(
            PolyN.const(nvars, 1) if j == i else PolyN.zero(nvars) for j in range(n_gen)
        )
        basis.append(_TrackedPoly(g, combo))
    _check_basis_size(len(basis), budget)
    pairs: list[tuple[int, int]] = [
        (i, j) for i in range(len(basis)) for j in range(i + 1, len(basis))
    ]
    processed = 0
    while pairs:
        i, j = pairs.pop()
        processed += 1
        if processed > budget.max_pairs:
            raise GroebnerBudgetExceeded(f"exceeded {budget.max_pairs} S-polynomial pairs")
        lm_i = order.leading_monomial(basis[i].poly)
        lm_j = order.leading_monomial(basis[j].poly)
        if lm_i is None or lm_j is None:
            continue
        if _coprime(lm_i, lm_j):
            continue
        lcm_ij = tuple(max(a, b) for a, b in zip(lm_i, lm_j, strict=True))
        if _chain_criterion_skips(basis, i, j, lcm_ij, pairs, order):
            continue
        s = _s_poly_tracked(basis[i], basis[j], order)
        reduced = _reduce_tracked(s, basis, order)
        if reduced.poly.is_zero():
            continue
        _check_poly_budget(reduced.poly, budget)
        basis.append(reduced)
        new_index = len(basis) - 1
        _check_basis_size(len(basis), budget)
        for k in range(new_index):
            pairs.append((k, new_index))
    return basis


def buchberger(
    generators: Sequence[PolyN],
    order: MonomialOrderName = "degrevlex",
    *,
    budget: GroebnerBudget = DEFAULT_GROEBNER_BUDGET,
) -> tuple[PolyN, ...]:
    """A (generally non-reduced, non-minimal) Groebner basis of ``(generators)``.

    Uses both classical Buchberger pair-skipping criteria. Raises
    :class:`GroebnerBudgetExceeded` rather than running unbounded.
    """
    mo = MonomialOrder(order)
    tracked = _buchberger_tracked(generators, mo, budget)
    return tuple(tp.poly for tp in tracked if not tp.poly.is_zero())


def reduced_groebner_basis(
    generators: Sequence[PolyN],
    order: MonomialOrderName = "degrevlex",
    *,
    budget: GroebnerBudget = DEFAULT_GROEBNER_BUDGET,
) -> tuple[PolyN, ...]:
    """The monic, minimal, fully interreduced Groebner basis of ``(generators)``."""
    mo = MonomialOrder(order)
    raw = [p for p in buchberger(generators, order, budget=budget) if not p.is_zero()]
    if not raw:
        return ()
    monic = []
    for g in raw:
        _, lc = mo.leading_term_data(g)
        monic.append(g * (Fraction(1) / lc))
    leading = [mo.leading_monomial(g) for g in monic]
    minimal: list[PolyN] = []
    for i, g in enumerate(monic):
        li = leading[i]
        dominated = any(
            j != i
            and leading[j] is not None
            and li is not None
            and _divides(leading[j], li)
            and (leading[j] != li or j < i)
            for j in range(len(monic))
        )
        if not dominated:
            minimal.append(g)
    reduced = list(minimal)
    for _ in range(len(reduced) + 2):
        changed = False
        for i in range(len(reduced)):
            others = reduced[:i] + reduced[i + 1 :]
            if not others:
                continue
            _, remainder = _divide_plain(reduced[i], others, mo)
            if remainder != reduced[i]:
                if remainder.is_zero():
                    raise ArithmeticError(
                        "a Groebner basis element reduced to zero during "
                        "interreduction; the generating set was inconsistent"
                    )
                _, lc = mo.leading_term_data(remainder)
                reduced[i] = remainder * (Fraction(1) / lc)
                changed = True
        if not changed:
            break
    return tuple(sorted(reduced, key=lambda g: mo.key(mo.leading_monomial(g))))


@dataclass(frozen=True)
class IdealMembership:
    """A membership verdict with an exact cofactor witness.

    ``f == sum(cofactors[i] * generators[i]) + remainder`` holds exactly over
    ``Q`` by construction; :func:`verify_ideal_membership` replays that
    identity without rerunning Buchberger. ``is_member`` is ``True`` iff
    ``remainder`` is the zero polynomial.
    """

    is_member: bool
    remainder: PolyN
    cofactors: tuple[PolyN, ...]
    basis: tuple[PolyN, ...]


def ideal_member(
    f: PolyN,
    generators: Sequence[PolyN],
    order: MonomialOrderName = "degrevlex",
    *,
    budget: GroebnerBudget = DEFAULT_GROEBNER_BUDGET,
) -> IdealMembership:
    """Decide ``f in (generators)`` and return an exact cofactor witness."""
    if not generators:
        return IdealMembership(f.is_zero(), f, (), ())
    nvars = generators[0].nvars
    if f.nvars != nvars or any(g.nvars != nvars for g in generators):
        raise ValueError("f and every generator must share the same number of variables")
    mo = MonomialOrder(order)
    tracked_basis = _buchberger_tracked(generators, mo, budget)
    quotients, remainder = _divide_plain(f, [tp.poly for tp in tracked_basis], mo)
    n_gen = len(generators)
    cofactors = [PolyN.zero(nvars) for _ in range(n_gen)]
    for idx, q in enumerate(quotients):
        if q.is_zero():
            continue
        for j in range(n_gen):
            comb_j = tracked_basis[idx].combo[j]
            if not comb_j.is_zero():
                cofactors[j] = cofactors[j] + q * comb_j
    return IdealMembership(
        remainder.is_zero(), remainder, tuple(cofactors), tuple(tp.poly for tp in tracked_basis)
    )


def verify_ideal_membership(
    f: PolyN,
    generators: Sequence[PolyN],
    membership: IdealMembership,
) -> bool:
    """Replay ``f == sum(cofactors*generators) + remainder`` by plain arithmetic."""
    if len(membership.cofactors) != len(generators):
        return False
    reconstructed = membership.remainder
    for cofactor, g in zip(membership.cofactors, generators, strict=True):
        reconstructed = reconstructed + cofactor * g
    if reconstructed != f:
        return False
    return membership.is_member == membership.remainder.is_zero()


def radical_member(
    f: PolyN,
    generators: Sequence[PolyN],
    order: MonomialOrderName = "degrevlex",
    *,
    budget: GroebnerBudget = DEFAULT_GROEBNER_BUDGET,
) -> IdealMembership:
    """Rabinowitsch test: ``f in sqrt(generators)`` iff ``1 in (generators, 1-t*f)``.

    The returned witness lives in the Rabinowitsch-extended ring (one fresh
    variable ``t`` appended), not directly as cofactors of ``f`` over the
    original generators: radical membership is a strictly weaker fact than
    ideal membership and does not, in general, admit a witness in the
    original ring at all.
    """
    if not generators:
        raise ValueError("radical membership needs at least one generator")
    nvars = f.nvars
    if any(g.nvars != nvars for g in generators):
        raise ValueError("f and every generator must share the same number of variables")
    embedded_generators = [
        PolyN(nvars + 1, {(*mon, 0): coeff for mon, coeff in g.terms.items()}) for g in generators
    ]
    embedded_f = PolyN(nvars + 1, {(*mon, 0): coeff for mon, coeff in f.terms.items()})
    t = PolyN.var(nvars + 1, nvars)
    rabinowitsch = PolyN.const(nvars + 1, 1) - t * embedded_f
    system = [*embedded_generators, rabinowitsch]
    one = PolyN.const(nvars + 1, 1)
    return ideal_member(one, system, order, budget=budget)


__all__ = [
    "DEFAULT_GROEBNER_BUDGET",
    "GroebnerBudget",
    "GroebnerBudgetExceeded",
    "IdealMembership",
    "MonomialOrder",
    "MonomialOrderName",
    "buchberger",
    "ideal_member",
    "radical_member",
    "reduced_groebner_basis",
    "verify_ideal_membership",
]
