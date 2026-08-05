# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Creative telescoping: the annihilating recurrence of a parametrised sum.

Given a summand ``F(n, k)`` (a callable returning an exact :class:`~fractions.Fraction`,
zero outside a finite ``k``-support), the definite sum ``f(n) = sum_k F(n, k)`` is
P-recursive whenever ``F`` is proper hypergeometric (Zeilberger's theorem). We recover its
recurrence in the omnibias *guess-then-verify* style:

* **guess** the minimal annihilator ``L`` of ``f`` from an exact prefix
  (:func:`omnibias.holonomic._core.guess.guess_recurrence` -- exact rational null space);
* **verify** ``L[f](n) = 0`` on a range by exact evaluation (the residual is exactly zero
  by construction of the null space, and re-checked here).

The recurrence is the *creative-telescoping* output; that ``L`` continues to annihilate
``f`` for **all** ``n`` (rather than only the fitted range) is the holonomic-continuation
obligation discharged, per-point, by :mod:`omnibias.holonomic._core.certify`. The honesty
label is therefore ``guessed (holonomic ansatz), verified on range`` -- distinct from the
fully rigorous, unconditional :func:`gosper_definite_sum` below.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from fractions import Fraction
from math import lcm

from omnibias.holonomic._core import poly2 as P2
from omnibias.holonomic._core.gosper import gosper_sum
from omnibias.holonomic._core.guess import guess_recurrence
from omnibias.holonomic._core.hyperterm import ProperTerm
from omnibias.holonomic._core.linalg import null_space, solve_exact
from omnibias.holonomic._core.ore import OrePolynomial, shift_algebra
from omnibias.holonomic._core.poly2 import Poly2, Rat2
from omnibias.holonomic._core.rational_poly import Poly, peval, to_poly

Summand = Callable[[int, int], Fraction | int]
KBound = Callable[[int], tuple[int, int]]


def _default_k_bound(n: int) -> tuple[int, int]:
    return (0, n)


def summand_sum(summand: Summand, n: int, k_bound: KBound) -> Fraction:
    """Exact ``sum_{k=lo}^{hi} F(n, k)`` over the (inclusive) support ``k_bound(n)``."""
    lo, hi = k_bound(n)
    total = Fraction(0)
    for k in range(lo, hi + 1):
        total += Fraction(summand(n, k))
    return total


@dataclass(frozen=True)
class Telescoper:
    """A creative-telescoping result: the sum values and their guessed annihilator."""

    name: str
    values: tuple[Fraction, ...]
    recurrence: OrePolynomial

    @property
    def order(self) -> int:
        """Order of the annihilating recurrence ``L``."""
        return int(self.recurrence.order)

    def residual(self, n: int) -> Fraction:
        """``L[f](n) = sum_i c_i(n) f(n+i)`` (exactly zero where fitted)."""
        acc = Fraction(0)
        for i, c in enumerate(self.recurrence.coeffs):
            if c:
                acc += peval(c, n) * self.values[n + i]
        return acc

    def max_residual(self, upto: int | None = None) -> Fraction:
        """Largest ``|L[f](n)|`` over the verifiable range (``0`` iff satisfied)."""
        hi = (len(self.values) if upto is None else upto) - self.order
        return max((abs(self.residual(n)) for n in range(hi)), default=Fraction(0))


def creative_telescoping(
    summand: Summand,
    *,
    name: str = "sum",
    n_max: int = 16,
    k_bound: KBound | None = None,
    max_order: int = 4,
    max_index_degree: int = 3,
) -> Telescoper:
    """Guess (and range-verify) the P-recursive annihilator of ``f(n) = sum_k F(n, k)``.

    Raises ``ValueError`` if no recurrence within the search bounds fits the sum prefix.
    """
    kb = k_bound or _default_k_bound
    values = tuple(summand_sum(summand, n, kb) for n in range(n_max + 1))
    op = guess_recurrence(values, max_order=max_order, max_index_degree=max_index_degree)
    if op is None:
        raise ValueError(
            f"no P-recursive annihilator for '{name}' within order<={max_order}, "
            f"degree<={max_index_degree}"
        )
    return Telescoper(name=name, values=values, recurrence=op)


def gosper_definite_sum(
    num: Poly, den: Poly, term0: Fraction | int, a: int, b: int
) -> Fraction | None:
    r"""Rigorous definite hypergeometric sum ``sum_{k=a}^{b-1} t(k)`` via Gosper.

    ``t`` is the hypergeometric term with ``t(k+1)/t(k) = num(k)/den(k)`` and ``t(a) =
    term0``. When Gosper finds an antidifference ``T = R t`` the sum telescopes exactly to
    ``T(b) - T(a) = R(b) t(b) - R(a) t(a)`` -- returned as an exact rational. Returns
    ``None`` (a genuine finding, not a guess) when the term is **not** Gosper-summable.

    This is unconditional and closed-form: no fitting, no range conjecture.
    """
    result = gosper_sum(num, den)
    if not result.summable:
        return None
    if b == a:
        return Fraction(0)
    # Build t(k) forward from t(a) = term0 using the ratio; T(k) = R(k) t(k).
    t = {a: Fraction(term0)}
    for k in range(a, b):
        d = peval(den, k)
        if d == 0:
            raise ValueError(f"term ratio denominator vanishes at k={k}; shift the lower limit")
        t[k + 1] = t[k] * peval(num, k) / d
    return Fraction(result.certificate(b) * t[b] - result.certificate(a) * t[a])


# --------------------------------------------------------------------------- #
# True Zeilberger creative telescoping + WZ certificates (exact, all-n).
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ZeilbergerCertificate:
    r"""An exact creative-telescoping certificate proving ``L`` for **all** ``n``.

    The telescoper ``L = sum_i c_i(n) S^i`` and the rational cofactor ``R(n, k)`` satisfy
    the term-level identity

    .. math::

        \sum_i c_i(n)\, F(n+i, k) = R(n, k+1) F(n, k+1) - R(n, k) F(n, k),

    a **bivariate rational-function identity** verified exactly (:meth:`verify_symbolic`).
    Summed over the finite ``k``-support it collapses to ``L[f](n) = 0`` for every ``n`` --
    not merely a fitted range. :attr:`grid_degree_n` / :attr:`grid_degree_k` bound the
    polynomial degree of the cross-multiplied identity, so vanishing on a
    ``(grid_degree_n+1) x (grid_degree_k+1)`` integer grid re-proves it for all ``n, k``
    (the finite obligation shipped to Lean by :mod:`.certify`).
    """

    term: ProperTerm
    recurrence: OrePolynomial
    coeffs: tuple[Poly, ...]
    certificate: Rat2
    grid_degree_n: int
    grid_degree_k: int

    @property
    def order(self) -> int:
        """Order of the telescoper ``L``."""
        return int(self.recurrence.order)

    def cofactor(self, n: int, k: int) -> Fraction:
        """Evaluate the certificate cofactor ``R(n, k)`` (``ValueError`` on a pole)."""
        den = Fraction(P2.p2_eval(self.certificate[1], n, k))
        if den == 0:
            raise ValueError(f"certificate cofactor has a pole at (n, k) = ({n}, {k})")
        return Fraction(P2.p2_eval(self.certificate[0], n, k)) / den

    def relation_residual(self, n: int, k: int) -> Fraction:
        r"""``sum_i c_i(n) F(n+i,k) - [R(n,k+1)F(n,k+1) - R(n,k)F(n,k)]`` (exactly 0).

        Raises ``ValueError`` at a certificate pole; use pole-free grid points.
        """
        lhs = sum(
            (peval(c, n) * self.term.value(n + i, k) for i, c in enumerate(self.coeffs)),
            Fraction(0),
        )
        gk1 = self.cofactor(n, k + 1) * self.term.value(n, k + 1)
        gk = self.cofactor(n, k) * self.term.value(n, k)
        return Fraction(lhs - (gk1 - gk))

    def annihilation_residual(self, n: int, k_lo: int, k_hi: int) -> Fraction:
        r"""``L[f](n) = sum_i c_i(n) sum_{k} F(n+i, k)`` over the support (exactly 0)."""
        acc = Fraction(0)
        for i, c in enumerate(self.coeffs):
            acc += peval(c, n) * self.term.sum_over_k(n + i, k_lo, k_hi)
        return Fraction(acc)

    def verify_symbolic(self) -> bool:
        """Exactly verify the bivariate rational-function telescoping identity (all ``n``)."""
        lhs = _lhs_rat(self.term, self.coeffs)
        rhs = P2.r2_sub(P2.r2_mul(P2.r2_shift_k(self.certificate, 1), self.term.ratio_k()), self.certificate)
        return bool(P2.r2_equal(lhs, rhs))

    def identity_terms(self, n0: int, k0: int) -> list[Fraction]:
        r"""Signed terms of the *pole-free* cross-multiplied identity ``N(n0, k0)``.

        ``N`` is the numerator of ``sum_i c_i(n) F(n+i,k) - [ΔR](n,k)`` over the common
        denominator; it is identically zero, so ``sum`` of the returned terms is ``0`` at
        every ``(n0, k0)``. Vanishing on a ``(grid_degree_n+1) x (grid_degree_k+1)`` integer
        grid re-proves ``N ≡ 0`` (hence ``L[f] = 0`` for all ``n``) -- the finite Lean
        obligation, evaluated purely from polynomials (no cofactor poles).
        """
        d = len(self.coeffs) - 1
        ps = _p_ratios(self.term, d)
        p_num = [p[0] for p in ps]
        p_den = [p[1] for p in ps]
        bn, bd = self.term.ratio_k()
        cert_p, w = self.certificate
        wk = P2.p2_shift_k(w, 1)
        prod_pid = P2.p2_const(1)
        for pd in p_den:
            prod_pid = P2.p2_mul(prod_pid, pd)
        wwkbd = P2.p2_mul(P2.p2_mul(w, wk), bd)
        terms: list[Fraction] = []
        for i, c in enumerate(self.coeffs):
            rest = P2.p2_const(1)
            for j in range(d + 1):
                if j != i:
                    rest = P2.p2_mul(rest, p_den[j])
            block = P2.p2_mul(P2.p2_mul(p_num[i], rest), wwkbd)
            terms.append(Fraction(peval(c, n0)) * Fraction(P2.p2_eval(block, n0, k0)))
        r_term = P2.p2_mul(P2.p2_mul(prod_pid, wk), bd)
        terms.append(Fraction(P2.p2_eval(cert_p, n0, k0)) * Fraction(P2.p2_eval(r_term, n0, k0)))
        sr_term = P2.p2_mul(P2.p2_mul(bn, prod_pid), w)
        terms.append(
            -Fraction(P2.p2_eval(P2.p2_shift_k(cert_p, 1), n0, k0)) * Fraction(P2.p2_eval(sr_term, n0, k0))
        )
        return terms


def _lhs_rat(term: ProperTerm, coeffs: tuple[Poly, ...]) -> Rat2:
    ps = _p_ratios(term, len(coeffs) - 1)
    acc: Rat2 = (P2.p2_zero(), P2.p2_const(1))
    for i, c in enumerate(coeffs):
        c_poly2: Poly2 = {(e, 0): v for e, v in enumerate(c) if v != 0}
        acc = P2.r2_add(acc, P2.r2_mul((c_poly2, P2.p2_const(1)), ps[i]))
    return acc


def _p_ratios(term: ProperTerm, d: int) -> list[Rat2]:
    """``p_i = F(n+i, k) / F(n, k)`` for ``i = 0..d`` as bivariate rational functions."""
    a_ratio = term.ratio_n()
    ps: list[Rat2] = [P2.r2_one()]
    cur = P2.r2_one()
    for t in range(d):
        cur = P2.r2_mul(cur, P2.r2_shift_n(a_ratio, t))
        ps.append(cur)
    return ps


def _mono(a: int, b: int) -> Poly2:
    return {(a, b): Fraction(1)}


def _denom_candidates(term: ProperTerm, d: int) -> list[Poly2]:
    """Certificate-denominator ansätze, smallest first (Gosper's cofactor lives here)."""
    ps = _p_ratios(term, d)
    _bn, bd = term.ratio_k()
    prod_pid = P2.p2_const(1)
    for p in ps:
        prod_pid = P2.p2_mul(prod_pid, p[1])
    cands = [P2.p2_const(1), bd, prod_pid, P2.p2_mul(bd, prod_pid)]
    seen: list[Poly2] = []
    for c in cands:
        if not P2.p2_is_zero(c) and all(not P2.r2_equal((c, P2.p2_const(1)), (s, P2.p2_const(1))) for s in seen):
            seen.append(c)
    return seen


def _clear_denoms(coeffs: list[Poly], cert_num: Poly2) -> tuple[list[Poly], Poly2]:
    dens = [v.denominator for c in coeffs for v in c]
    dens += [v.denominator for v in cert_num.values()]
    scale = lcm(*dens) if dens else 1
    if scale == 1:
        return coeffs, cert_num
    s = Fraction(scale)
    new_c = [tuple(v * s for v in c) for c in coeffs]
    new_p = {kk: v * s for kk, v in cert_num.items()}
    return new_c, new_p


def _grid_degrees(term: ProperTerm, d: int, dc: int, dpn: int, dpk: int, w: Poly2) -> tuple[int, int]:
    """Conservative (upper-bound) degrees of the cross-multiplied identity ``N(n, k)``."""
    ps = _p_ratios(term, d)
    bn, bd = term.ratio_k()
    wk = P2.p2_shift_k(w, 1)
    prod_pid = P2.p2_const(1)
    for p in ps:
        prod_pid = P2.p2_mul(prod_pid, p[1])
    wwkbd = P2.p2_mul(P2.p2_mul(w, wk), bd)
    dn = dk = 0
    for i in range(d + 1):
        rest = P2.p2_const(1)
        for j in range(d + 1):
            if j != i:
                rest = P2.p2_mul(rest, ps[j][1])
        block = P2.p2_mul(P2.p2_mul(ps[i][0], rest), wwkbd)
        dn = max(dn, P2.p2_degree_n(block) + dc)
        dk = max(dk, P2.p2_degree_k(block))
    r_block = P2.p2_mul(P2.p2_mul(prod_pid, wk), bd)
    dn = max(dn, P2.p2_degree_n(r_block) + dpn)
    dk = max(dk, P2.p2_degree_k(r_block) + dpk)
    s_block = P2.p2_mul(P2.p2_mul(bn, prod_pid), w)
    dn = max(dn, P2.p2_degree_n(s_block) + dpn)
    dk = max(dk, P2.p2_degree_k(s_block) + dpk)
    return dn, dk


def _build_system(
    term: ProperTerm, d: int, dc: int, dpn: int, dpk: int, w: Poly2
) -> tuple[list[tuple[int, int]], list[tuple[int, int]], dict[tuple[int, int], Poly2], dict[tuple[int, int], Poly2]]:
    ps = _p_ratios(term, d)
    pin = [p[0] for p in ps]
    pid = [p[1] for p in ps]
    bn, bd = term.ratio_k()
    wk = P2.p2_shift_k(w, 1)
    prod_pid = P2.p2_const(1)
    for p in pid:
        prod_pid = P2.p2_mul(prod_pid, p)
    wwkbd = P2.p2_mul(P2.p2_mul(w, wk), bd)

    contrib_c: dict[tuple[int, int], Poly2] = {}
    for i in range(d + 1):
        rest = P2.p2_const(1)
        for j in range(d + 1):
            if j != i:
                rest = P2.p2_mul(rest, pid[j])
        base_i = P2.p2_mul(P2.p2_mul(pin[i], rest), wwkbd)
        for e in range(dc + 1):
            contrib_c[(i, e)] = P2.p2_mul(_mono(e, 0), base_i)

    r_mult = P2.p2_mul(P2.p2_mul(prod_pid, wk), bd)
    shiftr_mult = P2.p2_mul(P2.p2_mul(bn, prod_pid), w)
    contrib_p: dict[tuple[int, int], Poly2] = {}
    for a in range(dpn + 1):
        for b in range(dpk + 1):
            mon = _mono(a, b)
            term1 = P2.p2_mul(mon, r_mult)
            term2 = P2.p2_mul(P2.p2_shift_k(mon, 1), shiftr_mult)
            contrib_p[(a, b)] = P2.p2_sub(term1, term2)

    c_keys = [(i, e) for i in range(d + 1) for e in range(dc + 1)]
    p_keys = [(a, b) for a in range(dpn + 1) for b in range(dpk + 1)]
    return c_keys, p_keys, contrib_c, contrib_p


def _assemble(
    contribs: list[Poly2],
) -> tuple[list[tuple[int, int]], list[list[Fraction]]]:
    monos: set[tuple[int, int]] = set()
    for cp in contribs:
        monos.update(cp.keys())
    ordered = sorted(monos)
    rows = [[cp.get(m, Fraction(0)) for cp in contribs] for m in ordered]
    return ordered, rows


def _finish(
    term: ProperTerm,
    d: int,
    dpn: int,
    dpk: int,
    w: Poly2,
    c_polys: list[Poly],
    cert_num: Poly2,
    *,
    normalize: bool = True,
) -> ZeilbergerCertificate | None:
    # Trim trailing zero telescoper coefficients.
    while len(c_polys) > 1 and not any(v != 0 for v in c_polys[-1]):
        c_polys = c_polys[:-1]
    if normalize:
        # Clearing denominators scales L and R by the same constant, preserving the
        # identity -- but it changes L, so the fixed-normalisation WZ path opts out.
        c_polys, cert_num = _clear_denoms(c_polys, cert_num)
    coeffs = tuple(to_poly(list(c)) for c in c_polys)
    recurrence = shift_algebra().operator([list(c) for c in coeffs])
    certificate: Rat2 = (cert_num, w)
    dc_actual = max((len(c) - 1 for c in coeffs), default=0)
    dpn_actual = max((a for (a, _b) in cert_num), default=0)
    dpk_actual = max((b for (_a, b) in cert_num), default=0)
    dn, dk = _grid_degrees(term, d, dc_actual, dpn_actual, dpk_actual, w)
    result = ZeilbergerCertificate(
        term=term,
        recurrence=recurrence,
        coeffs=coeffs,
        certificate=certificate,
        grid_degree_n=dn,
        grid_degree_k=dk,
    )
    return result if result.verify_symbolic() else None


def zeilberger(
    term: ProperTerm, *, max_order: int = 4, max_c_degree: int = 3, max_cert_degree: int = 4
) -> ZeilbergerCertificate:
    r"""Exact creative telescoping of a proper hypergeometric term (true Zeilberger).

    Solves for the minimal-order telescoper ``L`` and the rational certificate ``R``
    simultaneously by an exact rational null space, so it needs no guesser and handles
    degenerate sums (e.g. ``sum_k (-1)^k C(n,k) = [n=0]``) natively. Raises ``ValueError``
    if no certificate is found within the search bounds.
    """
    for d in range(1, max_order + 1):
        for w in _denom_candidates(term, d):
            for dpk in range(max_cert_degree + 1):
                for dpn in range(max_cert_degree + 1):
                    c_keys, p_keys, contrib_c, contrib_p = _build_system(
                        term, d, max_c_degree, dpn, dpk, w
                    )
                    contribs = [contrib_c[k] for k in c_keys] + [contrib_p[k] for k in p_keys]
                    _monos, rows = _assemble(contribs)
                    basis = null_space(rows)
                    n_c = len(c_keys)
                    for sol in basis:
                        if not any(sol[i] != 0 for i in range(n_c)):
                            continue
                        c_polys = _extract_c(sol, d, max_c_degree)
                        cert_num = _extract_p(sol[n_c:], p_keys)
                        found = _finish(term, d, dpn, dpk, w, c_polys, cert_num)
                        if found is not None:
                            return found
    raise ValueError("no Zeilberger certificate found within the search bounds")


def _extract_c(sol: list[Fraction], d: int, dc: int) -> list[Poly]:
    out: list[Poly] = []
    idx = 0
    for _i in range(d + 1):
        coeffs = [sol[idx + e] for e in range(dc + 1)]
        idx += dc + 1
        out.append(to_poly(coeffs))
    return out


def _extract_p(psol: list[Fraction], p_keys: list[tuple[int, int]]) -> Poly2:
    out: Poly2 = {}
    for val, key in zip(psol, p_keys, strict=True):
        if val != 0:
            out[key] = val
    return out


def wz_certificate(term: ProperTerm, *, max_cert_degree: int = 5) -> Rat2:
    r"""The WZ cofactor ``R(n, k)`` for a constant sum ``sum_k F(n, k) = const``.

    Uses the fixed telescoper ``F(n+1, k) - F(n, k) = R(n,k+1)F(n,k+1) - R(n,k)F(n,k)``
    (``c = [-1, 1]``) and solves for ``R`` exactly. Raises ``ValueError`` if ``term`` is
    not WZ-summable within the degree bound.
    """
    coeffs: tuple[Poly, ...] = ((Fraction(-1),), (Fraction(1),))
    cert = _solve_fixed_c(term, coeffs, max_cert_degree)
    if cert is None:
        raise ValueError("no WZ certificate found within the degree bound")
    return cert.certificate


def wz_pair(term: ProperTerm, *, max_cert_degree: int = 5) -> tuple[Callable[[int, int], Fraction], Rat2]:
    r"""Return ``(G, R)`` where ``G(n, k) = R(n, k) F(n, k)`` is the WZ companion."""
    cert = wz_certificate(term, max_cert_degree=max_cert_degree)
    num, den = cert

    def g(n: int, k: int) -> Fraction:
        d = Fraction(P2.p2_eval(den, n, k))
        if d == 0:
            raise ValueError(f"WZ companion pole at (n, k) = ({n}, {k})")
        return Fraction(Fraction(P2.p2_eval(num, n, k)) / d * term.value(n, k))

    return g, cert


def _solve_fixed_c(
    term: ProperTerm, coeffs: tuple[Poly, ...], max_cert_degree: int
) -> ZeilbergerCertificate | None:
    d = len(coeffs) - 1
    for w in _denom_candidates(term, d):
        for dpk in range(max_cert_degree + 1):
            for dpn in range(max_cert_degree + 1):
                _c_keys, p_keys, contrib_c, contrib_p = _build_system(
                    term, d, max(len(c) - 1 for c in coeffs), dpn, dpk, w
                )
                dc = max(len(c) - 1 for c in coeffs)
                known: Poly2 = {}
                for i, c in enumerate(coeffs):
                    for e in range(dc + 1):
                        cval = c[e] if e < len(c) else Fraction(0)
                        if cval != 0:
                            known = P2.p2_add(known, P2.p2_scale(contrib_c[(i, e)], cval))
                contribs = [contrib_p[k] for k in p_keys]
                monos, rows = _assemble([*contribs, known])
                a_rows = [row[:-1] for row in rows]
                b_rhs = [-row[-1] for row in rows]
                sol = solve_exact(a_rows, b_rhs)
                if sol is None:
                    continue
                cert_num = _extract_p(sol, p_keys)
                found = _finish(
                    term, d, dpn, dpk, w, [list(c) for c in coeffs], cert_num, normalize=False
                )
                if found is not None:
                    return found
    return None


__all__ = [
    "Summand",
    "Telescoper",
    "ZeilbergerCertificate",
    "creative_telescoping",
    "gosper_definite_sum",
    "summand_sum",
    "wz_certificate",
    "wz_pair",
    "zeilberger",
]
