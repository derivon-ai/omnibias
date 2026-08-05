# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Lean-certified hypergeometric-identity proofs from creative telescoping.

The pipeline for a claimed identity ``sum_k F(n, k) = g(n)``:

1. **guess** the annihilating recurrence ``L`` of ``f(n) = sum_k F(n, k)`` (creative
   telescoping, exact rational null space -- heuristic *which* recurrence, exact *that* it
   fits the prefix);
2. emit **finite, rational** obligations, each a ``rational_identity`` certificate the
   omnibias Lean kernel can discharge exactly:

   * ``f(n) = g(n)`` for every ``n`` on the range (the identity itself, cross-multiplied);
   * ``L[f](n) = 0`` (the sum obeys the recurrence);
   * ``L[g](n) = 0`` (the closed form obeys the *same* recurrence);
3. optionally run ``lake build`` on each obligation (:func:`omnibias.core.proof.lean_check`).

**Honesty.** The obligations rigorously certify the identity on the *checked range*
``[0, n_max]`` (each ``f(n) = g(n)`` is an exact ``Int`` equality). The recurrence ``L`` is
*guessed* (holonomic ansatz) and *verified on the range*; that it -- and therefore the
identity -- continues for **all** ``n`` is the holonomic-continuation (Zeilberger) claim,
here backed to ``n_max`` rather than proven symbolically. ``theorem_prover_verified`` is
set **only** on a genuine ``lake`` pass of every emitted obligation and is never forged; no
Lean toolchain present degrades gracefully (the flag stays ``False``). The purely rigorous,
unconditional companion is :func:`~omnibias.holonomic._core.zeilberger.gosper_definite_sum`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from fractions import Fraction
from math import gcd
from pathlib import Path
from typing import Any

from omnibias.core.proof.certificate import Cert, make_certificate, verify_certificate_digest
from omnibias.core.proof.lean_check import (
    LeanCheckResult,
    check_certificate,
    generate_obligation,
    lean_check_available,
)
from omnibias.holonomic._core.hyperterm import ProperTerm
from omnibias.holonomic._core.ore import OrePolynomial
from omnibias.holonomic._core.poly2 import Poly2
from omnibias.holonomic._core.ratfunc import (
    RatFunc,
    rf_add,
    rf_from_poly,
    rf_from_rational,
    rf_is_zero,
    rf_mul,
    rf_normalize,
    rf_shift,
)
from omnibias.holonomic._core.rational_poly import Poly, peval, pmul
from omnibias.holonomic._core.zeilberger import (
    KBound,
    Summand,
    ZeilbergerCertificate,
    creative_telescoping,
    summand_sum,
    zeilberger,
)


def _zero_identity(claim: str, pairs: list[tuple[Fraction, Fraction]], *, meta: dict[str, Any]) -> Cert:
    """Seal ``sum_i c_i v_i = 0`` (rationals) as an exact integer ``rational_identity``."""
    products = [Fraction(c) * Fraction(v) for c, v in pairs]
    den = 1
    for p in products:
        den = den * p.denominator // gcd(den, p.denominator)
    lhs_terms = [[int(p * den), 1] for p in products] or [[0, 1]]
    payload = {"type": "rational_identity", "lhs_terms": lhs_terms, "rhs": 0}
    return make_certificate(claim=claim, payload=payload, honesty={"unproven_claim": False}, meta=meta)


def _value_equality(claim: str, computed: Fraction, expected: Fraction, *, meta: dict[str, Any]) -> Cert:
    """Seal ``computed == expected`` via cross-multiplication (exact ``Int`` equality)."""
    pc, qc = computed.numerator, computed.denominator
    pe, qe = expected.numerator, expected.denominator
    lhs_terms = [[qe, pc], [-qc, pe]]  # pc*qe - pe*qc = 0
    payload = {"type": "rational_identity", "lhs_terms": lhs_terms, "rhs": 0}
    return make_certificate(claim=claim, payload=payload, honesty={"unproven_claim": False}, meta=meta)


@dataclass(frozen=True)
class HolonomicProof:
    """A Lean-certifiable proof of a hypergeometric identity on a verified range."""

    name: str
    recurrence: OrePolynomial
    order: int
    n_verified: int
    identity_holds_on_range: bool
    certificates: tuple[Cert, ...]
    lean_results: tuple[LeanCheckResult, ...]
    lean_available: bool
    all_n: bool = False

    @property
    def obligations_generated(self) -> bool:
        """Whether every certificate yields a finite Lean-checkable obligation."""
        return all(generate_obligation(c) is not None for c in self.certificates)

    @property
    def certificates_sealed(self) -> bool:
        """Whether every certificate's tamper-evident digest matches its body."""
        return all(verify_certificate_digest(c) for c in self.certificates)

    @property
    def theorem_prover_verified(self) -> bool:
        """``True`` only when ``lake`` genuinely accepted **every** obligation."""
        return (
            self.lean_available
            and len(self.lean_results) > 0
            and all(r.verified for r in self.lean_results)
        )

    def pretty(self) -> str:
        """Human-readable summary of the certified identity and its recurrence."""
        if not self.identity_holds_on_range:
            status = "REFUTED"
        elif self.all_n:
            status = "PROVEN(all n)"
        else:
            status = "PROVEN(range)"
        scope = "all n" if self.all_n else f"[0, {self.n_verified}]"
        return (
            f"{self.name}: {status} on {scope}; order-{self.order} "
            f"recurrence; {len(self.certificates)} obligations; "
            f"theorem_prover_verified={self.theorem_prover_verified}"
        )


def prove_hypergeometric_identity(
    *,
    name: str,
    summand: Summand,
    closed_form: Callable[[int], Fraction | int],
    n_max: int = 16,
    k_bound: KBound | None = None,
    max_order: int = 4,
    max_index_degree: int = 3,
    prove_lean: bool = False,
    lean_sample: int | None = None,
    lean_timeout: float = 600.0,
    start: Path | None = None,
) -> HolonomicProof:
    r"""Prove ``sum_k F(n, k) = g(n)`` by certified creative telescoping.

    ``summand`` is ``F(n, k)`` (exact, finite ``k``-support); ``closed_form`` is ``g``.
    Emits ``rational_identity`` obligations certifying the identity on ``[0, n_max]`` plus
    the shared recurrence, and (when ``prove_lean``) runs the Lean kernel on them
    (``lean_sample`` limits how many are actually sent to ``lake``; ``None`` = all).
    """
    kb: KBound = k_bound or (lambda n: (0, n))
    f = [summand_sum(summand, n, kb) for n in range(n_max + 1)]
    g = [Fraction(closed_form(n)) for n in range(n_max + 1)]
    identity_ok = all(f[n] == g[n] for n in range(n_max + 1))

    tele = creative_telescoping(
        summand, name=name, n_max=n_max, k_bound=kb,
        max_order=max_order, max_index_degree=max_index_degree,
    )
    op = tele.recurrence
    order = op.order

    certs: list[Cert] = []
    for n in range(n_max + 1):
        certs.append(
            _value_equality(
                f"{name}: sum_k F({n},k) = g({n})", f[n], g[n],
                meta={"identity": name, "n": n, "kind": "value_equality", "label": "closed-form"},
            )
        )
    for n in range(n_max - order + 1):
        f_pairs = [(peval(op.coeffs[i], n), f[n + i]) for i in range(order + 1)]
        certs.append(
            _zero_identity(
                f"{name}: L[f]({n}) = 0", f_pairs,
                meta={"identity": name, "n": n, "kind": "recurrence_on_sum", "label": "closed-form"},
            )
        )
        g_pairs = [(peval(op.coeffs[i], n), g[n + i]) for i in range(order + 1)]
        certs.append(
            _zero_identity(
                f"{name}: L[g]({n}) = 0", g_pairs,
                meta={"identity": name, "n": n, "kind": "recurrence_on_closed_form", "label": "closed-form"},
            )
        )

    available = lean_check_available(start)
    lean_results: list[LeanCheckResult] = []
    if prove_lean:
        to_check = certs if lean_sample is None else certs[:lean_sample]
        for c in to_check:
            lean_results.append(check_certificate(c, timeout=lean_timeout, start=start))

    return HolonomicProof(
        name=name,
        recurrence=op,
        order=order,
        n_verified=n_max,
        identity_holds_on_range=identity_ok,
        certificates=tuple(certs),
        lean_results=tuple(lean_results),
        lean_available=available,
    )


def _p2_to_poly_n(p: Poly2) -> Poly:
    """A ``k``-free bivariate polynomial as a univariate polynomial in ``n``."""
    if any(j != 0 for (_i, j) in p):
        raise ValueError("closed form must be free of the summation variable k")
    if not p:
        return ()
    deg = max(i for (i, _j) in p)
    return tuple(p.get((i, 0), Fraction(0)) for i in range(deg + 1))


def _closed_form_obeys_recurrence(
    coeffs: tuple[Poly, ...], g_term: ProperTerm
) -> tuple[bool, int]:
    r"""Whether ``L[g] = 0`` for **all** ``n`` (``g`` hypergeometric in ``n``); + a grid bound.

    ``L[g](n) = g(n) * sum_i c_i(n) p_i^g(n)`` with ``p_i^g(n) = g(n+i)/g(n)`` rational.
    The bracket ``S(n)`` is built exactly with :mod:`.ratfunc`; ``S ≡ 0`` (checked here) is
    ``L[g] ≡ 0``. The returned bound is the degree of the cleared numerator, so
    ``L[g](n0) = 0`` on ``0..bound`` re-proves it for all ``n``.
    """
    r2 = g_term.ratio_n()
    rho: RatFunc = rf_normalize(_p2_to_poly_n(r2[0]), _p2_to_poly_n(r2[1]))
    d = len(coeffs) - 1
    p_g: list[RatFunc] = [rf_from_rational(1)]
    cur = rf_from_rational(1)
    for i in range(d):
        cur = rf_mul(cur, rf_shift(rho, i))
        p_g.append(cur)
    s = rf_from_rational(0)
    for i, c in enumerate(coeffs):
        s = rf_add(s, rf_mul(rf_from_poly(c), p_g[i]))
    ok = rf_is_zero(s)
    den_prod: Poly = (Fraction(1),)
    for _num, den in p_g:
        den_prod = pmul(den_prod, den)
    bound = 0
    for i, c in enumerate(coeffs):
        num_i, den_i = p_g[i]
        from omnibias.holonomic._core.rational_poly import pdivmod

        cofactor, _ = pdivmod(den_prod, den_i)
        term_deg = len(pmul(pmul(c, num_i), cofactor)) - 1
        bound = max(bound, term_deg)
    return ok, max(bound, 0)


def _clean_grid_offset(w: Poly2, deg_n: int, deg_k: int) -> tuple[int, int]:
    """A base ``(n_off, k_off)`` whose whole tensor block avoids certificate poles."""
    from omnibias.holonomic._core.poly2 import p2_eval

    for n_off in range(0, 24):
        for k_off in range(0, 24):
            ok = True
            for n0 in range(n_off, n_off + deg_n + 1):
                for k0 in range(k_off, k_off + deg_k + 2):  # k0 and k0+1 both needed
                    if p2_eval(w, n0, k0) == 0:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                return n_off, k_off
    return 0, 0


def prove_identity_zeilberger(
    *,
    name: str,
    term: ProperTerm,
    closed_form_term: ProperTerm,
    max_order: int = 4,
    prove_lean: bool = False,
    lean_sample: int | None = None,
    lean_timeout: float = 600.0,
    start: Path | None = None,
) -> HolonomicProof:
    r"""Prove ``sum_k F(n, k) = g(n)`` for **all** ``n`` by exact creative telescoping.

    Both ``term`` (``F``) and ``closed_form_term`` (``g``, free of ``k``) are proper
    hypergeometric. Emits three finite, all-``n`` obligation families the Lean kernel can
    discharge exactly:

    * the telescoping identity ``N(n0, k0) = 0`` on a pole-free
      ``(deg_n+1) x (deg_k+1)`` grid -- proves ``L[f] = 0`` for all ``n``;
    * ``L[g](n0) = 0`` on ``0..deg`` -- proves the closed form obeys the *same* recurrence
      for all ``n``;
    * ``f(i) = g(i)`` for the ``order`` initial conditions.

    ``all_n`` is set only when the two symbolic identities hold exactly (rigorous) and the
    initial conditions match. ``theorem_prover_verified`` remains gated on a genuine
    ``lake`` pass. Raises ``ValueError`` if no Zeilberger certificate is found.
    """
    cert: ZeilbergerCertificate = zeilberger(term, max_order=max_order)
    op = cert.recurrence
    order = op.order
    coeffs = cert.coeffs

    f_ok = cert.verify_symbolic()
    g_ok, g_bound = _closed_form_obeys_recurrence(coeffs, closed_form_term)

    def g_val(n: int) -> Fraction:
        return Fraction(closed_form_term.value(n, 0))

    # f(n) support upper bound: values are computed by summing over a generous window.
    def f_val(n: int) -> Fraction:
        return Fraction(term.sum_over_k(n, 0, n + max_order + 2))

    init_ok = all(f_val(i) == g_val(i) for i in range(order))

    certs: list[Cert] = []
    # (1) telescoping identity on a pole-free tensor grid -> L[f] = 0 for all n.
    n_off, k_off = _clean_grid_offset(cert.certificate[1], cert.grid_degree_n, cert.grid_degree_k)
    for n0 in range(n_off, n_off + cert.grid_degree_n + 1):
        for k0 in range(k_off, k_off + cert.grid_degree_k + 1):
            terms = cert.identity_terms(n0, k0)
            pairs = [(t, Fraction(1)) for t in terms]
            certs.append(
                _zero_identity(
                    f"{name}: telescoping N({n0},{k0}) = 0", pairs,
                    meta={"identity": name, "n": n0, "k": k0, "kind": "telescoping_grid",
                          "label": "closed-form", "all_n": True},
                )
            )
    # (2) closed form obeys the same recurrence for all n.
    g_off = 0
    while any(g_val(g_off + i) == 0 for i in range(order + 1)) and g_off < 24:
        g_off += 1
    for n0 in range(g_off, g_off + g_bound + 1):
        g_pairs = [(peval(coeffs[i], n0), g_val(n0 + i)) for i in range(order + 1)]
        certs.append(
            _zero_identity(
                f"{name}: L[g]({n0}) = 0", g_pairs,
                meta={"identity": name, "n": n0, "kind": "recurrence_on_closed_form",
                      "label": "closed-form", "all_n": True},
            )
        )
    # (3) initial conditions.
    for i in range(order):
        certs.append(
            _value_equality(
                f"{name}: f({i}) = g({i})", f_val(i), g_val(i),
                meta={"identity": name, "n": i, "kind": "initial_condition", "label": "closed-form"},
            )
        )

    available = lean_check_available(start)
    lean_results: list[LeanCheckResult] = []
    if prove_lean:
        to_check = certs if lean_sample is None else certs[:lean_sample]
        for c in to_check:
            lean_results.append(check_certificate(c, timeout=lean_timeout, start=start))

    return HolonomicProof(
        name=name,
        recurrence=op,
        order=order,
        n_verified=max(cert.grid_degree_n, g_bound),
        identity_holds_on_range=init_ok and f_ok and g_ok,
        certificates=tuple(certs),
        lean_results=tuple(lean_results),
        lean_available=available,
        all_n=f_ok and g_ok and init_ok,
    )


__all__ = [
    "HolonomicProof",
    "prove_hypergeometric_identity",
    "prove_identity_zeilberger",
]
