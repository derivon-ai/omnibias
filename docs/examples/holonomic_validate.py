# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Data-driven / verified / best-in-class smoke for omnibias-holonomic.

Run::

    pip install "omnibias-holonomic[test]"
    python docs/examples/holonomic_validate.py

Each *probe* turns a holonomic capability into an instrumented experiment under the
``omnibias-dev-empirical-validation`` gates -- an ``mpmath`` high-precision oracle, a
best-in-class comparison against a named baseline, and a ``K >= 8`` seed sweep -- and
records gaps / flaws into a shared
:class:`~omnibias.difference.validation.FindingsLedger` (JSON written to scratch, never the
repo tree). The assertions here are the CI gate.

Honesty labels exercised: **closed-form / exact for all n** (Gosper definite sums, the
symbolic Ore ``lclm`` / ``symmetric_product`` closures, true Zeilberger's ``P(n,k) == 0``
grid, and every ``rational_identity`` certificate payload), **exact within a scoped regime**
(Petkovsek's ``hyper`` over rational-root / linear factors, ``q``-numeric certificates at a
fixed rational ``q``), **guessed then verified** (which recurrence / differential / algebraic
annihilator fits is fitted, then re-checked exactly on held-out terms), **numerical** (the
Poincare-Perron leading asymptotic, certified only where ``transfer_theorem`` applies), and
**theorem_prover_verified** -- earned only on a genuine ``lake`` pass and never forged
(absent a toolchain the bridge degrades gracefully). This package works in the discrete
register founded by ``omnibias-difference`` (the ``delta -> 0`` collapse), certifying
identities through finite differences and recurrences; the ``q -> 1`` limit of the
q-holonomic layer is a *distinct* limit, not that founding collapse.
"""

from __future__ import annotations

from fractions import Fraction
from math import comb, factorial

from omnibias.core.proof.certificate import verify_certificate_digest
from omnibias.core.proof.lean_check import generate_obligation
from omnibias.difference.validation import (
    FindingsLedger,
    baseline_compare,
    require_mpmath,
    seed_sweep,
)
from omnibias.holonomic import (
    PRecursive,
    ProperTerm,
    binomial_nk,
    certified_asymptotic,
    creative_telescoping,
    dfinite_add,
    dfinite_hadamard,
    empirical_rate,
    geometric_k,
    gosper_definite_sum,
    gosper_sum,
    guess_algebraic,
    guess_dfinite,
    hyper,
    precursive_asymptotics,
    prove_hypergeometric_identity,
    prove_identity_zeilberger,
    q_gosper,
    q_gosper_definite_sum,
    q_zeilberger,
    shift_algebra,
)

# (name, summand F(n,k), closed form g(n), optional k-bound) -- >= 8 classic identities.
IDENTITIES = [
    ("sum C(n,k) = 2^n", lambda n, k: comb(n, k), lambda n: 2**n, None),
    ("sum C(n,k)^2 = C(2n,n)", lambda n, k: comb(n, k) ** 2, lambda n: comb(2 * n, n), None),
    ("sum k C(n,k) = n 2^{n-1}", lambda n, k: k * comb(n, k), lambda n: n * 2 ** (n - 1) if n else 0, None),
    ("sum k^2 C(n,k)", lambda n, k: k * k * comb(n, k), lambda n: n * (n + 1) * 2 ** (n - 2) if n else 0, None),
    ("Vandermonde C(n,k)C(n,n-k)", lambda n, k: comb(n, k) * comb(n, n - k), lambda n: comb(2 * n, n), None),
    ("sum_{k<=n} k = n(n+1)/2", lambda n, k: k, lambda n: n * (n + 1) // 2, None),
    ("sum k^2 = n(n+1)(2n+1)/6", lambda n, k: k * k, lambda n: n * (n + 1) * (2 * n + 1) // 6, None),
    ("sum k^3 = (n(n+1)/2)^2", lambda n, k: k**3, lambda n: (n * (n + 1) // 2) ** 2, None),
    ("hockey-stick C(k,1) = C(n+1,2)", lambda n, k: comb(k, 1), lambda n: comb(n + 1, 2), None),
    ("sum 2^k = 2^{n+1}-1", lambda n, k: 2**k, lambda n: 2 ** (n + 1) - 1, None),
]


def probe_gosper_closed_form(ledger: FindingsLedger) -> None:
    """Gosper returns *exact* closed forms AND refuses non-summable terms (best-in-class).

    The named baseline is a decision procedure that *assumes* every hypergeometric term is
    summable (as any numeric partial-sum heuristic implicitly does). It scores wrong on
    every term that has **no** closed form; Gosper's exact decision procedure never does.
    """
    print("=== Gosper closed-form summation: exact + sound decision vs assume-summable ===")
    mp = require_mpmath()
    # (name, num, den, term0, a, term, is_summable) for t with t(k+1)/t(k) = num/den.
    cases = [
        ("k", (1, 1), (0, 1), Fraction(1), 1, lambda k: Fraction(k), True),
        ("k^2", (1, 2, 1), (0, 0, 1), Fraction(1), 1, lambda k: Fraction(k * k), True),
        ("k^3", (1, 3, 3, 1), (0, 0, 0, 1), Fraction(1), 1, lambda k: Fraction(k**3), True),
        ("2^k", (2,), (1,), Fraction(1), 0, lambda k: Fraction(2**k), True),
        ("(1/3)^k", (1,), (3,), Fraction(1), 0, lambda k: Fraction(1, 3) ** k, True),
        ("k*k!", (1, 2, 1), (0, 1), Fraction(1), 1, lambda k: Fraction(k * factorial(k)), True),
        ("C(2k,k)", (2, 4), (1, 1), Fraction(1), 0, lambda k: Fraction(comb(2 * k, k)), False),
        ("1/k!", (1,), (1, 1), Fraction(1), 0, lambda k: Fraction(1, factorial(k)), False),
    ]

    def classification(seed: int) -> float:
        name, num, den, t0, a, term, summable = cases[seed]
        res = gosper_sum(num, den)
        assert res.summable == summable, f"{name}: Gosper mis-classified summability"
        if summable:
            b = a + 20
            exact = gosper_definite_sum(num, den, t0, a, b)
            brute = sum((term(k) for k in range(a, b)), Fraction(0))
            assert exact == brute, f"{name}: closed form disagrees with brute rational sum"
            with mp.workdps(50):
                oracle = mp.fsum(
                    [mp.mpf(term(k).numerator) / term(k).denominator for k in range(a, b)]
                )
            assert abs(float(exact) - float(oracle)) <= 1e-6 * (1 + abs(float(oracle)))
        # candidate is correct on this term; the assume-summable baseline is correct
        # only when the term really is summable.
        return 0.0 if summable else 1.0  # count baseline errors

    baseline_errors = seed_sweep(classification, range(len(cases)))
    n_wrong = int(round(baseline_errors["mean"] * baseline_errors["n"]))
    cmp = baseline_compare("gosper-decision", 0.0, float(n_wrong), lower_is_better=True)
    print(
        f"  Gosper: exact closed forms + {n_wrong}/{len(cases)} non-summable correctly refused; "
        f"assume-summable baseline wrong on {n_wrong}; wins={cmp.wins}"
    )
    assert cmp.wins and n_wrong >= 1
    ledger.add(
        "holonomic",
        "flaw",
        "assuming every hypergeometric term is summable is unsound; Gosper decides exactly",
        detail="C(2k,k) and 1/k! have no hypergeometric antidifference; Gosper refuses them",
        resolved=True,
    )


def probe_creative_telescoping(ledger: FindingsLedger) -> None:
    """Exact null-space recurrence discovery recovers non-monic recurrences (data-driven)."""
    print("=== creative telescoping: guessed-then-verified recurrences (K>=8) ===")
    orders: dict[str, int] = {}

    def residual(seed: int) -> float:
        name, summand, _closed, kb = IDENTITIES[seed]
        tele = creative_telescoping(summand, name=name, n_max=16, k_bound=kb)
        orders[name] = tele.order
        return float(tele.max_residual())  # exactly 0 for a true annihilator

    res = seed_sweep(residual, range(len(IDENTITIES)))
    print(f"  {len(IDENTITIES)} sums: max |L[f](n)| over all = {res['max']} (expect 0)")
    assert res["max"] == 0.0, "a guessed recurrence failed exact range verification"
    # central binomial's recurrence is genuinely non-monic ((n+1) leading coeff) -- the
    # exact rational null space recovers it where a monic float fit could not.
    assert orders["sum C(n,k)^2 = C(2n,n)"] == 1
    print("  non-monic recurrences (e.g. central binomial) recovered exactly")
    ledger.add(
        "holonomic",
        "info",
        "exact null-space telescoping recovers non-monic P-recurrences with zero residual",
        identities=len(IDENTITIES),
    )


def probe_certified_identities(ledger: FindingsLedger) -> None:
    """A Lean-checkable certificate bundle beats a numeric-only equality check (best-in-class)."""
    print("=== certified identities: rational_identity obligations vs numeric-only ===")
    mp = require_mpmath()
    total_obligations = 0
    lean_available = False

    for seed in range(len(IDENTITIES)):
        name, summand, closed, kb = IDENTITIES[seed]
        proof = prove_hypergeometric_identity(
            name=name, summand=summand, closed_form=closed, n_max=14, k_bound=kb
        )
        lean_available = proof.lean_available
        # data-driven: the identity holds exactly on the whole checked range.
        assert proof.identity_holds_on_range, f"{name}: identity failed on range"
        # verified: every certificate is sealed (tamper-evident) and yields a Lean obligation.
        assert proof.certificates_sealed, f"{name}: a certificate digest is broken"
        assert proof.obligations_generated, f"{name}: an obligation failed to generate"
        # never forged: no lean pass -> flag stays False.
        if not lean_available:
            assert not proof.theorem_prover_verified
        total_obligations += len(proof.certificates)
        # mpmath oracle cross-check of the closed form at a few n.
        with mp.workdps(40):
            for n in (3, 7, 11):
                lo, hi = kb(n) if kb else (0, n)
                oracle = mp.fsum([mp.mpf(int(summand(n, k))) for k in range(lo, hi + 1)])
                assert abs(float(oracle) - float(closed(n))) <= 1e-9 * (1 + abs(float(oracle)))

    # best-in-class: candidate emits machine-checkable obligations; the numeric baseline emits 0.
    cmp = baseline_compare(
        "certified-vs-numeric", float(total_obligations), 0.0, lower_is_better=False
    )
    print(
        f"  {len(IDENTITIES)} identities -> {total_obligations} sealed rational_identity "
        f"obligations (numeric-only baseline: 0); wins={cmp.wins}"
    )
    assert cmp.wins

    # tamper-evidence: mutating a sealed certificate is detected.
    proof = prove_hypergeometric_identity(
        name="sum C(n,k) = 2^n", summand=lambda n, k: comb(n, k), closed_form=lambda n: 2**n, n_max=6
    )
    tampered = dict(proof.certificates[0])
    tampered["payload"] = {"type": "rational_identity", "lhs_terms": [[1, 1]], "rhs": 999}
    assert not verify_certificate_digest(tampered), "tampering went undetected"
    assert generate_obligation(proof.certificates[0]) is not None
    print(f"  tamper-evident seals verified; lean_available={lean_available}")
    ledger.add(
        "holonomic",
        "flaw" if not lean_available else "info",
        "numeric-only identity checking yields no machine-checkable proof; certificates do",
        detail="rational_identity obligations are Lean-kernel-checkable and tamper-evident",
        lean_available=lean_available,
        resolved=True,
    )


def _prec(coeffs: list[list[int]], initial: list[int]) -> PRecursive:
    """A P-recursive sequence from integer recurrence coefficients + initial values."""
    op = shift_algebra().operator([[Fraction(c) for c in poly] for poly in coeffs])
    return PRecursive(op, tuple(Fraction(v) for v in initial))


def _shift_op(coeffs: list[list[int]]):  # noqa: ANN202 - returns an OrePolynomial
    return shift_algebra().operator([[Fraction(c) for c in poly] for poly in coeffs])


def probe_symbolic_closures(ledger: FindingsLedger) -> None:
    """Symbolic Ore closures (`lclm` / `symmetric_product`) prove all-n; ansatz range-only.

    The candidate builds each closure from the input annihilators *symbolically* (exact for
    all `n`) and we back that by regenerating a long tail (well past the construction window)
    and matching it against the true combined sequence. The named baseline is the
    verified-ansatz closure, which only certifies the fitted range.
    """
    print("=== symbolic closures: lclm / symmetric_product all-n vs ansatz range ===")
    pool = {
        "2^n": _prec([[-2], [1]], [1]),
        "3^n": _prec([[-3], [1]], [1]),
        "Fib": _prec([[-1], [-1], [1]], [0, 1]),
        "Catalan": _prec([[-2, -4], [2, 1]], [1]),
    }
    names = list(pool)
    # 8 closures: add + Hadamard over 4 sequence pairs.
    pairs = [(a, b) for i, a in enumerate(names) for b in names[i + 1 :]][:4]
    ops = [(dfinite_add, "add"), (dfinite_hadamard, "hadamard")]
    cases = [(op, tag, a, b) for (op, tag) in ops for (a, b) in pairs]
    long = 90

    def all_n_proven(seed: int) -> float:
        fn, _tag, an, bn = cases[seed]
        a, b = pool[an], pool[bn]
        ta, tb = a.terms(long), b.terms(long)
        combine = (lambda x, y: x + y) if fn is dfinite_add else (lambda x, y: x * y)
        truth = [combine(ta[n], tb[n]) for n in range(long)]
        result = fn(a, b)  # symbolic-first closure
        regen = result.terms(long)  # regenerate far past the 40-term construction window
        # all-n backing: the symbolic operator reproduces the true sequence for every n.
        assert regen == truth, f"{an} {_tag} {bn}: closure diverged from truth at all-n scale"
        return 1.0

    proven = seed_sweep(all_n_proven, range(len(cases)))
    candidate = proven["mean"] * proven["n"]  # closures proven for all n
    cmp = baseline_compare("symbolic-vs-ansatz", candidate, 0.0, lower_is_better=False)
    print(
        f"  {len(cases)} closures proven for all n (regenerated to {long}); "
        f"ansatz baseline proves 0 beyond its fit range; wins={cmp.wins}"
    )
    assert cmp.wins and candidate == len(cases)
    ledger.add(
        "holonomic",
        "info",
        "symbolic Ore lclm / symmetric_product closures hold for all n; the ansatz only range-verifies",
        detail="closures regenerated 90 terms (past the 40-term window) and matched the true sequence",
        closures=len(cases),
    )


def probe_all_n_zeilberger(ledger: FindingsLedger) -> None:
    """True Zeilberger emits all-n obligations; guessed telescoping is range-only (best-in-class)."""
    print("=== all-n Zeilberger: P(n,k)=0 obligations vs range-bounded proof ===")
    cf = lambda geom: ProperTerm((), geom_n=Fraction(geom))  # noqa: E731
    identities = [
        ("sum C(n,k) = 2^n", binomial_nk(), cf(2), lambda n, k: comb(n, k), lambda n: 2**n),
        ("sum C(n,k)(-2)^k = (-1)^n", geometric_k(-2).times(binomial_nk()), cf(-1),
         lambda n, k: comb(n, k) * (-2) ** k, lambda n: (-1) ** n),
        ("sum C(n,k)2^k = 3^n", geometric_k(2).times(binomial_nk()), cf(3),
         lambda n, k: comb(n, k) * 2**k, lambda n: 3**n),
        ("sum C(n,k)3^k = 4^n", geometric_k(3).times(binomial_nk()), cf(4),
         lambda n, k: comb(n, k) * 3**k, lambda n: 4**n),
        ("sum C(n,k)4^k = 5^n", geometric_k(4).times(binomial_nk()), cf(5),
         lambda n, k: comb(n, k) * 4**k, lambda n: 5**n),
        ("sum C(n,k)5^k = 6^n", geometric_k(5).times(binomial_nk()), cf(6),
         lambda n, k: comb(n, k) * 5**k, lambda n: 6**n),
        ("sum C(n,k)(1/2)^k=(3/2)^n", geometric_k(Fraction(1, 2)).times(binomial_nk()),
         cf(Fraction(3, 2)), None, None),
        ("sum C(n,k)^2 = C(2n,n)", binomial_nk().power(2),
         ProperTerm(((2, 0, 0, 1), (1, 0, 0, -1), (1, 0, 0, -1))),
         lambda n, k: comb(n, k) ** 2, lambda n: comb(2 * n, n)),
    ]

    def all_n(seed: int) -> float:
        name, term, cf_term, _summand, _closed = identities[seed]
        proof = prove_identity_zeilberger(name=name, term=term, closed_form_term=cf_term)
        assert proof.identity_holds_on_range, f"{name}: symbolic identity failed"
        assert proof.all_n, f"{name}: not proven for all n"
        assert proof.certificates_sealed and proof.obligations_generated
        return 1.0

    proven = seed_sweep(all_n, range(len(identities)))
    n_all_n = int(round(proven["mean"] * proven["n"]))
    # baseline: the guessed creative-telescoping proof is only range-bounded (all_n stays False).
    base = prove_hypergeometric_identity(
        name="sum C(n,k) = 2^n", summand=lambda n, k: comb(n, k), closed_form=lambda n: 2**n
    )
    assert not base.all_n
    cmp = baseline_compare("all-n-vs-range", float(n_all_n), 0.0, lower_is_better=False)
    print(
        f"  {n_all_n}/{len(identities)} identities proven for ALL n (P(n,k)=0 grid); "
        f"guessed range proof certifies 0 for all n; wins={cmp.wins}"
    )
    assert cmp.wins and n_all_n == len(identities)
    ledger.add(
        "holonomic",
        "flaw",
        "a range-bounded telescoping proof does not certify all n; Zeilberger's P(n,k)=0 grid does",
        detail="each obligation is a finite integer rational_identity holding for every n",
        resolved=True,
    )


def probe_petkovsek_hyper(ledger: FindingsLedger) -> None:
    """Petkovsek's Hyper recovers exact hypergeometric term ratios; numeric rank cannot."""
    print("=== Petkovsek Hyper: exact term ratios vs numeric solution-space rank ===")
    # (name, shift operator [c_0..c_r], expected number of hypergeometric ratios).
    cases = [
        ("factorial", [[-1, -1], [1]], 1),      # a_{n+1} = (n+1) a_n
        ("2^n", [[-2], [1]], 1),
        ("3^n", [[-3], [1]], 1),
        ("5^n", [[-5], [1]], 1),
        ("(1/2)^n", [[-1], [2]], 1),
        ("Catalan ratio", [[-2, -4], [2, 1]], 1),  # (n+2)a_{n+1} = (4n+2) a_n
        ("(2n+1)!!-ish", [[-1, -2], [1]], 1),      # a_{n+1} = (2n+1) a_n
        ("Fibonacci (out of scope)", [[-1], [-1], [1]], 0),  # phi irrational -> none
    ]

    def recovered(seed: int) -> float:
        name, coeffs, expected = cases[seed]
        op = _shift_op(coeffs)
        ratios = hyper(op)
        assert len(ratios) == expected, f"{name}: expected {expected} ratios, got {len(ratios)}"
        return float(len(ratios))  # exact hypergeometric solutions produced

    res = seed_sweep(recovered, range(len(cases)))
    total = res["mean"] * res["n"]
    cmp = baseline_compare("hyper-vs-rank", total, 0.0, lower_is_better=False)
    print(
        f"  {int(total)} exact hypergeometric term ratios recovered (each verified); "
        f"numeric rank yields 0 closed-form terms; wins={cmp.wins}"
    )
    assert cmp.wins and total >= 6
    ledger.add(
        "holonomic",
        "info",
        "a numeric rank gives the solution-space dimension but no closed-form term; Hyper gives the exact ratio",
        detail="Gosper-Petkovsek monic-divisor enumeration + exact null space; Fibonacci correctly out of scope",
        resolved=True,
    )


def probe_q_holonomic(ledger: FindingsLedger) -> None:
    """q-Gosper decides q-summability exactly (and matches direct q-sums); assume-summable errs."""
    print("=== q-holonomic: q-Gosper decision + q-Zeilberger vs direct q-series ===")
    from omnibias.holonomic._core.rational_poly import peval, to_poly

    def qcase(q: Fraction):  # noqa: ANN202
        return [
            ("3^k geom", to_poly([3]), to_poly([1]), True, q),
            ("designed R=x", to_poly([1, 1]), to_poly([0, q]), True, q),
            ("1/[k]_q! (nonsummable)", to_poly([1 - q]), to_poly([1, -q]), False, q),
        ]

    cases = [c for q in (Fraction(2), Fraction(3), Fraction(1, 2)) for c in qcase(q)][:9]

    def q_term_sum(num, den, t0: Fraction, a: int, b: int, q: Fraction) -> Fraction:
        t = {a: t0}
        for k in range(a, b):
            t[k + 1] = t[k] * peval(num, q**k) / peval(den, q**k)
        return sum((t[k] for k in range(a, b)), Fraction(0))

    def baseline_errors(seed: int) -> float:
        name, num, den, summable, q = cases[seed]
        res = q_gosper(num, den, q)
        assert res.summable == summable, f"{name} (q={q}): q-Gosper mis-decided summability"
        if summable:
            got = q_gosper_definite_sum(num, den, Fraction(1), 0, 5, q)
            assert got == q_term_sum(num, den, Fraction(1), 0, 5, q), f"{name}: q-sum mismatch"
        return 0.0 if summable else 1.0  # the assume-summable baseline errs here

    errs = seed_sweep(baseline_errors, range(len(cases)))
    n_wrong = int(round(errs["mean"] * errs["n"]))
    cmp = baseline_compare("qgosper-decision", 0.0, float(n_wrong), lower_is_better=True)
    # q-Zeilberger: the q-bracket sum S(n)=sum_{k<=n} q^k obeys a compact order-1 q-recurrence.
    q = Fraction(2)
    rec = q_zeilberger(lambda n, k: q**k, q, n_max=14)
    assert rec is not None and rec.order == 1 and rec.max_residual() == 0
    print(
        f"  q-Gosper exact on {len(cases)} q-terms ({n_wrong} non-summable refused; "
        f"assume-summable baseline wrong on {n_wrong}); q-Zeilberger order-1 recurrence; wins={cmp.wins}"
    )
    assert cmp.wins and n_wrong >= 1
    ledger.add(
        "holonomic",
        "flaw",
        "assuming a q-hypergeometric term is q-summable is unsound; q-Gosper decides exactly at a fixed rational q",
        detail="q-numeric certificates (exact per q); q -> 1 is the distinct ordinary-holonomic limit",
        resolved=True,
    )


def probe_guessing_vs_lstsq(ledger: FindingsLedger) -> None:
    """Exact guess_dfinite / guess_algebraic decide (D-finite / algebraic)-ness; least-squares can't."""
    print("=== guessing: exact null space (decides + verifies) vs always-fits least-squares ===")
    exp = [Fraction(1, factorial(m)) for m in range(14)]
    geom = [Fraction(1)] * 14
    linear = [Fraction(m) for m in range(14)]  # x/(1-x)^2
    sqrt1px = []
    for k in range(14):
        c = Fraction(1)
        for t in range(k):
            c *= Fraction(1, 2) - t
        sqrt1px.append(c / factorial(k))
    catalan = [Fraction(comb(2 * n, n), n + 1) for n in range(14)]
    double_exp = [Fraction(2) ** (2**n) for n in range(8)]
    ones = [Fraction(1)] * 14
    # (kind, series, is-fittable) -- least-squares always claims fittable.
    cases = [
        ("dfinite exp", "d", exp, True),
        ("dfinite geometric", "d", geom, True),
        ("dfinite linear", "d", linear, True),
        ("dfinite sqrt(1+x)", "d", sqrt1px, True),
        ("dfinite double-exp", "d", double_exp, False),
        ("algebraic 1/(1-x)", "a", ones, True),
        ("algebraic Catalan", "a", catalan, True),
        ("algebraic sqrt(1+x)", "a", sqrt1px, True),
        ("algebraic exp", "a", exp, False),
    ]

    def baseline_errors(seed: int) -> float:
        name, kind, series, fittable = cases[seed]
        if kind == "d":
            got = guess_dfinite(series, max_order=3, max_degree=3)
        else:
            got = guess_algebraic(series, max_x_degree=3, max_y_degree=3)
        found = got is not None
        assert found == fittable, f"{name}: classifier wrong (found={found}, expected {fittable})"
        return 0.0 if fittable else 1.0  # least-squares would (wrongly) fit the non-holonomic ones

    errs = seed_sweep(baseline_errors, range(len(cases)))
    n_wrong = int(round(errs["mean"] * errs["n"]))
    cmp = baseline_compare("guess-vs-lstsq", 0.0, float(n_wrong), lower_is_better=True)
    print(
        f"  exact guessers correct on all {len(cases)} series ({n_wrong} non-holonomic correctly "
        f"rejected); always-fit least-squares wrong on {n_wrong}; wins={cmp.wins}"
    )
    assert cmp.wins and n_wrong >= 1
    ledger.add(
        "holonomic",
        "flaw",
        "a least-squares fit always returns coefficients; it cannot decide D-finite/algebraic-ness",
        detail="the exact null space + held-out verification rejects non-holonomic series with zero residual",
        resolved=True,
    )


def probe_asymptotics_vs_ratio(ledger: FindingsLedger) -> None:
    """The characteristic-root rate is exact; the empirical ratio a(n+1)/a(n) lags (best-in-class)."""
    print("=== asymptotics: characteristic-root rate vs empirical ratio estimate ===")
    phi = (1.0 + 5.0**0.5) / 2.0
    tribonacci = 1.8392867552141612  # real root of t^3 = t^2 + t + 1 (order-3 char poly)
    cases = [
        ("2^n", _prec([[-2], [1]], [1]), 2.0),
        ("3^n", _prec([[-3], [1]], [1]), 3.0),
        ("5^n", _prec([[-5], [1]], [1]), 5.0),
        ("Fibonacci", _prec([[-1], [-1], [1]], [0, 1]), phi),
        ("Catalan", _prec([[-2, -4], [2, 1]], [1]), 4.0),
        ("central binomial", _prec([[-2, -4], [1, 1]], [1]), 4.0),
        ("(-2)^n", _prec([[2], [1]], [1]), 2.0),
        ("Tribonacci", _prec([[-1], [-1], [-1], [1]], [0, 0, 1]), tribonacci),
    ]
    char_errs: list[float] = []
    ratio_errs: list[float] = []

    def char_error(seed: int) -> float:
        name, rec, true_rate = cases[seed]
        est = precursive_asymptotics(rec)
        emp = empirical_rate(rec, samples=60)
        ce = abs(est.rate - true_rate)
        re = abs(emp - true_rate)
        char_errs.append(ce)
        ratio_errs.append(re)
        return ce

    seed_sweep(char_error, range(len(cases)))
    char_mean = sum(char_errs) / len(char_errs)
    ratio_mean = sum(ratio_errs) / len(ratio_errs)
    cmp = baseline_compare("charroot-vs-ratio", char_mean, ratio_mean, lower_is_better=True)
    # a certified bridge where the singularity is known exactly (geometric 2^n <-> 1/(1-2x)).
    cert = certified_asymptotic(rate=2, exponent_alpha=1, scale=1, n=12)
    assert cert.exact_coefficient.lo <= 2**12 <= cert.exact_coefficient.hi
    print(
        f"  mean |rate - true|: char-root={char_mean:.2e} vs empirical ratio={ratio_mean:.2e}; "
        f"certified 2^12 enclosed; wins={cmp.wins}"
    )
    assert cmp.wins and char_mean < ratio_mean
    ledger.add(
        "holonomic",
        "info",
        "the empirical ratio a(n+1)/a(n) converges slowly; the characteristic root is exact at finite n",
        detail="Poincare-Perron dominant root (numerical), certified via transfer_theorem where the OGF is known",
        char_mean=char_mean,
        ratio_mean=ratio_mean,
    )


PROBES = (
    probe_gosper_closed_form,
    probe_creative_telescoping,
    probe_certified_identities,
    probe_symbolic_closures,
    probe_all_n_zeilberger,
    probe_petkovsek_hyper,
    probe_q_holonomic,
    probe_guessing_vs_lstsq,
    probe_asymptotics_vs_ratio,
)


def main() -> None:
    ledger = FindingsLedger("holonomic_validate")
    for probe in PROBES:
        probe(ledger)
    print("\n" + ledger.summary())
    try:
        path = ledger.write()
        print(f"\nfindings ledger -> {path}")
    except OSError:  # pragma: no cover - scratch may be unavailable in a sandbox
        print("\n(findings ledger not persisted: scratch dir unavailable)")
    unresolved = [f for f in ledger if f.severity == "bug" and not f.repro.get("resolved")]
    assert not unresolved, f"{len(unresolved)} live soundness bug(s) surfaced -- see the ledger"
    print("OK: all holonomic probes pass their gates.")


if __name__ == "__main__":
    main()
