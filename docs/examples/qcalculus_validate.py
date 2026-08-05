# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Data-driven / verified / best-in-class smoke for omnibias-qcalculus.

Run:

    pip install "omnibias-qcalculus[test]"
    python docs/examples/qcalculus_validate.py

Each *probe* turns a q-calculus capability into an instrumented experiment under the
``omnibias-dev-empirical-validation`` gates -- an mpmath high-precision oracle, a
best-in-class comparison against a named baseline, and a ``K >= 8`` seed sweep -- and
records its gaps / flaws into a shared
:class:`~omnibias.difference.validation.FindingsLedger` (JSON written to scratch, never
the repo tree). The assertions here are the CI gate.

This exercises the **``q -> 1`` limit** (q-calculus collapsing to ordinary calculus), a
*distinct* limit from the ``delta -> 0`` founding bias-collapse of ``omnibias-difference``
and the ``beta -> inf`` feasibility penalty elsewhere -- never conflated. Honesty labels:
**closed-form / exact** (q-numbers, Gaussian binomials, q-Bernoulli / q-Euler), and
**numerical (certified)** (the q-exponential / basic-hypergeometric enclosures).
"""

from __future__ import annotations

import math
from fractions import Fraction

from omnibias.difference.validation import (
    FindingsLedger,
    baseline_compare,
    require_mpmath,
    seed_sweep,
)
from omnibias.qcalculus import (
    basic_hypergeometric,
    basic_hypergeometric_enclosure,
    q_bernoulli,
    q_binomial,
    q_binomial_poly,
    q_bracket,
    q_euler,
    q_exp_enclosure,
    q_factorial,
)


def probe_q_to_one_collapse(ledger: FindingsLedger) -> None:
    """The ``q -> 1`` limit recovers ordinary combinatorics / special numbers, exactly."""
    print("=== q -> 1 collapse (distinct from delta -> 0): exact reductions ===")
    from omnibias.difference import bernoulli_number, euler_number

    for n in range(9):
        assert q_bracket(n, 1) == n
        assert q_factorial(n, 1) == math.factorial(n)
        assert q_bernoulli(n, 1) == bernoulli_number(n)
    for n in range(6):
        for k in range(n + 1):
            assert q_binomial(n, k, 1) == math.comb(n, k)
    for n in range(0, 9, 2):
        assert q_euler(n, 1) == euler_number(n)
    print("  [n]_q -> n, [n]_q! -> n!, q-binomial -> C(n,k), q-B/q-E -> classical: exact")
    ledger.add(
        "qcalc",
        "info",
        "q -> 1 collapse recovers ordinary calculus exactly (distinct from delta->0 / beta->inf)",
    )


def probe_gaussian_binomial_exact(ledger: FindingsLedger) -> None:
    """Exact integer-polynomial Gaussian binomials beat float evaluation (best-in-class)."""
    print("=== Gaussian binomial: exact integer polynomial vs float ===")
    max_float_err = 0.0
    for n in range(8):
        for k in range(n + 1):
            poly = q_binomial_poly(n, k)
            assert poly == poly[::-1]  # palindrome (closed-form structure)
            # Exact rational value at a numeric q equals the polynomial evaluation.
            q = Fraction(3, 7)
            exact = q_binomial(n, k, q)
            poly_val = sum((Fraction(c) * q**i for i, c in enumerate(poly)), Fraction(0))
            assert exact == poly_val
            # A float Horner evaluation drifts; the exact rational never does.
            qf = 3.0 / 7.0
            float_val = 0.0
            for c in reversed(poly):
                float_val = float_val * qf + c
            max_float_err = max(max_float_err, abs(float_val - float(exact)))
    cmp = baseline_compare("gaussian-binomial-exactness", 0.0, max_float_err, lower_is_better=True)
    print(f"  exact rational error 0 vs float Horner error {max_float_err:.2e}: wins={cmp.wins}")
    assert cmp.wins
    ledger.add("qcalc", "info", "exact integer-polynomial Gaussian binomial has zero rounding error")


def probe_certified_qhyper(ledger: FindingsLedger) -> None:
    """Certified enclosures contain the mpmath oracle; a naive +/- first-omitted band does not."""
    print("=== certified q-exponential / basic-hypergeometric enclosures ===")
    mp = require_mpmath()

    # (1) q_exp enclosure: contains the oracle across K = 8 seeds; the naive band fails.
    def qexp_width(seed: int) -> float:
        q = Fraction(1, 4) + Fraction(seed, 40)  # 0.25 .. 0.425
        z = Fraction(1, 10) + Fraction(seed, 50)  # 0.10 .. 0.24
        iv = q_exp_enclosure(z, q, terms=6)
        # oracle e_q(z) = sum z^n / [n]_q!.
        with mp.workdps(50):
            oracle = _mp_qexp(mp, z, q)
        assert iv.lo <= oracle <= iv.hi, f"seed {seed}: certified band escaped"
        # naive band S_6 +/- |t_6| (a common but UNSOUND heuristic for positive series).
        partial, first_omitted = _qexp_partial(z, q, 6)
        naive_lo, naive_hi = partial - first_omitted, partial + first_omitted
        naive_ok = naive_lo <= oracle <= naive_hi
        return 0.0 if naive_ok else 1.0  # count naive failures

    naive_failures = seed_sweep(qexp_width, range(8))
    print(
        f"  q_exp certified band sound over K=8; naive +/-first-term failures: "
        f"{int(naive_failures['mean'] * naive_failures['n'])}/8"
    )
    assert naive_failures["max"] == 1.0, "expected the naive heuristic band to fail at least once"

    # (2) basic hypergeometric 2phi1: certified enclosure vs the direct-summation baseline.
    a = [Fraction(1, 5), Fraction(3, 10)]
    b = [Fraction(2, 5)]
    for seed in range(8):
        q = Fraction(3, 10) + Fraction(seed, 40)
        z = Fraction(1, 20) + Fraction(seed, 100)
        iv = basic_hypergeometric_enclosure(a, b, q, z, terms=40)
        with mp.workdps(50):
            oracle = float(
                mp.qhyper(
                    [mp.mpf(x.numerator) / x.denominator for x in a],
                    [mp.mpf(x.numerator) / x.denominator for x in b],
                    mp.mpf(q.numerator) / q.denominator,
                    mp.mpf(z.numerator) / z.denominator,
                )
            )
        assert iv.lo <= oracle <= iv.hi, f"seed {seed}: 2phi1 certified band escaped"
        baseline = basic_hypergeometric(
            [float(x) for x in a], [float(x) for x in b], float(q), float(z), terms=40
        )
        assert iv.lo <= baseline <= iv.hi  # the baseline lands inside the guaranteed band
    print("  2phi1 certified enclosure contains mpmath oracle AND the float baseline over K=8")
    ledger.add(
        "qcalc",
        "flaw",
        "naive +/-first-omitted-term band is unsound for positive q-series (certified band wins)",
        detail="the geometric-tail enclosure guarantees containment; the heuristic band does not",
    )


def _mp_qexp(mp: object, z: Fraction, q: Fraction) -> float:
    """High-precision e_q(z) = sum_n z^n / [n]_q! (the oracle)."""
    zf = mp.mpf(z.numerator) / z.denominator  # type: ignore[attr-defined]
    qf = mp.mpf(q.numerator) / q.denominator  # type: ignore[attr-defined]
    total = mp.mpf(0)  # type: ignore[attr-defined]
    for n in range(200):
        bracket_fac = mp.mpf(1)  # type: ignore[attr-defined]
        for k in range(1, n + 1):
            bracket_fac *= (1 - qf**k) / (1 - qf)
        total += zf**n / bracket_fac
    return float(total)


def _qexp_partial(z: Fraction, q: Fraction, terms: int) -> tuple[float, float]:
    """Return (partial sum S_terms, |t_terms|) for e_q(z), in float."""
    partial = 0.0
    term = 1.0
    for n in range(terms):
        partial += term
        term *= float(z) / float(q_bracket(n + 1, q))
    return partial, abs(term)


def probe_q_umbral(ledger: FindingsLedger) -> None:
    """q-umbral / q-Sheffer: the exact ``Q s_n = [n]_q s_{n-1}`` recurrence + the q -> 1 collapse."""
    print("=== q-umbral / q-Sheffer: exact recurrence, q -> 1 collapse, exact-vs-float ===")
    from omnibias.difference import appell_sequence, sheffer_sequence
    from omnibias.qcalculus import q_derivative_poly
    from omnibias.qcalculus.umbral import (
        q_appell_sequence,
        q_delta_operator_apply,
        q_sheffer_sequence,
    )

    g = [Fraction(1), Fraction(1, 2), Fraction(-1, 3), Fraction(1, 4)]
    f = [Fraction(0), Fraction(1), Fraction(1, 2), Fraction(-1, 5), Fraction(1, 6)]
    n_max = 6

    def _strip(coeffs: tuple[Fraction, ...]) -> tuple[Fraction, ...]:
        out = list(coeffs)
        while len(out) > 1 and out[-1] == 0:
            out.pop()
        return tuple(out)

    def _residual(a: tuple[Fraction, ...], b: tuple[Fraction, ...]) -> float:
        width = max(len(a), len(b))
        aa = list(a) + [Fraction(0)] * (width - len(a))
        bb = list(b) + [Fraction(0)] * (width - len(b))
        return float(max((abs(x - y) for x, y in zip(aa, bb, strict=True)), default=Fraction(0)))

    # (1) The exact closed-form q-Sheffer recurrence Q s_n = [n]_q s_{n-1} (Q = f(D_q))
    # holds to zero residual across K = 8 distinct rational q -- not merely q -> 1.
    def sheffer_recurrence_residual(seed: int) -> float:
        q = Fraction(1, 4) + Fraction(seed, 12)  # 1/4 .. 11/12, all != 1
        seq = q_sheffer_sequence(g, f, n_max, q)
        worst = 0.0
        for n in range(1, n_max + 1):
            lhs = _strip(q_delta_operator_apply(f, seq[n], q))
            rhs = _strip(tuple(q_bracket(n, q) * c for c in seq[n - 1]))
            worst = max(worst, _residual(lhs, rhs))
        return worst

    sweep = seed_sweep(sheffer_recurrence_residual, range(8))
    print(f"  q-Sheffer recurrence Q s_n=[n]_q s_{{n-1}} residual over K=8 q: max {sweep['max']:.0f} (0==exact)")
    if sweep["max"] != 0.0:
        ledger.add("qumbral", "bug", "q-Sheffer recurrence residual non-zero", max_resid=sweep["max"])
    assert sweep["max"] == 0.0

    # (2) q-Appell derivative property D_q p_n = [n]_q p_{n-1} (exact at a fixed rational q).
    q = Fraction(2, 3)
    constants = [Fraction(1), Fraction(2), Fraction(-1), Fraction(3), Fraction(0), Fraction(-2), Fraction(1)]
    appell = q_appell_sequence(constants, q)
    for n in range(1, len(appell)):
        lhs = _strip(q_derivative_poly(appell[n], q))
        rhs = _strip(tuple(q_bracket(n, q) * c for c in appell[n - 1]))
        assert _residual(lhs, rhs) == 0.0
    print("  q-Appell property D_q p_n=[n]_q p_{n-1}: exact at q=2/3")

    # (3) Headline q -> 1 collapse: every op reduces to the classical difference umbral.
    one = Fraction(1)
    assert [tuple(p) for p in q_sheffer_sequence(g, f, 5, one)] == [tuple(p) for p in sheffer_sequence(g, f, 5)]
    assert [tuple(p) for p in q_appell_sequence(constants, one)] == [tuple(p) for p in appell_sequence(constants)]
    print("  q -> 1: q_sheffer_sequence == difference.sheffer_sequence, q_appell == appell (exact)")
    ledger.add(
        "qumbral",
        "info",
        "exact q-Sheffer recurrence Q s_n=[n]_q s_{n-1}; collapses to the classical umbral at q -> 1",
    )

    # (4) Best-in-class: exact Fraction q-Appell coefficients vs a float q-binomial pipeline.
    q = Fraction(3, 7)
    qf = 3.0 / 7.0
    exact = q_appell_sequence(constants, q)

    def _q_factorial_float(m: int) -> float:
        value = 1.0
        for i in range(1, m + 1):
            value *= (1.0 - qf**i) / (1.0 - qf)
        return value

    max_float_err = 0.0
    for n in range(len(constants)):
        for k in range(n + 1):
            fq_binom = _q_factorial_float(n) / (_q_factorial_float(k) * _q_factorial_float(n - k))
            float_coeff = fq_binom * float(constants[n - k])
            max_float_err = max(max_float_err, abs(float_coeff - float(exact[n][k])))
    cmp = baseline_compare("exact q-Appell coeffs vs float q-binomial", 0.0, max_float_err, lower_is_better=True)
    print(f"  exact rational q-Appell coeffs vs float q-binomial: max err {max_float_err:.2e} wins={cmp.wins}")
    assert cmp.wins
    ledger.add(
        "qumbral",
        "flaw",
        "float q-binomial evaluation drifts; exact Fraction q-Sheffer has zero rounding error",
        detail="[n choose k]_q at rational q are non-integer rationals; float Horner/ratio accumulates error",
        max_coeff_err=max_float_err,
    )


PROBES = (
    probe_q_to_one_collapse,
    probe_gaussian_binomial_exact,
    probe_certified_qhyper,
    probe_q_umbral,
)


def main() -> None:
    ledger = FindingsLedger("qcalculus_validate")
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
    print("OK: all qcalculus probes pass their gates.")


if __name__ == "__main__":
    main()
