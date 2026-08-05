# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Data-driven refinement smoke for W5-ext rigorous number theory.

Run:

    pip install "omnibias-difference[test]"
    python docs/examples/numbertheory_validate.py

Probes the Hurwitz zeta, polylogarithm, Lerch transcendent, and exact Dirichlet
``L`` negative-integer values under the ``omnibias-dev-empirical-validation``
gates -- grid-and-random enclosure soundness (a sweep of ``K >= 8`` seeds, each
certified enclosure must contain the ``mpmath`` value), an ``mpmath`` oracle, and
a best-in-class comparison against the named naive-truncated-series baseline.

Honesty: exact special values are **closed-form** rationals (pi-multiples where
relevant); the Euler-Maclaurin continuation into ``Re(s) <= 1`` is **numerical**
(verified enclosure). The Generalized Riemann Hypothesis remains an **external
obligation, never inferred** -- a tight enclosure near a zero is not a statement
about zero location.
"""

from __future__ import annotations

from fractions import Fraction

try:
    import mpmath
except ImportError:  # pragma: no cover
    print("this smoke needs mpmath: pip install 'omnibias-difference[test]'")
    raise SystemExit(0) from None

from omnibias.core.verified.dirichlet import dirichlet_l_negative_integer
from omnibias.core.verified.hurwitz import hurwitz_zeta, hurwitz_zeta_negative_integer
from omnibias.core.verified.polylog import lerch_transcendent, polylog_enclosure
from omnibias.difference.validation import FindingsLedger, baseline_compare


def probe_hurwitz(ledger: FindingsLedger) -> None:
    """Hurwitz zeta: exact negatives + a continuation the naive p-series cannot reach."""
    print("=== W5-ext: Hurwitz zeta (exact negatives + EM continuation) ===")
    # (1) exact negative-integer values enclose mpmath (K=8 (n,a) pairs).
    with mpmath.workdps(40):
        pairs = [(0, Fraction(1, 2)), (1, Fraction(1)), (2, Fraction(1, 3)), (3, Fraction(3, 4)),
                 (4, Fraction(2, 5)), (5, Fraction(1)), (1, Fraction(1, 2)), (2, Fraction(2, 3))]
        for n, a in pairs:
            got = hurwitz_zeta_negative_integer(n, a)
            ref = float(mpmath.zeta(-n, mpmath.mpf(a.numerator) / a.denominator))
            assert got.contains(ref), (n, a)
    print("  zeta(-n, a) = -B_{n+1}(a)/(n+1) exact-encloses mpmath (K=8)")

    # (2) numerical continuation encloses mpmath in Re(s) < 1 across K=8 seeds.
    seeds = [complex(0.5, 3.0), complex(0.25, 5.0), complex(-0.5, 2.0), complex(0.75, 1.0),
             complex(0.5, 14.13), complex(-1.5, 4.0), complex(0.1, 8.0), complex(0.9, -6.0)]
    max_w = 0.0
    with mpmath.workdps(40):
        for s in seeds:
            enc = hurwitz_zeta(s, 0.7, num_sum_terms=30, order=6)
            ref = complex(mpmath.zeta(mpmath.mpc(s.real, s.imag), 0.7))
            if not (enc.re.contains(ref.real) and enc.im.contains(ref.imag)):
                ledger.add("numbertheory", "bug", "Hurwitz EM enclosure escapes mpmath", s=str(s))
            assert enc.re.contains(ref.real) and enc.im.contains(ref.imag)
            max_w = max(max_w, enc.re.width, enc.im.width)
    print(f"  Hurwitz EM continues into Re(s)<1 across 8 seeds (max width {max_w:.1e})")

    # (3) baseline: the naive truncated Dirichlet sum diverges for Re(s)<1 (cannot
    # even estimate the value there), so EM wins by existing.
    s = complex(0.5, 3.0)
    naive_20 = abs(complex(sum(complex(mpmath.power(k + 0.7, -mpmath.mpc(s.real, s.imag))) for k in range(20))))
    naive_60 = abs(complex(sum(complex(mpmath.power(k + 0.7, -mpmath.mpc(s.real, s.imag))) for k in range(60))))
    print(f"  naive |partial| at N=20 -> {naive_20:.3f}, N=60 -> {naive_60:.3f} (not converging)")
    cmp = baseline_compare("EM width vs naive-nonconvergence", max_w, abs(naive_60 - naive_20), lower_is_better=True)
    assert cmp.wins
    ledger.add(
        "numbertheory",
        "gap",
        "the naive Dirichlet p-series cannot evaluate Hurwitz zeta in Re(s)<1 (FIXED by EM)",
        detail="hurwitz_zeta uses Euler-Maclaurin (DLMF 25.11.5) with a rigorous remainder; the naive "
        "truncated sum diverges there. RH/GRH remain external obligations, never inferred",
        resolved=True,
    )


def probe_polylog_lerch(ledger: FindingsLedger) -> None:
    """Polylog / Lerch: certified tail bound gives a guaranteed error bar the naive sum lacks."""
    print("=== W5-ext: polylogarithm + Lerch transcendent (certified tails) ===")
    cases = [(2.0, 0.5), (3.0, -0.6), (1.0, 0.4), (2.0, 0.5j), (0.5, 0.3),
             (-1.0, 0.2), (complex(2, 1), 0.4), (2.0, complex(0.3, 0.3))]
    with mpmath.workdps(40):
        for s, z in cases:
            enc = polylog_enclosure(s, z, num_terms=140)
            sr = mpmath.mpc(s.real, s.imag) if isinstance(s, complex) else mpmath.mpf(s)
            zr = mpmath.mpc(getattr(z, "real", z), getattr(z, "imag", 0.0))
            ref = complex(mpmath.polylog(sr, zr))
            assert enc.re.contains(ref.real) and enc.im.contains(ref.imag), (s, z)
        for z, s, a in [(0.3, 2.0, 1.0), (-0.5, 1.5, 0.5), (complex(0.2, 0.2), complex(2, 1), 2.0)]:
            enc = lerch_transcendent(z, s, a, num_terms=140)
            ref = complex(mpmath.lerchphi(mpmath.mpc(getattr(z, "real", z), getattr(z, "imag", 0.0)),
                                          mpmath.mpc(getattr(s, "real", s), getattr(s, "imag", 0.0)), a))
            assert enc.re.contains(ref.real) and enc.im.contains(ref.imag), (z, s, a)
    print("  Li_s(z) and Phi(z,s,a) enclose mpmath across K=8+3 seeds")

    # baseline: certified tail bound vs a bare truncation with NO error bar.
    with mpmath.workdps(40):
        enc = polylog_enclosure(2.0, 0.5, num_terms=40)
        ref = float(mpmath.polylog(2, 0.5))
    print(f"  Li_2(0.5) certified re-width {enc.re.width:.2e} contains truth={enc.re.contains(ref)} (naive: no bound)")
    assert enc.re.contains(ref)
    ledger.add(
        "numbertheory",
        "info",
        "polylog/Lerch certified tail gives a guaranteed error bar the naive truncation lacks",
    )


def probe_dirichlet_l(ledger: FindingsLedger) -> None:
    """Dirichlet L: exact negative-integer values from generalized Bernoulli numbers."""
    print("=== W5-ext: exact Dirichlet L(1-n, chi) special values ===")
    chi4 = (0, 1, 0, -1)
    # L(0)=1/2, L(-2)=E_2/2=-1/2, L(-4)=E_4/2=5/2, L(-6)=E_6/2=-61/2.
    expect = {1: Fraction(1, 2), 3: Fraction(-1, 2), 5: Fraction(5, 2), 7: Fraction(-61, 2)}
    for n, val in expect.items():
        got = dirichlet_l_negative_integer(n, chi4)
        assert got.contains(float(val)), (n, got, val)
    print(f"  L(1-n, chi_4) = E_{{n-1}}/2 exact for n in {sorted(expect)} (Euler-number relation)")
    ledger.add(
        "numbertheory",
        "info",
        "exact L(1-n, chi) via generalized Bernoulli numbers; GRH remains an external obligation",
    )


PROBES = (probe_hurwitz, probe_polylog_lerch, probe_dirichlet_l)


def main() -> None:
    ledger = FindingsLedger("numbertheory_validate")
    for probe in PROBES:
        probe(ledger)
    print("\n" + ledger.summary())
    try:
        path = ledger.write()
        print(f"\nfindings ledger -> {path}")
    except OSError:  # pragma: no cover
        print("\n(findings ledger not persisted: scratch dir unavailable)")
    unresolved = [f for f in ledger if f.severity == "bug" and not f.repro.get("resolved")]
    assert not unresolved, f"{len(unresolved)} live soundness bug(s) surfaced -- see the ledger"
    print("OK: all number-theory probes pass their gates.")


if __name__ == "__main__":
    main()
