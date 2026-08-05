# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Data-driven refinement smoke for omnibias-difference (the all-tiers program).

Run:

    pip install "omnibias-difference[test]"
    python docs/examples/difference_validate.py

Each *probe* turns a tier capability into an instrumented experiment under the
``omnibias-dev-empirical-validation`` gates -- grid-and-random enclosure
soundness, an mpmath high-precision oracle, and a best-in-class comparison
against a named baseline -- and records the gaps / flaws / bugs it exposes into a
shared :class:`~omnibias.difference.validation.FindingsLedger`. The ledger JSON
is written to scratch (never the repo tree); the assertions here are the CI gate.

This exercises the ``delta -> 0`` founding collapse (a smooth ``sigma^(K-1)``
derivative), never the ``beta -> inf`` feasibility penalty. Honesty labels stay
**closed-form** (towers + exact coefficients) and **numerical** (finite-difference
estimate and the mpmath reference); there is no autodiff-exact path here.
"""

from __future__ import annotations

import math
from fractions import Fraction

try:
    import mpmath
except ImportError:  # pragma: no cover - the [test] extra ships mpmath
    print("this smoke needs mpmath: pip install 'omnibias-difference[test]'")
    raise SystemExit(0) from None

from omnibias.core.proof.lean_check import lean_check_available
from omnibias.core.verified import (
    E_IV,
    PI_IV,
    digamma_iv,
    euler_maclaurin_sum,
    log_gamma_iv,
)
from omnibias.core.verified.interval import Interval
from omnibias.difference import (
    bernoulli_number,
    certified_derivative_enclosure,
    certified_fd_error,
    check_derivative_certificate,
    euler_number,
    finite_difference_estimate,
)
from omnibias.difference.validation import (
    FindingsLedger,
    baseline_compare,
    enclosure_soundness,
    high_precision_derivative,
    seed_sweep,
)

# mpmath references for the supported smooth activations (the numerical oracle).
_MP_FUNC = {
    "tanh": mpmath.tanh,
    "sigmoid": lambda z: 1 / (1 + mpmath.e ** (-z)),
    "gaussian": lambda z: mpmath.e ** (-z * z / 2),
}


def probe_constants(ledger: FindingsLedger) -> None:
    """Phase 0: the newly exported certified constants enclose pi / e."""
    print("=== Phase 0: certified constants PI_IV / E_IV ===")
    with mpmath.workdps(60):
        assert PI_IV.lo <= mpmath.pi <= PI_IV.hi
        assert E_IV.lo <= mpmath.e <= E_IV.hi
    print(f"  PI_IV width {PI_IV.width:.2e}  E_IV width {E_IV.width:.2e}  (both enclose truth)")
    ledger.add("phase0", "info", "PI_IV/E_IV enclose pi/e", pi_w=PI_IV.width, e_w=E_IV.width)


def probe_tower_soundness(ledger: FindingsLedger) -> None:
    """Tier 1 anchor: the closed-form tower encloses a grid AND random sample of truth."""
    print("=== tower soundness (grid + random) ===")
    for name in ("tanh", "gaussian"):
        for order in (2, 4):
            box = Interval(0.2, 0.8)
            enclosure = certified_derivative_enclosure(name, box, order).value
            report = enclosure_soundness(
                enclosure,
                lambda x, n=name, o=order: high_precision_derivative(_MP_FUNC[n], x, o),
                box,
                grid=25,
                random_samples=40,
            )
            status = "sound" if report.sound else f"ESCAPES ({report.max_escape:.2e})"
            print(f"  {name}^({order}) over {box!r}: {status}, width {enclosure.width:.2e}")
            if not report.sound:
                ledger.add(
                    "tower",
                    "bug",
                    f"{name}^({order}) enclosure escapes truth",
                    max_escape=report.max_escape,
                )
            assert report.sound


def probe_fd_beats_naive(ledger: FindingsLedger) -> None:
    """Tier 1: FD certificate sandwich holds; closed form beats naive 1/delta^m FD."""
    print("=== certified FD sandwich + closed-form-beats-naive-FD ===")
    name, z, order = "tanh", 0.6, 3
    true = high_precision_derivative(mpmath.tanh, z, order)
    enclosure = certified_derivative_enclosure(name, z, order).value

    for delta in (1e-1, 1e-2, 1e-3):
        cert = certified_fd_error(name, z, order, delta, "central")
        assert cert.certified
        assert abs(cert.estimate - true) <= cert.error_bound + 1e-12
    print("  FD certificate sandwich holds for delta in {1e-1,1e-2,1e-3}")

    naive_err = abs(finite_difference_estimate(name, z, order, 1e-5, "central").estimate - true)
    closed_err = max(enclosure.lo - true, true - enclosure.hi, 0.0)
    cmp = baseline_compare("closed-form vs naive-FD (delta=1e-5)", closed_err, naive_err)
    print(f"  closed-form err {closed_err:.2e}  vs  naive-FD err {naive_err:.2e}  wins={cmp.wins}")
    if not cmp.wins:
        ledger.add("fd", "flaw", "closed form did not beat naive FD", ratio=cmp.ratio)
    assert cmp.wins


def probe_special_numbers(ledger: FindingsLedger) -> None:
    """Tier 1/2: special numbers read off the towers match the mpmath reference."""
    print("=== special numbers off the towers vs mpmath ===")
    with mpmath.workdps(50):
        for n in (2, 6, 12):
            got = bernoulli_number(n)
            ref = mpmath.bernoulli(n)
            got_mpf = mpmath.mpf(got.numerator) / mpmath.mpf(got.denominator)
            assert abs(got_mpf - ref) < mpmath.mpf(10) ** (-40)
    euler_row = [euler_number(n) for n in range(9)]
    assert euler_row == [1, 0, -1, 0, 5, 0, -61, 0, 1385]
    print(f"  Bernoulli B_2/B_6/B_12 match mpmath; E_0..8 = {euler_row}")
    ledger.add("special", "info", "Bernoulli/Euler match mpmath reference")


def probe_euler_maclaurin(ledger: FindingsLedger) -> None:
    """W1 (Tier 1): certified Euler-Maclaurin -> log_gamma / digamma + the engine."""
    print("=== W1: certified Euler-Maclaurin -> log_gamma / digamma ===")
    for name, iv_fn, mp_fn in (
        ("log_gamma", log_gamma_iv, mpmath.loggamma),
        ("digamma", digamma_iv, mpmath.digamma),
    ):
        box = Interval(2.0, 5.0)
        enclosure = iv_fn(box)
        report = enclosure_soundness(
            enclosure, lambda x, f=mp_fn: float(f(x)), box, grid=25, random_samples=40
        )
        print(f"  {name} over {box!r}: {'sound' if report.sound else 'ESCAPES'}")
        if not report.sound:
            ledger.add("w1", "bug", f"{name} box enclosure escapes truth", max_escape=report.max_escape)
        assert report.sound
        point = iv_fn(Interval.point(3.25))
        with mpmath.workdps(40):
            true = float(mp_fn(3.25))
        assert point.lo <= true <= point.hi
        print(f"    {name}(3.25) point width {point.width:.2e}")

    def deriv(k: int, x: Interval) -> Interval:
        coeff = Interval.from_rational(Fraction((-1) ** k * math.factorial(k + 1)))
        return coeff * x.pow_int(k + 2).reciprocal()

    a, b = 4, 30
    integral = Interval.from_rational(Fraction(1, a) - Fraction(1, b))
    exact = float(sum(Fraction(1, k * k) for k in range(a, b + 1)))
    em = euler_maclaurin_sum(deriv, integral, a, b, terms=4)
    assert em.lo <= exact <= em.hi
    em_err = max(em.lo - exact, exact - em.hi, 0.0)
    trapezoid = float(integral.mid) + 0.5 * (1.0 / a**2 + 1.0 / b**2)
    cmp = baseline_compare(
        "Euler-Maclaurin vs trapezoid (sum 1/k^2)", em_err, abs(trapezoid - exact)
    )
    print(f"  EM err {em_err:.2e}  vs trapezoid err {abs(trapezoid - exact):.2e}  wins={cmp.wins}")
    if not cmp.wins:
        ledger.add("w1", "flaw", "Euler-Maclaurin did not beat trapezoid", ratio=cmp.ratio)
    assert cmp.wins
    ledger.add("w1", "info", "certified log_gamma/digamma enclose mpmath; EM beats trapezoid")


def probe_discrete_discovery(ledger: FindingsLedger) -> None:
    """W3 (Tier 1): exact P-recursive recurrence discovery vs a least-squares baseline."""
    print("=== W3: discrete recurrence discovery (exact) vs least-squares baseline ===")
    try:
        from math import comb, factorial

        import numpy as np
        from omnibias.symbolic import (
            build_difference_relation_library,
            discover_recurrence,
            discover_recurrence_least_squares,
            polynomial_from_samples,
            verify_binomial_recurrence,
        )
    except ImportError:
        print("  omnibias-symbolic not installed: skipping W3 discrete-discovery probe")
        ledger.add("w3", "info", "omnibias-symbolic absent: discrete-discovery probe skipped")
        return

    # Exact finder recovers C-finite (Fibonacci) and P-recursive (Catalan, factorial).
    fib = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377]
    catalan = [comb(2 * n, n) // (n + 1) for n in range(13)]
    fac = [factorial(n) for n in range(11)]
    for label, seq, want_order, want_deg in (
        ("Fibonacci", fib, 2, 0),
        ("Catalan", catalan, 1, 1),
        ("factorial", fac, 1, 1),
    ):
        rel = discover_recurrence(seq)
        assert rel is not None and rel.max_abs_residual(seq) == 0
        assert (rel.order, rel.index_degree) == (want_order, want_deg)
        print(f"  {label}: {rel.pretty()}  (exact, residual 0)")

    # Baseline: monic least-squares cannot represent Catalan's (n+1) leading coeff.
    design, _names, target = build_difference_relation_library(catalan, order=1, index_degree=1)
    ls_eq = discover_recurrence_least_squares(catalan, order=1, index_degree=1)
    ls_resid = float(np.max(np.abs(ls_eq.predict(design) - target)) / max(np.max(np.abs(target)), 1.0))
    cmp = baseline_compare("exact vs least-squares recurrence (Catalan)", 0.0, ls_resid)
    print(f"  Catalan baseline: exact resid 0  vs  least-squares rel resid {ls_resid:.2e}  wins={cmp.wins}")
    if ls_resid > 1e-9:
        ledger.add(
            "w3",
            "flaw",
            "monic least-squares cannot recover a non-constant leading coefficient (Catalan)",
            detail="a_n regression target forces p_0(n)=1; the exact rational null-space finder is the fix",
            ls_rel_resid=ls_resid,
        )
    assert cmp.wins

    # Bell / partition are not finitely P-recursive: the finder correctly returns None.
    bell = [1, 1, 2, 5, 15, 52, 203, 877, 4140, 21147, 115975]
    partition = [1, 1, 2, 3, 5, 7, 11, 15, 22, 30, 42]
    assert discover_recurrence(bell, max_order=4, max_index_degree=3) is None
    assert discover_recurrence(partition, max_order=4, max_index_degree=3) is None
    assert verify_binomial_recurrence(bell)
    print("  Bell / p(n): no fixed-order P-recursive relation (correct); Bell binomial recurrence holds")
    ledger.add(
        "w3",
        "gap",
        "fixed-order P-recursive discovery cannot capture non-holonomic sequences",
        detail="Bell numbers / partition p(n) are not P-recursive; their laws are full-history "
        "convolutions (binomial / pentagonal), verified separately -- discover_recurrence "
        "returns None rather than forcing a false finite-order fit",
    )

    # Faulhaber: exact Newton-series polynomial reads exact rationals; float fit cannot.
    power = 2
    faul = [sum(k**power for k in range(1, i + 1)) for i in range(15)]
    exact_coeffs = polynomial_from_samples(faul)  # (0, 1/6, 1/2, 1/3)
    assert [str(c) for c in exact_coeffs] == ["0", "1/6", "1/2", "1/3"]
    float_coeffs = np.polyfit(np.arange(len(faul)), np.asarray(faul, dtype=float), power + 1)[::-1]
    max_read_err = max(abs(float(c) - fc) for c, fc in zip(exact_coeffs, float_coeffs, strict=True))
    cmp2 = baseline_compare("exact-rational vs float polynomial coefficients (Faulhaber)", 0.0, max_read_err)
    print(f"  Faulhaber p=2 coeffs {tuple(str(c) for c in exact_coeffs)}; float read err {max_read_err:.2e} wins={cmp2.wins}")
    if max_read_err > 0.0:
        ledger.add(
            "w3",
            "flaw",
            "float polynomial fit cannot read exact rational Faulhaber coefficients",
            detail="np.polyfit returns inexact 1/6, 1/3, ...; the exact Newton-series path is bit-exact",
            max_coeff_err=max_read_err,
        )
    assert cmp2.wins

    # K = 8 seeds: recover random order-2 C-finite recurrences exactly.
    def recover_random_cfinite(seed: int) -> float:
        import random as _random

        rng = _random.Random(seed)
        p, q = rng.randint(1, 4), rng.randint(1, 4)
        seq = [rng.randint(0, 3), rng.randint(1, 4)]
        for _ in range(14):
            seq.append(p * seq[-1] + q * seq[-2])
        rel = discover_recurrence(seq)
        if rel is None:
            return 1.0
        # exact recovery: a_n - p a_{n-1} - q a_{n-2} = 0 (up to a global scale).
        return float(rel.max_abs_residual(seq))
    sweep = seed_sweep(recover_random_cfinite, range(8))
    print(f"  8-seed random C-finite recovery: max residual {sweep['max']:.0f} (0 == exact)")
    if sweep["max"] != 0.0:
        ledger.add("w3", "bug", "random C-finite recurrence not recovered exactly", max_resid=sweep["max"])
    assert sweep["max"] == 0.0
    ledger.add("w3", "info", "exact recurrence discovery recovers Fibonacci/Catalan/factorial + Faulhaber; beats LS")


def probe_analytic_combinatorics(ledger: FindingsLedger) -> None:
    """W4 (Tier 2): CERTIFIED asymptotic enclosures + refined Bell saddle point."""
    print("=== W4: proof-carrying analytic combinatorics (certified enclosures) ===")
    from math import comb, expm1, log

    from omnibias.difference import (
        bell_asymptotic_relative_error,
        bell_dobinski_enclosure,
        bernoulli_enclosure,
        catalan_asymptotic,
        euler_enclosure,
        log_bell_number_asymptotic_refined,
        recommended_bell_fallback_n,
    )
    from omnibias.difference._core.asymptotics import log_bell_number_asymptotic
    from omnibias.difference._core.bernoulli import bernoulli_number
    from omnibias.difference._core.euler import euler_number
    from omnibias.difference._core.stirling import bell_number

    def encloses(iv: Interval, exact: Fraction | int) -> bool:
        return Fraction(iv.lo) <= Fraction(exact) <= Fraction(iv.hi)

    # (1) Certified enclosures provably contain the exact number -- the error bar the
    # float-only asymptotics lacked. Grid (K=8 even indices) + mpmath oracle.
    with mpmath.workdps(50):
        for n in (4, 8, 12, 16, 20, 24, 28, 32):
            b_iv, e_iv = bernoulli_enclosure(n), euler_enclosure(n)
            assert encloses(b_iv, bernoulli_number(n)) and encloses(e_iv, euler_number(n))
            assert b_iv.contains(float(mpmath.bernoulli(n)))
            assert e_iv.contains(float(mpmath.eulernum(n)))
        for n in (1, 5, 10, 17, 23, 30):
            assert encloses(bell_dobinski_enclosure(n), bell_number(n))
    print("  Bernoulli/Euler/Bell certified enclosures contain exact + mpmath (K=8)")
    ledger.add("w4", "info", "certified enclosures close the float-only-asymptotics honesty gap")

    # Finding (fixed): the value-space Euler/Bernoulli enclosure intermediately
    # overflowed float around n~130 before the pi-power was folded in first.
    for n in (130, 160, 170):
        iv = euler_enclosure(n)
        assert math.isfinite(iv.lo) and math.isfinite(iv.hi) and encloses(iv, euler_number(n))
    ledger.add(
        "w4",
        "bug",
        "value-space Euler/Bernoulli enclosure overflowed float at large n (FIXED)",
        detail="2^{n+2}*(2m)! overflowed ~n=130 though E_n is representable to ~180; "
        "fold the tiny pi^-(n+1) factor in before the large factorial",
        resolved=True,
    )

    # (2) Best-in-class: refined Moser-Wyman beats the raw r e^r = n baseline.
    n = 30
    le = log(bell_number(n))
    raw_err = abs(expm1(log_bell_number_asymptotic(n) - le))
    ref_err = abs(expm1(log_bell_number_asymptotic_refined(n) - le))
    cmp = baseline_compare("refined vs raw Bell saddle point (n=30)", ref_err, raw_err)
    improvement = raw_err / ref_err
    print(f"  Bell saddle n=30: refined {ref_err:.2e} vs raw {raw_err:.2e}  wins={cmp.wins} ({improvement:.0f}x)")
    if not cmp.wins:
        ledger.add("w4", "flaw", "refined Bell saddle point did not beat raw baseline", ratio=cmp.ratio)
    assert cmp.wins and improvement > 20.0

    # (3) Finding (fixed): the fallback threshold was unsound under the non-monotone
    # small-n error bump (first-crossing returned n=5, but n=6..34 exceed 1e-4).
    naive_first = next(k for k in range(2, 200) if bell_asymptotic_relative_error(k) <= 1e-4)
    sound_cut = recommended_bell_fallback_n(1e-4, n_max=200)
    violations = [k for k in range(sound_cut, 201) if bell_asymptotic_relative_error(k) > 1e-4]
    print(f"  fallback@1e-4: naive first-crossing={naive_first} (unsound), sound cutoff={sound_cut}, violations={len(violations)}")
    if naive_first < sound_cut:
        ledger.add(
            "w4",
            "flaw",
            "first-crossing fallback threshold is unsound for the non-monotone Bell error (FIXED)",
            detail="error dips at n=5 then rises through n~10; use last-exceedance not first-crossing",
            resolved=True,
        )
    assert not violations

    # (4) Catalan singularity analysis sanity (numerical asymptotic ratio -> 1).
    cat_ratio = catalan_asymptotic(100) / (comb(200, 100) // 101)
    print(f"  Catalan 4^n/(sqrt(pi) n^3/2) ratio at n=100: {cat_ratio:.4f}")
    assert 1.0 < cat_ratio < 1.02  # leading term; +9/(8n) correction ~1.1% at n=100
    ledger.add("w4", "info", "refined Bell saddle beats raw ~90x; sound fallback threshold; certified enclosures")


def probe_truncation_error(ledger: FindingsLedger) -> None:
    """W5 (Tier 2): remainder engine decoupled from the activation dictionary."""
    print("=== W5: certified truncation error (engine decoupled from the dictionary) ===")
    from math import exp as _exp

    from omnibias.core.verified.transcend import exp_iv
    from omnibias.difference import certified_fd_error_general

    # exp is NOT one of the nine built-in activations; the general engine certifies
    # it from a plain float + a derivative-tower oracle (exp^(k) = exp).
    exp_bound = lambda k, box: exp_iv(box)  # noqa: E731
    for order in (1, 2):
        cert = certified_fd_error_general(_exp, exp_bound, 0.6, order, 1e-2, "central", name="exp")
        true = _exp(0.6)
        assert cert.certified and cert.true_derivative_interval.contains(true)
    print("  certified_fd_error_general certifies exp^(1), exp^(2) (outside the 9-name dictionary)")
    ledger.add(
        "w5",
        "gap",
        "remainder engine was hard-coupled to the 9-name activation dictionary (FIXED)",
        detail="certified_fd_error_general(f_float, deriv_bound, ...) now accepts any "
        "DerivBound oracle; certified_fd_error is a thin wrapper binding sigma_tower_interval",
        resolved=True,
    )

    # If omnibias-verify is present, certify PDE stencils and match the empirical order.
    try:
        from omnibias.core.verified.sigma import sigma_tower_interval
        from omnibias.verify import (
            certified_laplacian_truncation,
            certified_stencil_truncation,
            measured_consistency_order,
        )
    except ImportError:
        print("  omnibias-verify not installed: skipping certified_stencil_truncation probe")
        ledger.add("w5", "info", "omnibias-verify absent: PDE-stencil truncation probe skipped")
        return

    import math

    sin_bound = lambda k, box: sigma_tower_interval("sin", box, k)[k]  # noqa: E731
    cos_bound = lambda k, box: sigma_tower_interval("cos", box, k)[k]  # noqa: E731
    steps = [0.2, 0.1, 0.05, 0.025]
    for deriv_order, stencil, want_p in ((1, "central", 2), (2, "central", 2), (1, "forward", 1)):
        cert = certified_stencil_truncation(math.sin, sin_bound, 0.4, deriv_order, 0.05, stencil)
        emp = measured_consistency_order(math.sin, sin_bound, 0.4, deriv_order, steps, stencil)
        cmp = baseline_compare(
            f"certified vs measured order (d{deriv_order} {stencil})",
            abs(emp - want_p),
            0.5,  # a naive quote could be off by half an order
        )
        print(f"  {stencil} d{deriv_order}: certified p={cert.consistency_order}, measured={emp:.2f}  matches={cmp.wins}")
        assert cert.consistency_order == want_p and abs(emp - want_p) < 0.2

    # Separable Laplacian of sin(x)+cos(y): sum of two 1-D certified stencils.
    lap = certified_laplacian_truncation([math.sin, math.cos], [sin_bound, cos_bound], [0.4, 0.9], 0.05)
    true = -math.sin(0.4) - math.cos(0.9)
    print(f"  2D separable Laplacian: enclosed={lap.laplacian_enclosure.contains(true)}  bound={lap.truncation_bound:.2e}")
    assert lap.laplacian_enclosure.contains(true) and lap.consistent
    ledger.add(
        "w5",
        "gap",
        "coupled multivariate stencils need a TaylorModelMV derivative oracle",
        detail="certified_laplacian_truncation handles axis-separable fields via the 1-D engine; "
        "fully coupled mixed partials are out of scope for the 1-D remainder math",
    )
    ledger.add("w5", "info", "certified PDE-stencil consistency order matches the measured empirical order")


def probe_certified_diff_lean(ledger: FindingsLedger) -> None:
    """W2 (Tier 1): sealed derivative certificate + Lean-kernel round-trip."""
    print("=== W2: certified derivative -> sealed certificate -> Lean bridge ===")
    available = lean_check_available()
    print(f"  Lean toolchain available: {available} (graceful degradation otherwise)")

    for z in (0.3 + 0.1 * i for i in range(8)):  # K = 8 sign-definite tanh' > 0 points
        verdict = check_derivative_certificate(certified_derivative_enclosure("tanh", z, 1))
        assert verdict.sign == "positive"
        assert verdict.obligation_generated
        assert verdict.sealed_ok
        # honesty: the flag is exactly the kernel verdict, never forged.
        assert verdict.theorem_prover_verified == verdict.lean.verified
    print("  8/8 sign-definite tanh' certificates seal + emit a Lean obligation")

    if not available:
        ledger.add("w2", "info", "Lean toolchain absent: bridge degraded gracefully (flag stays False)")

    # Documented obligation-coverage gap: a straddling enclosure has no sign obligation.
    straddle = check_derivative_certificate(certified_derivative_enclosure("tanh", 0.0, 2))
    if not straddle.obligation_generated:
        ledger.add(
            "w2",
            "gap",
            "interval-payload obligation path only covers sign-definite derivatives",
            detail="tanh''(0)=0 straddles zero, so no enclosed_quantity_pos/neg obligation is emitted",
        )
    print(f"  straddling tanh''(0): obligation_generated={straddle.obligation_generated} (documented gap)")
    ledger.add("w2", "info", "derivative certificates seal, round-trip, and never forge the formal flag")


def probe_dynamics_bridge(ledger: FindingsLedger) -> None:
    """W6 (Tier 2): sigma-tower -> validated-dynamics bridge (the missing jet ingest)."""
    print("=== W6: coefficient engine -> validated-dynamics bridge ===")
    try:
        from omnibias.core.verified.lohner import lohner_flow
        from omnibias.core.verified.ode import integrate_ivp
        from omnibias.dynamics import (
            discrete_periodic_point,
            sigma_oscillator_field,
            vector_field_from_sigma_tower,
        )
    except ImportError:
        print("  omnibias-dynamics not installed: skipping W6 dynamics-bridge probe")
        ledger.add("w6", "info", "omnibias-dynamics absent: dynamics-bridge probe skipped")
        return

    # (1) Soundness: the tower-built tanh field flows and *encloses* a fine RK4
    # reference across K=8 initial conditions (grid over the state line).
    field, jac = vector_field_from_sigma_tower("tanh")
    h, n = 0.025, 20
    max_escape = 0.0
    for x0 in (-1.5, -0.8, -0.3, 0.0, 0.25, 0.5, 0.9, 1.4):
        box = lohner_flow(field, jac, [Interval.point(x0)], h, n, order=6).to_box()[0]
        x, dt = x0, h * n / 8000
        for _ in range(8000):
            k1 = math.tanh(x)
            k2 = math.tanh(x + 0.5 * dt * k1)
            k3 = math.tanh(x + 0.5 * dt * k2)
            k4 = math.tanh(x + dt * k3)
            x += dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        max_escape = max(max_escape, box.lo - x, x - box.hi)
    print(f"  tanh-field flow encloses fine RK4 across 8 ICs (max escape {max_escape:.2e})")
    if max_escape > 0.0:
        ledger.add("w6", "bug", "tower-built flow escapes the RK4 reference", max_escape=max_escape)
    assert max_escape <= 0.0
    ledger.add(
        "w6",
        "gap",
        "validated dynamics had no jet-ingest bridge (FIXED)",
        detail="vector_field_from_sigma_tower composes the closed-form sigma tower onto the "
        "state's time-Taylor series (compose_jet), yielding the (VectorField, JacobianEnclosure) "
        "pair the Lohner flow consumes -- previously every field was hand-written",
        resolved=True,
    )

    # (2) Best-in-class: on the 2-D sigma-oscillator from an initial *box*, the
    # QR-Lohner frame beats the naive interval-Taylor integrator on wrapping.
    ofield, ojac = sigma_oscillator_field("tanh", stiffness=1.0)
    rad = 0.01
    y0 = [Interval(0.5 - rad, 0.5 + rad), Interval(-rad, rad)]
    hh, nn = 0.03, 60
    loh = lohner_flow(ofield, ojac, y0, hh, nn, order=8).to_box()
    ivp = integrate_ivp(ofield, y0, 0.0, hh * nn, order=8, n_steps=nn)
    cmp = baseline_compare("Lohner vs naive interval-Taylor flow (x width)", loh[0].width, ivp[0].width)
    improvement = ivp[0].width / loh[0].width
    print(f"  oscillator wrapping: Lohner x-width {loh[0].width:.2e} vs naive {ivp[0].width:.2e}  wins={cmp.wins} ({improvement:.1f}x tighter)")
    if not cmp.wins:
        ledger.add("w6", "flaw", "Lohner did not beat naive interval flow on wrapping", ratio=cmp.ratio)
    assert cmp.wins and improvement > 2.0 and loh[1].width < ivp[1].width

    # (3) Discrete bridge: prove a period-2 orbit of the logistic map r=3.2.
    r = 3.2
    one, two = Interval.point(1.0), Interval.point(2.0)
    rr = Interval.point(r)
    g = lambda x: rr * x * (one - x)  # noqa: E731
    dg = lambda x: rr * (one - two * x)  # noqa: E731
    x_star = (r + 1 + math.sqrt((r - 3) * (r + 1))) / (2 * r)
    orbit = discrete_periodic_point(g, dg, x_star, 2)
    assert orbit.exists and orbit.enclosure is not None
    lo, hi = orbit.enclosure
    assert lo <= x_star <= hi
    print(f"  logistic r=3.2 period-2 orbit proved: x* in [{lo:.9f}, {hi:.9f}] (kappa {orbit.kappa:.2e})")
    ledger.add("w6", "info", "tower->dynamics bridge flows tanh-field (encloses RK4), beats naive wrapping, proves discrete period-2")


def probe_zeta_special_values(ledger: FindingsLedger) -> None:
    """W7 (Tier 3): exact zeta / beta special values + honest critical-strip enclosure."""
    print("=== W7: rigorous zeta / L special values + attempted critical-strip enclosure ===")
    from omnibias.core.verified.dirichlet import (
        dirichlet_beta_odd,
        zeta_enclosure,
        zeta_euler_maclaurin,
        zeta_even,
        zeta_negative_odd,
    )

    # (1) Exact closed-form special values enclose mpmath (K=8 even orders).
    with mpmath.workdps(50):
        for m in range(1, 9):
            assert zeta_even(m).contains(float(mpmath.zeta(2 * m)))
            assert zeta_negative_odd(m).contains(float(mpmath.zeta(1 - 2 * m)))
        assert dirichlet_beta_odd(0).contains(float(mpmath.pi / 4))
        for m in range(1, 9):
            beta = float(
                mpmath.nsum(lambda k, m=m: (-1) ** k / (2 * k + 1) ** (2 * m + 1), [0, mpmath.inf])
            )
            assert dirichlet_beta_odd(m).contains(beta)
    print("  zeta(2m), zeta(1-2m)=-B_2m/2m, beta(2m+1) exact-enclose mpmath (K=8)")
    ledger.add("w7", "info", "exact zeta/beta special values (closed-form pi-multiples) enclose mpmath")

    # (2) The Euler-Maclaurin engine enters the critical strip where the p-series
    # majorant wall (the named baseline) *cannot* -- K=8 strip seeds vs mpmath.zeta.
    strip = [
        complex(0.5, 3.0), complex(0.5, 14.134725), complex(0.5, 21.02204),
        complex(0.25, 10.0), complex(0.75, 1.0), complex(0.1, 5.0),
        complex(0.9, -7.0), complex(0.5, 30.0),
    ]
    max_w = 0.0
    with mpmath.workdps(40):
        for s in strip:
            enc = zeta_euler_maclaurin(s, num_sum_terms=25, order=6)
            ref = complex(mpmath.zeta(mpmath.mpc(s.real, s.imag)))
            if not (enc.re.contains(ref.real) and enc.im.contains(ref.imag)):
                ledger.add("w7", "bug", "EM strip enclosure escapes mpmath.zeta", s=str(s))
            assert enc.re.contains(ref.real) and enc.im.contains(ref.imag)
            max_w = max(max_w, enc.re.width, enc.im.width)
    # Baseline: the Re(s)>1 majorant refuses the strip (raises) -- EM wins by existing.
    wall_refuses = False
    try:
        zeta_enclosure(complex(0.5, 14.134725))
    except ValueError:
        wall_refuses = True
    print(f"  EM continues into 0<Re(s)<1 across 8 seeds (max width {max_w:.1e}); p-series wall refuses={wall_refuses}")
    assert wall_refuses
    ledger.add(
        "w7",
        "gap",
        "the p-series majorant wall stops at Re(s)>1; Euler-Maclaurin is the continuation path (FIXED)",
        detail="zeta_euler_maclaurin (DLMF 25.2.3 + rigorous remainder) numerically encloses zeta in "
        "the critical strip; RH stays an external obligation, never inferred",
        resolved=True,
    )

    # (3) Honesty: near the first zero the enclosed magnitude is tiny, but that is
    # NOT an RH claim -- the enclosure resolves a small nonzero value, no inference.
    near = zeta_euler_maclaurin(complex(0.5, 14.134725), num_sum_terms=30, order=8)
    print(f"  near first zero: |zeta| enclosed <= {near.mag:.2e} (small, but no RH inference)")
    ledger.add("w7", "info", "critical-strip enclosure is numerical; RH remains an external obligation")


def probe_special_identities_lean(ledger: FindingsLedger) -> None:
    """W8 (Tier 3): rational special-number identities -> Lean-kernel equality obligation."""
    print("=== W8: Lean-checkable corpus of special-number identities ===")
    from math import comb

    from omnibias.core.proof.lean_check import generate_obligation
    from omnibias.core.verified.coeffs import bernoulli_number_exact, euler_number_exact
    from omnibias.difference import (
        bernoulli_recurrence_identity,
        bernoulli_sign_certificate,
        check_identity_certificate,
        euler_recurrence_identity,
        zeta_negative_odd_identity,
    )

    available = lean_check_available()
    print(f"  Lean toolchain available: {available} (graceful degradation otherwise)")

    zeta_neg = {1: Fraction(-1, 12), 2: Fraction(1, 120), 3: Fraction(-1, 252), 4: Fraction(1, 240)}

    # (1) The rational data is correct: the kernel's integer obligation is really 0,
    # and each equality obligation routes through the NEW `eq_of_mem_point` lemma.
    for n in range(2, 9):
        cert = bernoulli_recurrence_identity(n)
        lhs = sum(c * m for c, m in cert["payload"]["lhs_terms"])
        assert lhs == 0 and sum(comb(n, k) * bernoulli_number_exact(k) for k in range(n)) == 0
        assert "eq_of_mem_point" in (generate_obligation(cert) or "")
    for m in range(1, 5):
        e_cert = euler_recurrence_identity(m)
        assert sum(c * v for c, v in e_cert["payload"]["lhs_terms"]) == 0
        assert sum(comb(2 * m, 2 * k) * euler_number_exact(2 * k) for k in range(m + 1)) == 0
        z_cert = zeta_negative_odd_identity(m, zeta_neg[m])
        assert sum(c * v for c, v in z_cert["payload"]["lhs_terms"]) == 0
    print("  Bernoulli/Euler recurrences + zeta(1-2m) values reduce to exact Int equalities (K=8)")

    # (2) EM remainder-sign driver: sign(B_2m) = (-1)^{m+1} alternates +,-,+,-.
    for m in range(1, 5):
        obligation = generate_obligation(bernoulli_sign_certificate(m)) or ""
        want = "enclosed_quantity_pos" if (-1) ** (m + 1) > 0 else "enclosed_quantity_neg"
        assert want in obligation
    print("  sign(B_2m) alternation (EM remainder-sign driver) emits pos/neg obligations")

    # (3) Honesty: the flag is exactly the kernel verdict; earned only on a real pass.
    verdict = check_identity_certificate(bernoulli_recurrence_identity(6))
    assert verdict.theorem_prover_verified == verdict.lean.verified
    assert verdict.obligation_generated and verdict.sealed_ok
    if available:
        assert verdict.theorem_prover_verified
        print("  lake present: Bernoulli recurrence earned theorem_prover_verified (genuine kernel pass)")
    else:
        assert not verdict.theorem_prover_verified
        ledger.add("w8", "info", "Lean toolchain absent: identity bridge degraded gracefully (flag stays False)")

    ledger.add(
        "w8",
        "gap",
        "the kernel had no rational-EQUALITY lemma; only sign/gap/LDLT obligations (FIXED)",
        detail="added ZInterval.eq_of_mem_point + Certificate.enclosed_quantity_eq (a value in the "
        "point interval [0,0] equals 0), so a special-number identity earns a kernel-checked equality; "
        "verified locally: true identities pass lake build, a false one is rejected",
        resolved=True,
    )
    ledger.add("w8", "info", "Bernoulli/Euler recurrence + zeta-value identities are Lean-kernel-checkable equalities")


def probe_singularity_pade_sheffer(ledger: FindingsLedger) -> None:
    """W7-ext (Tier 3): transfer theorem + Pade + Sheffer/Riordan (certified/exact)."""
    print("=== W7-ext: singularity analysis / Pade / Sheffer-Riordan ===")
    from math import comb, exp

    from omnibias.difference import (
        connection_constants,
        dominant_pole_coefficient_asymptotic,
        pade_approximant,
        pade_certified_remainder,
        pade_evaluate,
        rational_ogf_coefficients,
        riordan_array,
        riordan_inverse,
        riordan_product,
        series_reciprocal,
        sheffer_classify,
        singular_template_coefficient,
        transfer_theorem,
    )
    from omnibias.difference._core.stirling import stirling_second

    # (1) Transfer theorem: the exact singular-template coefficient PROVABLY encloses
    # the value, so the classical float asymptotic ships with an error bar (K=8).
    cases = [
        (1, Fraction(1, 2), 1), (-2, Fraction(1, 4), Fraction(-1, 2)),
        (3, Fraction(1, 3), Fraction(3, 2)), (1, Fraction(2, 3), 2),
        (1, Fraction(1, 5), Fraction(5, 2)), (-1, Fraction(1, 2), Fraction(-3, 2)),
        (2, Fraction(1, 4), 3), (1, Fraction(3, 4), Fraction(1, 2)),
    ]
    for scale, radius, alpha in cases:
        for n in (5, 12):
            est = transfer_theorem(scale, radius, alpha, n)
            tmpl = singular_template_coefficient(alpha, n)
            true = float(scale) * float(radius) ** (-n) * float(tmpl)
            assert est.exact_coefficient.contains(true)
    print("  transfer-theorem exact coefficient encloses the value across 8 (scale,rho,alpha) x 2 n")
    ledger.add("w7ext", "info", "transfer theorem: exact singular coefficient encloses the value (error bar on the asymptotic)")

    # (2) Best-in-class: Pade beats raw truncation at an EQUAL coefficient budget (K=8).
    def pade_vs_truncation(seed: int) -> float:
        c = 0.5 + 0.5 * (seed % 8)  # exp(c*x) family
        coeffs = [Fraction(c) ** k / math.factorial(k) for k in range(7)]
        p, q = pade_approximant(coeffs, 3, 3)
        x = 0.4
        pade_err = abs(exp(c * x) - float(pade_evaluate(p, q, Fraction(2, 5))))
        trunc_err = abs(exp(c * x) - sum(float(coeffs[k]) * x**k for k in range(7)))
        return trunc_err / max(pade_err, 1e-300)

    sweep = seed_sweep(pade_vs_truncation, range(8))
    print(f"  Pade[3/3] vs raw 7-term truncation: err ratio min {sweep['min']:.1f}x, mean {sweep['mean']:.1f}x")
    if sweep["min"] <= 1.0:
        ledger.add("w7ext", "flaw", "Pade did not beat raw truncation on some seed", ratio=sweep["min"])
    assert sweep["min"] > 1.0
    ledger.add("w7ext", "info", f"Pade beats raw truncation at equal budget across 8 seeds (min {sweep['min']:.1f}x)")

    # (3) Certified Pade remainder CONTAINS the true error over a disc (soundness).
    coeffs = [Fraction(1, math.factorial(k)) for k in range(13)]
    p, q = pade_approximant(coeffs, 3, 3)
    ivs = [Interval.from_rational(Fraction(1, math.factorial(k))) for k in range(13)]
    rem = pade_certified_remainder(p, q, ivs, 0.3, tail_bound=1.0, tail_ratio=0.5)
    true_err = abs(exp(0.3) - float(pade_evaluate(p, q, Fraction(3, 10))))
    print(f"  certified Pade remainder {rem.hi:.2e} contains true err {true_err:.2e}: {true_err <= rem.hi}")
    assert true_err <= rem.hi
    ledger.add("w7ext", "info", "certified Pade remainder rigorously contains the true approximation error")

    # (4) Dominant-pole asymptotic converges to the exact Fibonacci coefficient
    # (numerical growth law; exact coefficients come from rational_ogf_coefficients).
    errs = []
    for n in (10, 20, 30):
        exact = int(rational_ogf_coefficients([0, 1], [1, -1, -1], n + 1)[n])
        approx = dominant_pole_coefficient_asymptotic([0, 1], [1, -1, -1], n)
        errs.append(abs(approx - exact) / exact)
    assert errs[0] > errs[1] > errs[2] and errs[2] < 1e-9
    print(f"  Fibonacci dominant-pole asymptotic converges: rel err {errs[0]:.1e} -> {errs[2]:.1e}")

    # (5) Sheffer classification + Riordan group law + connection constants (exact).
    assert sheffer_classify([1, 2, 3], [0, 1]).kind == "appell"
    assert sheffer_classify([1], [0, 1, 1]).kind == "associated"
    d = [Fraction(1)] * 6
    h = [Fraction(0), *([Fraction(1)] * 5)]  # (1/(1-t), t/(1-t)) -> Pascal
    pascal = riordan_array(series_reciprocal([1, -1], 6), (Fraction(0), *series_reciprocal([1, -1], 5)), 6)
    assert all(pascal[n][k] == comb(n, k) for n in range(6) for k in range(n + 1))
    d_inv, h_inv = riordan_inverse(d, h, 5)
    prod_d, prod_h = riordan_product((d, h), (d_inv, h_inv), 5)
    assert prod_d[0] == 1 and all(c == 0 for c in prod_d[1:])
    # connection constants: x^n -> falling factorials recovers Stirling second kind.
    def _falling(k: int) -> list[Fraction]:
        coeffs_ = [Fraction(1)]
        for j in range(k):
            nxt = [Fraction(0)] * (len(coeffs_) + 1)
            for i, ci in enumerate(coeffs_):
                nxt[i] += -Fraction(j) * ci
                nxt[i + 1] += ci
            coeffs_ = nxt
        return coeffs_

    cc = connection_constants([[Fraction(0)] * n + [Fraction(1)] for n in range(6)], [_falling(k) for k in range(6)])
    assert all(cc[n][k] == stirling_second(n, k) for n in range(6) for k in range(n + 1))
    print("  Sheffer classify + Riordan group (Pascal, inverse->identity) + Stirling connection: exact")
    ledger.add("w7ext", "info", "Sheffer/Riordan group ops + connection constants recover textbook results exactly")


def probe_umbral_sheffer(ledger: FindingsLedger) -> None:
    """Umbral / Sheffer generation + operators (exact) vs a float polynomial baseline."""
    print("=== umbral / Sheffer generation + operator layer (exact rational) ===")
    import random as _random
    from math import factorial

    import numpy as np
    from omnibias.difference import (
        associated_sequence,
        bernoulli_polynomial,
        delta_operator_apply,
        falling_factorial_coeffs,
        pincherle_derivative,
        sheffer_sequence,
        stirling_second_row,
    )

    n_max = 7
    exp_minus_one = [Fraction(0)] + [Fraction(1, factorial(k)) for k in range(1, n_max + 1)]
    log_one_plus = [Fraction(0)] + [Fraction((-1) ** (k + 1), k) for k in range(1, n_max + 1)]
    bernoulli_g = [Fraction(1, factorial(k + 1)) for k in range(n_max + 1)]  # (e^t-1)/t

    # (1) Named-sequence generation is exact: associated(e^t-1) = falling factorial,
    # associated(log(1+t)) = Bell/Touchard (Stirling-2 rows), sheffer(Bernoulli) = B_n(x).
    falling = associated_sequence(exp_minus_one, n_max)
    touchard = associated_sequence(log_one_plus, n_max)
    bernoulli = sheffer_sequence(bernoulli_g, (0, 1), n_max)
    for n in range(n_max + 1):
        assert tuple(falling[n]) == tuple(Fraction(c) for c in falling_factorial_coeffs(n))
        assert tuple(touchard[n]) == tuple(Fraction(s) for s in stirling_second_row(n))
        assert tuple(bernoulli[n]) == bernoulli_polynomial(n)
    print("  associated(e^t-1)=(x)_n, associated(log(1+t))=Stirling2 rows, sheffer(Bernoulli)=B_n(x)")

    # (2) The Pincherle operator identity [f(D), X] = f'(D) holds to zero residual (K=8).
    def _strip(coeffs: tuple[Fraction, ...]) -> tuple[Fraction, ...]:
        out = list(coeffs)
        while len(out) > 1 and out[-1] == 0:
            out.pop()
        return tuple(out)

    def pincherle_residual(seed: int) -> float:
        rng = _random.Random(seed)
        f = [Fraction(0)] + [Fraction(rng.randint(-3, 3), rng.randint(1, 3)) for _ in range(5)]
        p = tuple(Fraction(rng.randint(-4, 4)) for _ in range(5))
        qxp = delta_operator_apply(f, (Fraction(0), *p))
        xqp = (Fraction(0), *delta_operator_apply(f, p))
        width = max(len(qxp), len(xqp))
        left = list(qxp) + [Fraction(0)] * (width - len(qxp))
        right = list(xqp) + [Fraction(0)] * (width - len(xqp))
        commutator = _strip(tuple(a - b for a, b in zip(left, right, strict=True)))
        rhs = delta_operator_apply(pincherle_derivative(f), p)
        w2 = max(len(commutator), len(rhs))
        cc = list(commutator) + [Fraction(0)] * (w2 - len(commutator))
        rr = list(rhs) + [Fraction(0)] * (w2 - len(rhs))
        return float(max((abs(a - b) for a, b in zip(cc, rr, strict=True)), default=Fraction(0)))

    sweep = seed_sweep(pincherle_residual, range(8))
    print(f"  Pincherle [f(D),X]=f'(D) residual over K=8: max {sweep['max']:.0f} (0 == exact)")
    if sweep["max"] != 0.0:
        ledger.add("umbral", "bug", "Pincherle operator identity residual non-zero", max_resid=sweep["max"])
    assert sweep["max"] == 0.0

    # (3) Best-in-class: exact rational Sheffer coefficients vs a float polynomial fit.
    n = 6
    exact = bernoulli[n]  # B_6(x): exact rationals (1/42, ...)
    xs = np.arange(n + 1, dtype=float)
    ys = np.array([float(sum((c * Fraction(int(x)) ** j for j, c in enumerate(exact)), Fraction(0))) for x in xs])
    float_coeffs = np.polyfit(xs, ys, n)[::-1]
    max_read_err = max(abs(float(c) - fc) for c, fc in zip(exact, float_coeffs, strict=True))
    cmp = baseline_compare("exact-rational vs float polyfit Sheffer coeffs (B_6)", 0.0, max_read_err)
    print(f"  B_6 coeffs {tuple(str(c) for c in exact)}; float polyfit read err {max_read_err:.2e} wins={cmp.wins}")
    if max_read_err > 0.0:
        ledger.add(
            "umbral",
            "flaw",
            "float polynomial fit cannot read exact rational Sheffer/Bernoulli coefficients",
            detail="sheffer_sequence yields exact 1/42, ...; np.polyfit on samples drifts",
            max_coeff_err=max_read_err,
        )
    assert cmp.wins
    ledger.add(
        "umbral",
        "info",
        "Sheffer/associated generation + umbral operators exact; Pincherle residual 0; exact beats float polyfit",
    )


# Registered probes; each new workstream appends its probe here.
PROBES = (
    probe_constants,
    probe_tower_soundness,
    probe_fd_beats_naive,
    probe_special_numbers,
    probe_euler_maclaurin,
    probe_certified_diff_lean,
    probe_discrete_discovery,
    probe_analytic_combinatorics,
    probe_truncation_error,
    probe_dynamics_bridge,
    probe_zeta_special_values,
    probe_special_identities_lean,
    probe_singularity_pade_sheffer,
    probe_umbral_sheffer,
)


def main() -> None:
    ledger = FindingsLedger("difference_validate")
    for probe in PROBES:
        probe(ledger)
    print("\n" + ledger.summary())
    try:
        path = ledger.write()
        print(f"\nfindings ledger -> {path}")
    except OSError:  # pragma: no cover - scratch may be unavailable in a sandbox
        print("\n(findings ledger not persisted: scratch dir unavailable)")
    # The gate fails only on a *live* soundness bug. Bugs the program found and
    # fixed are recorded with repro["resolved"]=True (each backed by a regression
    # test) so they stay in the curated ledger without tripping the gate.
    unresolved = [f for f in ledger if f.severity == "bug" and not f.repro.get("resolved")]
    assert not unresolved, f"{len(unresolved)} live soundness bug(s) surfaced -- see the ledger"
    resolved = sum(1 for f in ledger if f.severity == "bug" and f.repro.get("resolved"))
    print(f"OK: all difference refinement probes pass their gates ({resolved} fixed bug(s) on record).")


if __name__ == "__main__":
    main()
