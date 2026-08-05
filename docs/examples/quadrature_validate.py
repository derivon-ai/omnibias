# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Data-driven refinement smoke for W3-ext certified quadrature.

Run:

    pip install "omnibias-difference[test]" omnibias-verify
    python docs/examples/quadrature_validate.py

Each probe turns a certified quadrature rule into an instrumented experiment
under the ``omnibias-dev-empirical-validation`` gates -- grid-and-random
enclosure soundness (here a *sweep of K>=8 integrands*, each certified enclosure
must contain the exact integral), an ``mpmath.quad`` high-precision oracle, and a
best-in-class comparison against the named fixed-node trapezoid baseline -- and
records the gaps / flaws it exposes into a shared ``FindingsLedger``.

Honesty labels: the derivative *enclosures* feeding the remainders are
**closed-form** (sigma towers + exact coefficients); the Gauss nodes/weights are
certified data; the returned integral is the guaranteed sandwich (**numerical**
remainder). ``tanh_sinh_estimate`` is a labelled *numerical* estimate, never a
certified enclosure.
"""

from __future__ import annotations

import math

try:
    import mpmath
except ImportError:  # pragma: no cover - the [test] extra ships mpmath
    print("this smoke needs mpmath: pip install 'omnibias-difference[test]'")
    raise SystemExit(0) from None

from omnibias.core.verified.interval import Interval
from omnibias.core.verified.quadrature import (
    clenshaw_curtis_integral,
    gauss_legendre_integral,
    tanh_sinh_estimate,
    trapezoid_integral,
)
from omnibias.core.verified.sigma import sigma_tower_interval
from omnibias.core.verified.transcend import exp_iv
from omnibias.difference.validation import (
    FindingsLedger,
    seed_sweep,
)

# K = 9 integrands: (name, f_iv, deriv_bound(k, box), a, b, mpmath f).
_EXP = ("exp", exp_iv, lambda k, box: exp_iv(box), 0.0, 1.0, mpmath.exp)


def _sigma_case(name: str, a: float, b: float, mp_fn):  # type: ignore[no-untyped-def]
    def f_iv(x: Interval) -> Interval:
        return sigma_tower_interval(name, x, 0)[0]

    def deriv(k: int, box: Interval) -> Interval:
        return sigma_tower_interval(name, box, k)[k]

    return (name, f_iv, deriv, a, b, mp_fn)


_CASES = [
    _EXP,
    _sigma_case("tanh", 0.0, 1.0, mpmath.tanh),
    _sigma_case("sigmoid", -1.0, 1.0, lambda z: 1 / (1 + mpmath.e ** (-z))),
    _sigma_case("gaussian", -1.0, 1.0, lambda z: mpmath.e ** (-z * z / 2)),
    _sigma_case("sin", 0.0, math.pi, mpmath.sin),
    _sigma_case("cos", 0.0, 1.0, mpmath.cos),
    _sigma_case("sech", -1.0, 1.0, lambda z: mpmath.sech(z)),
    _sigma_case("softplus", 0.0, 1.0, lambda z: mpmath.log(1 + mpmath.e**z)),
    _sigma_case("silu", -1.0, 1.0, lambda z: z / (1 + mpmath.e ** (-z))),
]


def probe_gauss_soundness(ledger: FindingsLedger) -> None:
    """Certified Gauss-Legendre encloses the mpmath integral across K=9 integrands."""
    print("=== W3-ext: certified Gauss-Legendre encloses mpmath.quad (K=9) ===")
    n = 6
    max_w = 0.0
    with mpmath.workdps(40):
        for name, f_iv, deriv, a, b, mp_fn in _CASES:
            ref = float(mpmath.quad(mp_fn, [a, b]))
            box = Interval(a, b)
            enc = gauss_legendre_integral(f_iv, a, b, n=n, deriv_2n_bound=deriv(2 * n, box))
            ok = enc.contains(ref)
            max_w = max(max_w, enc.width)
            print(f"  {name:9s} int[{a},{b}] = {ref:+.9f}  enc width {enc.width:.2e}  {'OK' if ok else 'ESCAPES'}")
            if not ok:
                ledger.add("quadrature", "bug", f"Gauss enclosure escapes mpmath for {name}", escape=name)
            assert ok
    print(f"  all 9 certified Gauss enclosures contain the mpmath oracle (max width {max_w:.1e})")
    ledger.add("quadrature", "info", "certified Gauss-Legendre encloses mpmath.quad across K=9 integrands")


def probe_beats_trapezoid(ledger: FindingsLedger) -> None:
    """Best-in-class: Gauss-5 beats the fixed 8-panel trapezoid at an equal node budget."""
    print("=== W3-ext: Gauss beats the fixed-node trapezoid baseline (K=8 seeds) ===")

    def improvement(seed: int) -> float:
        # a family of smooth integrands exp(c x) on [0,1], c depends on the seed.
        c = 0.5 + 0.5 * (seed % 8)
        c_iv = Interval.point(c)

        def f_iv(x: Interval) -> Interval:
            return exp_iv(c_iv * x)

        d2n = exp_iv(c_iv * Interval(0.0, 1.0)) * c_iv.pow_int(10)
        gauss = gauss_legendre_integral(f_iv, 0.0, 1.0, n=5, deriv_2n_bound=d2n)
        nodes = [exp_iv(c_iv * Interval.point(k / 8.0)) for k in range(9)]
        d2 = exp_iv(c_iv * Interval(0.0, 1.0)) * c_iv.pow_int(2)
        trap = trapezoid_integral(nodes, 0.0, 1.0, d2)
        return trap.width / max(gauss.width, 1e-300)

    sweep = seed_sweep(improvement, range(8))
    print(f"  Gauss-5 vs trapezoid-8 width ratio: min {sweep['min']:.1e}x, mean {sweep['mean']:.1e}x")
    if sweep["min"] <= 1.0:
        ledger.add("quadrature", "flaw", "Gauss did not beat trapezoid on some seed", ratio=sweep["min"])
    assert sweep["min"] > 1e3  # many orders tighter on every seed
    ledger.add("quadrature", "info", f"Gauss beats fixed trapezoid by >=1e3x across 8 seeds (min {sweep['min']:.1e})")


def probe_endpoint_singularity(ledger: FindingsLedger) -> None:
    """Honesty: derivative-bound rules fail at an endpoint singularity; tanh-sinh handles it."""
    print("=== W3-ext: endpoint singularity -> tanh-sinh numerical estimator ===")
    # int_0^1 1/sqrt(x) dx = 2, but f' blows up at 0, so no finite derivative bound exists.
    est = tanh_sinh_estimate(lambda x: 1.0 / math.sqrt(x) if x > 0 else 0.0, 0.0, 1.0, level=9)
    print(f"  tanh-sinh int 1/sqrt(x) = {est.value:.9f} (exact 2), label={est.label!r}")
    assert abs(est.value - 2.0) < 1e-6 and est.label == "numerical"
    ledger.add(
        "quadrature",
        "gap",
        "derivative-bound rules cannot certify endpoint-singular integrands",
        detail="tanh_sinh_estimate is a labelled numerical estimator (a rigorous DE bound needs "
        "strip analyticity, not a real-derivative bound); use a certified rule when a bound is required",
    )


def probe_clenshaw_curtis_honesty(ledger: FindingsLedger) -> None:
    """Honesty: Clenshaw-Curtis is sound but its nodal remainder is deliberately loose."""
    print("=== W3-ext: Clenshaw-Curtis sound-but-loose nodal remainder ===")
    with mpmath.workdps(40):
        ref = float(mpmath.quad(mpmath.exp, [0, 1]))
    cc = clenshaw_curtis_integral(exp_iv, 0.0, 1.0, n=8, deriv_np1_bound=exp_iv(Interval(0.0, 1.0)))
    gauss = gauss_legendre_integral(exp_iv, 0.0, 1.0, n=6, deriv_2n_bound=exp_iv(Interval(0.0, 1.0)))
    print(f"  CC width {cc.width:.2e} (sound) vs Gauss width {gauss.width:.2e}; both contain truth")
    assert cc.contains(ref) and gauss.contains(ref)
    ledger.add(
        "quadrature",
        "flaw",
        "Clenshaw-Curtis remainder uses a loose max|omega|<=(b-a)^{n+1} nodal bound",
        detail="sound but conservative; a tighter Chebyshev-coefficient-decay bound would sharpen it. "
        "A weaker bound only widens the certified gap -- never silently tightened",
    )


PROBES = (
    probe_gauss_soundness,
    probe_beats_trapezoid,
    probe_endpoint_singularity,
    probe_clenshaw_curtis_honesty,
)


def main() -> None:
    ledger = FindingsLedger("quadrature_validate")
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
    print("OK: all certified-quadrature probes pass their gates.")


if __name__ == "__main__":
    main()
