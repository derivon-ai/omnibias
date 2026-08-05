# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Certified finite-difference -> derivative extraction + asymptotic reading -- omnibias-difference.

Run:

    pip install "omnibias-difference[test]"
    python docs/examples/certified_difference_towers.py

This is the founding **bias collapse**: ``K`` biases on a difference stencil
(spread ``delta``) coalesce as ``delta -> 0`` and the finite difference *becomes*
the derivative ``sigma^(K-1)``. The closed-form tower evaluates that limit
*exactly*, dodging the ``1/delta^(K-1)`` catastrophic cancellation. This is a
``delta -> 0`` derivative -- **not** the ``beta -> inf`` feasibility penalty of
``omnibias-convex`` / ``-control`` / ``-routing`` (do not conflate the two; see
``docs/theory.md``). The deterministic demo exercises both halves:

1. **Certified FD -> derivative.** For ``tanh`` the closed-form interval tower
   encloses the true ``sigma^(n)`` (mpmath), the numerical finite difference is
   *certified* to lie within a Taylor-remainder bound that shrinks as
   ``delta -> 0``, and a tiny ``delta`` shows the naive stencil losing to
   cancellation while the closed form stays exact.
2. **Asymptotic-coefficient reading.** Stirling numbers (off the Bell tower),
   Bernoulli numbers (off the ``tanh`` tower) and Euler numbers (off the ``sech``
   tower) are read out exactly and matched to an mpmath high-precision reference,
   with the leading asymptotics converging to it.

Honesty labels: **closed-form** (the towers + exact integer/rational
coefficients) and **numerical** (the finite-difference estimate and the mpmath
reference). No autodiff-exact path is used.
"""

from __future__ import annotations

import math
from fractions import Fraction

try:
    import mpmath
except ImportError:  # pragma: no cover - the [test] extra ships mpmath
    print("this demo needs mpmath: pip install 'omnibias-difference[test]'")
    raise SystemExit(0) from None

from omnibias.core.verified.interval import Interval
from omnibias.difference import (
    bell_number,
    bell_number_asymptotic,
    bernoulli_asymptotic,
    bernoulli_number,
    certified_derivative_enclosure,
    certified_fd_error,
    euler_asymptotic,
    euler_number,
    finite_difference_estimate,
    stirling_second_row,
)


def _true_derivative(z0: float, order: int) -> float:
    with mpmath.workdps(50):
        taylor = mpmath.taylor(lambda z: mpmath.tanh(z), z0, order)
        return float(taylor[order] * math.factorial(order))


def certified_extraction_demo() -> None:
    print("=== 1. certified finite-difference -> derivative (delta -> 0 collapse) ===")
    name, z, order = "tanh", 0.6, 3
    true = _true_derivative(z, order)

    enc = certified_derivative_enclosure(name, z, order)  # closed-form
    print(f"  closed-form enclosure of tanh^({order})({z}):")
    print(f"    [{enc.value.lo:.15f}, {enc.value.hi:.15f}]  (width {enc.value.width:.2e})")
    print(f"    true value (mpmath)       {true:.15f}")
    assert enc.value.lo <= true <= enc.value.hi
    assert enc.value.width <= 1e-8

    print("  numerical finite difference, certified to collapse into the enclosure:")
    for delta in (1e-1, 1e-2, 1e-3):
        cert = certified_fd_error(name, z, order, delta, "central")
        print(
            f"    delta={delta:<6} estimate={cert.estimate:.10f}  "
            f"|error| <= {cert.error_bound:.2e}  certified={cert.certified}"
        )
        assert cert.certified
        assert abs(cert.estimate - true) <= cert.error_bound + 1e-12

    naive = finite_difference_estimate(name, z, order, 1e-5, "central").estimate
    print("  the naive 1/delta^m stencil at a tiny delta loses to cancellation:")
    print(f"    naive FD (delta=1e-5)      {naive:.6f}   (true {true:.6f})")
    print(f"    closed-form error          0  (exact enclosure, width {enc.value.width:.1e})")
    assert abs(naive - true) > 1e3 * enc.value.width
    print()


def asymptotic_reading_demo() -> None:
    print("=== 2. asymptotic-coefficient reading off the closed-form towers ===")
    print(f"  Stirling second-kind row S(5,.) = {stirling_second_row(5)}  (partitions into k blocks)")
    print(f"  Bell(0..6)                       = {[bell_number(n) for n in range(7)]}")

    print("  Bernoulli (off the tanh tower)   vs mpmath:")
    for n in (2, 6, 12):
        got = bernoulli_number(n)
        with mpmath.workdps(50):
            ref = mpmath.bernoulli(n)
            got_mpf = mpmath.mpf(got.numerator) / mpmath.mpf(got.denominator)
            assert abs(got_mpf - ref) < mpmath.mpf(10) ** (-40)
        print(f"    B_{n:<2} = {str(got):>10}  (matches mpmath)")

    print("  Euler / secant numbers (off the sech tower):")
    euler_row = [euler_number(n) for n in range(9)]
    assert euler_row == [1, 0, -1, 0, 5, 0, -61, 0, 1385]
    print(f"    E_0..8 = {euler_row}")

    print("  leading asymptotics converge to the exact values:")
    for n in (10, 20):
        rel_b = abs(float(bernoulli_number(n)) / bernoulli_asymptotic(n) - 1.0)
        rel_e = abs(float(euler_number(n)) / euler_asymptotic(n) - 1.0)
        print(f"    n={n:<3} Bernoulli rel.err {rel_b:.2e}   Euler rel.err {rel_e:.2e}")
        assert rel_b < 2e-2 and rel_e < 2e-2
    rel_bell = abs(bell_number(20) / bell_number_asymptotic(20) - 1.0)
    print(f"    Bell(20) saddle-point rel.err {rel_bell:.2e}")
    assert rel_bell < 5e-2

    # the certified enclosure also pins the exact special number
    enc = certified_derivative_enclosure("sech", 0.0, 6).value  # encloses E_6 = -61
    assert round((enc.lo + enc.hi) / 2) == euler_number(6) == -61
    assert enc.lo <= float(Interval.from_rational(Fraction(-61)).lo) <= enc.hi
    print("  certified sech-tower enclosure of E_6 pins the exact -61.\n")


def main() -> None:
    certified_extraction_demo()
    asymptotic_reading_demo()
    print("OK: certified FD->derivative collapse holds; special numbers exact & asymptotics converge.")


if __name__ == "__main__":
    main()
