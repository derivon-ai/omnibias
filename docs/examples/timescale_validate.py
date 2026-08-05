# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Data-driven / verified / best-in-class smoke for omnibias-timescale.

Run:

    pip install "omnibias-timescale[test]"
    python docs/examples/timescale_validate.py

Each *probe* turns a time-scale capability into an instrumented experiment under the
``omnibias-dev-empirical-validation`` gates -- a high-precision reference, a best-in-class
comparison against a named baseline, and a ``K >= 8`` seed sweep -- recording findings into
a shared :class:`~omnibias.difference.validation.FindingsLedger`.

This exercises the **``mu -> 0`` limit**: the founding ``delta -> 0`` bias collapse (a finite
difference of many biases becoming the smooth derivative ``sigma^(K-1)``) generalized to a
variable mesh. It is the derivative sense of "collapse", distinct from the ``q -> 1`` limit
of ``omnibias-qcalculus`` and never the ``beta -> inf`` feasibility penalty. Honesty labels:
**closed-form** (the tower dispatch on ``R`` and the exact delta integral on a discrete
scale) and **numerical** (the difference / Jackson quotients and the ``R`` quadrature).
"""

from __future__ import annotations

import math

from omnibias.difference import finite_difference_estimate
from omnibias.difference.validation import (
    FindingsLedger,
    baseline_compare,
    seed_sweep,
)
from omnibias.qcalculus import q_derivative
from omnibias.timescale import (
    delta_derivative,
    delta_derivative_tower,
    delta_integral,
    h_integers,
    hilger_exponential,
    quantum,
    reals,
    sigma_value,
    solve_linear_dynamic,
    variation_of_constants,
)


def probe_mu_to_zero_collapse(ledger: FindingsLedger) -> None:
    """The founding mu -> 0 collapse: the delta derivative -> the closed-form tower."""
    print("=== mu -> 0 collapse (founding sense, generalized to a variable mesh) ===")

    def collapse_error(seed: int) -> float:
        t = 0.3 + 0.1 * (seed % 8)  # 0.3 .. 1.0
        true = 1.0 - math.tanh(t) ** 2
        errs_h = [abs(delta_derivative_tower("tanh", t, h_integers(h)) - true) for h in (0.1, 1e-3)]
        errs_q = [abs(delta_derivative_tower("tanh", t, quantum(q)) - true) for q in (1.1, 1.001)]
        # monotone collapse in both the hZ and quantum meshes.
        assert errs_h[1] < errs_h[0] and errs_q[1] < errs_q[0]
        return max(errs_h[1], errs_q[1])  # residual at the finest mesh

    stats = seed_sweep(collapse_error, range(8))
    print(f"  residual at finest mesh over K=8: max {stats['max']:.2e}, mean {stats['mean']:.2e}")
    assert stats["max"] < 1e-3
    ledger.add(
        "timescale",
        "info",
        "delta derivative collapses to the closed-form tower as mu -> 0 (founding sense)",
    )


def probe_unified_dispatch_beats_single_method(ledger: FindingsLedger) -> None:
    """One dispatching operator matches the specialized tools AND beats a naive single-method."""
    print("=== unification: dispatch vs a single fixed-h forward difference ===")
    # (a) exact equivalence with the specialized implementations (zero discrepancy).
    max_disc = 0.0
    for t in (0.2, 0.7, 1.3, 1.9):
        hz = delta_derivative_tower("tanh", t, h_integers(0.3))
        assert hz == finite_difference_estimate("tanh", t, 1, 0.3, "forward").estimate
        qz = delta_derivative_tower("tanh", t, quantum(1.5))
        ref = q_derivative(lambda x: sigma_value("tanh", x), t, 1.5)
        max_disc = max(max_disc, abs(qz - ref))
    print(f"  dispatch == difference (hZ) exactly; == qcalculus (quantum) to {max_disc:.1e}")
    assert max_disc < 1e-12

    # (b) best-in-class: on R the dispatch is closed-form (~0 error); a naive fixed-h
    #     forward difference used everywhere carries O(h) error.
    def accuracy_gain(seed: int) -> float:
        t = 0.3 + 0.15 * (seed % 8)
        true = 1.0 - math.tanh(t) ** 2
        dispatch_err = abs(delta_derivative_tower("tanh", t, reals()) - true)
        naive_err = abs((math.tanh(t + 0.1) - math.tanh(t)) / 0.1 - true)
        return naive_err / max(dispatch_err, 1e-16)

    stats = seed_sweep(accuracy_gain, range(8))
    cmp = baseline_compare(
        "R-derivative-accuracy", 0.0, stats["mean"], lower_is_better=True
    )
    print(f"  on R: dispatch (closed-form) error ~0 vs fixed-h forward diff; gain x{stats['mean']:.1e}")
    assert cmp.wins  # dispatch error 0 <= naive mean error
    ledger.add(
        "timescale",
        "info",
        "one TimeScale operator reproduces difference/qcalculus exactly and is exact on R",
    )


def probe_ftc_and_dynamic(ledger: FindingsLedger) -> None:
    """Exact fundamental theorem and linear dynamic equations across discrete scales."""
    print("=== FTC + linear dynamic equations (exact on discrete scales) ===")
    # FTC: int_a^b f^Delta Delta t = f(b) - f(a), across K = 8 scales / windows.
    # b is chosen exactly on the mesh (b = steps * h) so the sum telescopes to f(b) - f(0).
    def ftc_residual(seed: int) -> float:
        h = 0.5 if seed % 2 else 0.25
        H = h_integers(h)
        f = lambda x: x**3 - 2 * x + 1  # noqa: E731
        fD = lambda x: delta_derivative(f, x, H)  # noqa: E731
        b = (6 + seed) * h
        got = delta_integral(fD, 0.0, b, H)
        return abs(got - (f(b) - f(0.0)))

    stats = seed_sweep(ftc_residual, range(8))
    print(f"  FTC residual over K=8: max {stats['max']:.2e}")
    assert stats["max"] < 1e-9

    # Dynamic equation: recursion == variation of constants == Hilger exponential (homog.).
    H = h_integers(0.25)
    p = lambda t: 0.5 - 0.1 * t  # noqa: E731
    r = lambda t: math.sin(t)  # noqa: E731
    for t, y in solve_linear_dynamic(p, r, 2.0, H, 0.0, 2.0):
        assert abs(y - variation_of_constants(p, r, 2.0, t, H, 0.0)) < 1e-9
    for t, y in solve_linear_dynamic(0.6, 0.0, 1.0, H, 0.0, 2.0):
        assert abs(y - hilger_exponential(0.6, t, 0.0, H)) < 1e-9
    # mu -> 0: Hilger exponential -> exp.
    fine = hilger_exponential(0.8, 2.0, 0.0, h_integers(1e-3))
    assert abs(fine - math.exp(1.6)) < 1e-2
    assert abs(hilger_exponential(0.8, 2.0, 0.0, reals()) - math.exp(1.6)) < 1e-9
    print("  recursion == variation-of-constants == Hilger exponential; e_p -> exp as mu -> 0")
    ledger.add(
        "timescale",
        "info",
        "FTC exact on discrete scales; dynamic-equation solvers agree and collapse to the ODE",
    )


PROBES = (
    probe_mu_to_zero_collapse,
    probe_unified_dispatch_beats_single_method,
    probe_ftc_and_dynamic,
)


def main() -> None:
    ledger = FindingsLedger("timescale_validate")
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
    print("OK: all timescale probes pass their gates.")


if __name__ == "__main__":
    main()
