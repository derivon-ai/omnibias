# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Data-driven refinement smoke for W6-ext certified FD gradient-checking.

Run:

    pip install "omnibias-difference[test]" omnibias-verify
    python docs/examples/gradcheck_validate.py

The everyday ``torch.autograd.gradcheck`` / hand-rolled finite-difference check
is a fixed-``eps`` heuristic with a hand-tuned tolerance -- it *false-passes* a
subtly wrong gradient (truncation hides the bug) and *false-rejects* a correct
one (cancellation at a small ``eps``). This probe turns
:func:`omnibias.verify.certified_gradient_check` into an instrumented experiment
under the ``omnibias-dev-empirical-validation`` gates: a correct gradient PASSES
across ``K>=8`` random nets, a deliberately-scaled-wrong gradient is REJECTED
*with a Lean-checkable proof of mismatch*, and against the **named baseline**
(naive fixed-``eps`` one-sided FD) the certified band both catches a bug the
baseline misses and clears a correct gradient the baseline wrongly fails.

Honesty labels: each true partial is enclosed by the **closed-form** derivative
tower intersected with the finite-difference sandwich; the residual band
``autodiff_grad_i - true_partial_i`` is a certified (**numerical**) enclosure; a
sign-definite band is a *proof* the coordinate is wrong.
"""

from __future__ import annotations

import math
import random

try:
    import mpmath  # noqa: F401  (kept for parity with the other probes' oracle contract)
except ImportError:  # pragma: no cover - the [test] extra ships mpmath
    print("this smoke needs mpmath: pip install 'omnibias-difference[test]'")
    raise SystemExit(0) from None

from omnibias.core.proof.lean_check import generate_obligation
from omnibias.difference.validation import FindingsLedger, seed_sweep
from omnibias.verify import (
    certified_gradient_check,
    gradient_residual_certificate,
    mlp_axis_oracles,
)

_D = 3
_H = 4


def _make_mlp(seed: int) -> tuple[list[list[float]], list[float], list[float]]:
    """A random one-hidden-layer scalar tanh MLP (weights as plain nested lists)."""
    rng = random.Random(seed)
    weights = [[rng.uniform(-1.2, 1.2) for _ in range(_D)] for _ in range(_H)]
    biases = [rng.uniform(-0.5, 0.5) for _ in range(_H)]
    out = [rng.uniform(-1.2, 1.2) for _ in range(_H)]
    return weights, biases, out


def _mlp_f(w: list[list[float]], b: list[float], v: list[float], x: list[float]) -> float:
    return sum(v[j] * math.tanh(sum(w[j][i] * x[i] for i in range(_D)) + b[j]) for j in range(_H))


def _mlp_grad(w: list[list[float]], b: list[float], v: list[float], x: list[float]) -> list[float]:
    """The exact analytic gradient (the 'autodiff' the check is meant to trust)."""
    grad = [0.0] * _D
    for j in range(_H):
        z = sum(w[j][i] * x[i] for i in range(_D)) + b[j]
        sech2 = 1.0 / math.cosh(z) ** 2
        for i in range(_D):
            grad[i] += v[j] * w[j][i] * sech2
    return grad


def _point(seed: int) -> list[float]:
    rng = random.Random(10_000 + seed)
    return [rng.uniform(-1.0, 1.0) for _ in range(_D)]


def _naive_fd_pass(
    f, point: list[float], grad: list[float], *, eps: float, tol: float
) -> bool:
    """The baseline: one-sided fixed-``eps`` FD check with a fixed tolerance."""
    f0 = f(point)
    for i in range(_D):
        shifted = list(point)
        shifted[i] += eps
        fd = (f(shifted) - f0) / eps
        if abs(grad[i] - fd) > tol:
            return False
    return True


def probe_correct_passes(ledger: FindingsLedger) -> None:
    """A correct gradient passes the certified check across K=8 random nets."""
    print("=== W6-ext: correct autodiff gradient PASSES (K=8 nets) ===")

    def worst_residual(seed: int) -> float:
        w, b, v = _make_mlp(seed)
        pt = _point(seed)
        grad = _mlp_grad(w, b, v, pt)
        fns, bounds = mlp_axis_oracles(w, b, v, "tanh", pt)
        gc = certified_gradient_check(fns, bounds, grad, pt, step=1e-2)
        assert gc.passed, f"correct gradient rejected on seed {seed}: {gc.mismatched_coordinates()}"
        return gc.max_abs_residual

    sweep = seed_sweep(worst_residual, range(8))
    print(f"  correct grad max|residual| across 8 nets: max {sweep['max']:.2e} (all bands contain 0)")
    assert sweep["max"] < 1e-9
    ledger.add("gradcheck", "info", f"correct gradient passes across 8 nets (max resid {sweep['max']:.1e})")


def probe_wrong_rejected(ledger: FindingsLedger) -> None:
    """A scaled-wrong gradient is REJECTED with a Lean-checkable mismatch obligation."""
    print("=== W6-ext: scaled-wrong gradient REJECTED with a proof of mismatch (K=8) ===")
    rejected = 0
    demo_theorem = ""
    for seed in range(8):
        w, b, v = _make_mlp(seed)
        pt = _point(seed)
        grad = _mlp_grad(w, b, v, pt)
        wrong = [1.5 * g for g in grad]  # a uniformly-scaled (wrong) gradient
        fns, bounds = mlp_axis_oracles(w, b, v, "tanh", pt)
        gc = certified_gradient_check(fns, bounds, wrong, pt, step=1e-2)
        if not gc.passed:
            rejected += 1
        # Seal the largest-magnitude mismatch and extract its finite Lean obligation.
        mism = gc.mismatched_coordinates()
        assert mism, f"scaled-wrong gradient not caught on seed {seed}"
        worst = max(mism, key=lambda i: gc.true_partials[i].mag)
        cert = gradient_residual_certificate(gc, worst)
        obligation = generate_obligation(cert)
        assert obligation is not None, "sign-definite residual must yield a finite Lean obligation"
        if not demo_theorem:
            demo_theorem = obligation
    print(f"  {rejected}/8 scaled-wrong gradients rejected; each mismatch seals a Lean obligation")
    print("  example emitted obligation (enclosed-quantity sign, re-checkable by the kernel):")
    theorem_line = next(
        (ln.strip() for ln in demo_theorem.splitlines() if ln.lstrip().startswith("theorem")),
        "theorem obligation ...",
    )
    print("   ", theorem_line)
    assert rejected == 8
    ledger.add(
        "gradcheck",
        "info",
        "scaled-wrong gradient rejected across 8 nets; sign-definite residual -> Lean obligation",
    )


def probe_baseline_false_pass(ledger: FindingsLedger) -> None:
    """Baseline false-pass: a subtle 1e-3 bug slips past naive fixed-eps FD; certified rejects."""
    print("=== W6-ext: certified catches a 1e-3 bug the naive loose-eps check MISSES (K=8) ===")
    naive_false_pass = 0
    cert_reject = 0
    for seed in range(8):
        w, b, v = _make_mlp(seed)
        pt = _point(seed)
        grad = _mlp_grad(w, b, v, pt)
        coord = seed % _D
        buggy = list(grad)
        buggy[coord] += 1.0e-3  # a small, realistic gradient bug

        def f(x: list[float], _w=w, _b=b, _v=v) -> float:
            return _mlp_f(_w, _b, _v, x)

        # Baseline: eps=1e-3 forces a loose tol=1e-2 to tolerate its own O(eps) truncation.
        if _naive_fd_pass(f, pt, buggy, eps=1e-3, tol=1e-2):
            naive_false_pass += 1
        fns, bounds = mlp_axis_oracles(w, b, v, "tanh", pt)
        gc = certified_gradient_check(fns, bounds, buggy, pt, step=1e-2)
        if not gc.passed and coord in gc.mismatched_coordinates():
            cert_reject += 1
    print(f"  naive fixed-eps FD false-passes the bug: {naive_false_pass}/8; certified rejects: {cert_reject}/8")
    assert naive_false_pass == 8 and cert_reject == 8
    ledger.add(
        "gradcheck",
        "flaw",
        "naive fixed-eps FD gradient check false-passes a 1e-3 bug the certified band rejects",
        detail="a loose tolerance chosen to absorb O(eps) truncation also hides real gradient errors; "
        "the certified residual encloses the truth to ~1e-14 regardless of the stencil step",
    )


def probe_baseline_false_reject(ledger: FindingsLedger) -> None:
    """Baseline false-reject: cancellation at a tiny eps fails a correct grad; certified passes."""
    print("=== W6-ext: certified clears a correct grad the naive tiny-eps check FALSE-REJECTS (K=8) ===")
    naive_false_reject = 0
    cert_pass = 0
    for seed in range(8):
        w, b, v = _make_mlp(seed)
        pt = _point(seed)
        grad = _mlp_grad(w, b, v, pt)  # correct

        def f(x: list[float], _w=w, _b=b, _v=v) -> float:
            return _mlp_f(_w, _b, _v, x)

        # Baseline: eps=1e-12 -> catastrophic cancellation in f(x+eps)-f(x) at tol=1e-6.
        if not _naive_fd_pass(f, pt, grad, eps=1e-12, tol=1e-6):
            naive_false_reject += 1
        fns, bounds = mlp_axis_oracles(w, b, v, "tanh", pt)
        gc = certified_gradient_check(fns, bounds, grad, pt, step=1e-2)
        if gc.passed:
            cert_pass += 1
    print(f"  naive tiny-eps FD false-rejects the correct grad: {naive_false_reject}/8; certified passes: {cert_pass}/8")
    assert naive_false_reject == 8 and cert_pass == 8
    ledger.add(
        "gradcheck",
        "flaw",
        "naive tiny-eps FD gradient check false-rejects a correct gradient (cancellation)",
        detail="there is no single fixed eps that avoids both O(eps) truncation and O(1/eps) cancellation; "
        "the closed-form tower enclosure the certified check uses has neither failure mode",
    )


PROBES = (
    probe_correct_passes,
    probe_wrong_rejected,
    probe_baseline_false_pass,
    probe_baseline_false_reject,
)


def main() -> None:
    ledger = FindingsLedger("gradcheck_validate")
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
    print("OK: all certified gradient-check probes pass their gates.")


if __name__ == "__main__":
    main()
