# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Certified decoding -- prove a Viterbi decode survives an input perturbation ball.

Run:

    pip install omnibias-struct
    python docs/examples/certified_decoding.py

The Viterbi decode of a linear chain is an ``argmax`` over exponentially many paths. This
demo certifies that the winning path ``p*`` stays *the* unique argmax for **every** emission
matrix in the ball ``||x - x0||_inf <= eps`` -- the structured analogue of adversarial
robustness (``omnibias.verify.certify_robustness``), but over discrete paths instead of a
scalar class margin.

The certified quantity is the worst-case winner-vs-runner-up margin
``M = min_{x in box} [ score(p*, x) - max_{p != p*} score(p, x) ]``, enclosed
(outward-rounded, no path enumeration) by a reduced min-plus DP measured relative to the
winner -- so emissions the winner and a competitor share cancel exactly and the bound is
tight. ``certified`` is ``M.lo > 0``. The certificate is v1-sealed (tamper-evident) and
handed to the Lean bridge; ``theorem_prover_verified`` is earned only by a genuine
``lake build`` pass and degrades gracefully to ``False`` with no toolchain -- never forged.

Terminology: certified decoding is a fact about the ``beta -> inf`` hard ``argmax`` (the
max-plus semiring), proved with rigorous intervals -- it does not involve the ``delta -> 0``
derivative tower at all (there is no derivative here, only a sign).
"""

from __future__ import annotations

import itertools

import numpy as np
from omnibias.struct import ChainTrellis, viterbi
from omnibias.struct.decode import certify_decoding, check_decoding_certificate


def _decode(emis: np.ndarray, trans: np.ndarray, start: np.ndarray) -> tuple[int, ...]:
    return viterbi(ChainTrellis(emis, trans, start))[1]


def certified_decoding_demo() -> None:
    print("=== 1. certify the Viterbi decode over an emission eps-ball ===")
    # A well-separated 3-step, 2-state chain: the "zig-zag" path 0->1->0 clearly wins.
    emissions = np.array([[3.0, 0.0], [0.0, 3.0], [3.0, 0.0]])
    transitions = np.array([[1.0, 0.0], [0.0, 1.0]])
    start = np.zeros(2)
    winner = _decode(emissions, transitions, start)
    print(f"  nominal Viterbi decode p* = {winner}")

    print(f"  {'eps':>6s} {'margin.lo':>11s} {'margin.hi':>11s}  certified")
    for eps in (0.1, 0.3, 0.49, 0.6):
        cert = certify_decoding(emissions, transitions, eps, start=start)
        print(f"  {eps:6.2f} {cert.margin.lo:11.4f} {cert.margin.hi:11.4f}  {cert.certified}")

    # The winner and the closest competitor differ at a single step, so the pointwise
    # margin is 1.0 and the worst case degrades as 1 - 2*eps -> crosses 0 at eps = 0.5.
    tight = certify_decoding(emissions, transitions, 0.3, start=start)
    assert tight.certified and abs(tight.min_margin - (1.0 - 2 * 0.3)) < 1e-9
    print(f"\n  worst-case margin at eps=0.3 is exactly 1 - 2*eps = {tight.min_margin:.4f} (tight).\n")


def empirical_soundness_demo() -> None:
    print("=== 2. empirical check: certified => decode is stable over the whole box ===")
    rng = np.random.default_rng(0)
    emissions = np.array([[3.0, 0.0], [0.0, 3.0], [3.0, 0.2]])
    transitions = np.array([[0.5, -0.5], [-0.5, 0.5]])
    start = np.zeros(2)
    eps = 0.3
    cert = certify_decoding(emissions, transitions, eps, start=start)
    assert cert.certified

    # Sample the box exhaustively at the corners plus a random interior cloud.
    flat = emissions.reshape(-1)
    corners = [
        (flat + eps * np.array(s)).reshape(emissions.shape)
        for s in itertools.product((-1.0, 1.0), repeat=flat.size)
    ]
    cloud = [emissions + rng.uniform(-eps, eps, emissions.shape) for _ in range(2000)]
    stable = all(_decode(x, transitions, start) == cert.winner for x in corners + cloud)
    print(f"  eps={eps}: certified min-margin {cert.min_margin:.4f}; decode == p* over "
          f"{len(corners)} corners + {len(cloud)} random points: {stable}")
    assert stable, "a certified decode must be stable across the entire box"

    # A ball big enough to flip the decode is honestly reported as NOT certified.
    big = certify_decoding(emissions, transitions, 2.0, start=start)
    flipped = next(
        x for x in (emissions + rng.uniform(-2.0, 2.0, emissions.shape) for _ in range(5000))
        if _decode(x, transitions, start) != cert.winner
    )
    print(f"  eps=2.0: certified={big.certified} (margin.lo {big.min_margin:.3f}); a sampled "
          f"perturbation decodes to {_decode(flipped, transitions, start)} != p*")
    assert not big.certified
    print()


def sealed_and_lean_demo() -> None:
    print("=== 3. seal the certificate + Lean-kernel bridge (honest, never forged) ===")
    emissions = np.array([[3.0, 0.0], [0.0, 3.0], [3.0, 0.0]])
    transitions = np.array([[1.0, 0.0], [0.0, 1.0]])
    cert = certify_decoding(emissions, transitions, 0.3)
    verdict = check_decoding_certificate(cert)
    print(f"  sealed digest verifies: {verdict.sealed_ok}")
    print(f"  finite Lean obligation generated (margin.lo>0): {verdict.obligation_generated}")
    print(f"  Lean toolchain available: {verdict.lean.available}   "
          f"theorem_prover_verified: {verdict.theorem_prover_verified}")
    assert verdict.sealed_ok and verdict.obligation_generated
    # The flag is *exactly* the real kernel verdict -- with no toolchain it stays False.
    assert verdict.theorem_prover_verified == verdict.lean.verified
    print("\n  Reading: certified decoding is a sound, tight, sealed sign fact about the hard")
    print("  argmax; the Lean flag is earned only by a genuine lake pass, never asserted.\n")


def main() -> None:
    certified_decoding_demo()
    empirical_soundness_demo()
    sealed_and_lean_demo()
    print("OK: winner certified over the eps-ball; stable across the box; larger ball honestly "
          "uncertified; certificate sealed + Lean-bridged.")


if __name__ == "__main__":
    main()
