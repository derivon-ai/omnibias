# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Exact fast-paths, a sound router, and a separated statistical layer -- omnibias-logic.

Run:

    pip install omnibias-logic
    python docs/examples/logic_exact_and_approx_counting.py

Model counting (`#SAT`) is `#P`-hard, so omnibias-logic never claims a general poly-time exact
count. Instead it offers a **guarantee taxonomy**, and this deterministic, CPU-tiny demo walks
all three tiers (no tensor backend required):

1. **Exact + sound on a tractable fragment.** On an affine (XOR) instance the affine GF(2)
   counter, the bounded-treewidth DP, the component-caching DPLL counter, and the certified
   inclusion-exclusion enclosure *all agree* on the exact count.
2. **A sound-only router.** `count()` auto-picks the cheapest sound method per instance
   (affine -> treewidth -> DPLL -> certified enclosure) and tags the guarantee it earned;
   under a tiny budget it falls back to the rigorous enclosure rather than overclaim.
3. **Certified + formally checkable.** A sound count seals into a tamper-evident v1 certificate
   whose *finite* obligation the Mathlib-free Lean kernel re-checks: a tight, unweighted count
   becomes the integer identity `Z0 - S1 + S2 - ... = #models` (includes certified UNSAT), a
   positive lower bound becomes a certified-satisfiability sign. `theorem_prover_verified` is
   earned only by a genuine `lake build` and degrades gracefully with no toolchain.
4. **A quarantined statistical layer (NOT worst-case sound).** `omnibias.logic.approx` gives an
   `(epsilon, delta)` XOR-hashing estimate and a split-conformal coverage interval -- returned
   as an `ApproxCount` that can never be mistaken for a rigorous `CountCertificate`.

The counting fast-paths are the two genuinely tractable regimes (affine and bounded
treewidth); `#2-SAT` / `#Horn-SAT` are `#P`-complete and deliberately excluded.
"""

from __future__ import annotations

import copy

import numpy as np
from omnibias.core.proof.certificate import schema_errors_v1
from omnibias.logic import (
    CountResult,
    check_certificate,
    count,
    count_enclosure,
    count_models_exact,
    exact_model_count,
    model_count,
    prove_model_count,
    seal_count_certificate,
    treewidth_model_count,
    verify_certificate_digest,
    xor_model_count,
)
from omnibias.logic.approx import ApproxCount, ConformalCounter, approx_model_count
from omnibias.logic.model_count import CountCertificate
from omnibias.logic.model_count.xor import XORClause, detect_xor_system


def _xor_to_cnf(xors: list[XORClause]) -> list[list[int]]:
    """The standard CNF encoding of a parity system (each XOR -> 2^(k-1) clauses)."""
    clauses: list[list[int]] = []
    for xc in xors:
        variables = list(xc.variables)
        k = len(variables)
        for pattern in range(1 << k):
            signs = [(pattern >> i) & 1 for i in range(k)]
            if sum(signs) % 2 == (1 - xc.parity):
                clauses.append([-variables[i] if signs[i] else variables[i] for i in range(k)])
    return clauses


def exact_agreement_demo() -> None:
    print("=== 1. exact sound counters agree on an affine (XOR) instance ===")
    # x1 xor x2 = 0, x2 xor x3 = 1, over n = 4 (x4 free): an affine subspace.
    xors = [XORClause((1, 2), 0), XORClause((2, 3), 1)]
    n = 4
    clauses = _xor_to_cnf(xors)
    mc = model_count(clauses, n_vars=n)

    detected = detect_xor_system(mc)
    assert detected is not None, "this CNF is exactly a parity system"
    affine = xor_model_count(detected, n)  # 2^(n - rank) = 2^(4 - 2) = 4
    dpll = count_models_exact(mc)
    tw, width = treewidth_model_count(mc, max_width=8)
    exact = exact_model_count(mc)  # O(2^n) enumeration oracle (ground truth)
    enc = count_enclosure(mc, order=len(clauses) + 1)  # full inclusion-exclusion -> tight

    print(f"  affine GF(2):     {affine}")
    print(f"  DPLL (#components): {dpll}")
    print(f"  treewidth DP:     {tw}  (heuristic width {width})")
    print(f"  enclosure:        [{enc.lower:.0f}, {enc.upper:.0f}]  tight={enc.tight}")
    print(f"  O(2^n) oracle:    {exact:.0f}")
    assert affine == dpll == tw == int(exact) == 4
    assert enc.contains(exact) and enc.tight
    print("  Reading: four independent sound methods land on the same exact count.\n")


def router_demo() -> None:
    print("=== 2. the sound router auto-picks the cheapest method (and tags the guarantee) ===")

    def report(label: str, result: CountResult, oracle: float) -> None:
        body = (
            f"exact = {result.value}"
            if result.is_exact
            else f"enclosure = [{result.lower:.0f}, {result.upper:.0f}]"
        )
        print(f"  {label:<22s} method={result.method:<18s} guarantee={result.guarantee:<20s} {body}")
        assert result.is_sound and result.contains(oracle)

    affine = model_count(_xor_to_cnf([XORClause((1, 2), 0), XORClause((2, 3), 1)]), n_vars=4)
    small = model_count([[1, 2], [2, 3], [-3, 4]], n_vars=4)
    report("affine instance", count(affine), exact_model_count(affine))
    report("small CNF", count(small), exact_model_count(small))

    # A denser instance forced to fall back: tiny treewidth cap + tiny DPLL budget.
    rng = np.random.default_rng(11)
    n = 12
    dense = model_count(
        [
            [int(s * v) for s, v in zip(rng.choice([-1, 1], 3), rng.choice(np.arange(1, n + 1), 3, replace=False), strict=True)]
            for _ in range(int(4.3 * n))
        ],
        n_vars=n,
    )
    report("dense (fallback)", count(dense, max_width=1, node_budget=1), exact_model_count(dense))
    print("  Reading: every route is worst-case sound; the router never overclaims.\n")


def certified_demo() -> None:
    print("=== 3. seal a sound count into a tamper-evident, Lean-checkable certificate ===")
    # A tight, unweighted enclosure -> the exact count, sealed as a finite integer identity.
    mc = model_count([[1, 2], [-1, 3], [2, -3]], n_vars=3)
    cert = count_enclosure(mc, order=len(mc.cnf.clauses) + 1)  # order >= #clauses -> tight
    sealed = seal_count_certificate(cert, problem=mc)
    exact = int(exact_model_count(mc))

    payload = sealed["payload"]
    assembled = sum(c * m for c, m in payload["lhs_terms"])
    print(f"  payload={payload['type']}   claim: {sealed['claim']}")
    print(f"  kernel obligation: sum_i c_i m_i = {assembled} == rhs {payload['rhs']} == exact {exact}")
    assert payload["type"] == "rational_identity"
    assert assembled == payload["rhs"] == exact
    assert schema_errors_v1(sealed) == [] and verify_certificate_digest(sealed)

    # Hand it to the Lean kernel (degrades gracefully when no lake toolchain is installed).
    result = check_certificate(sealed)
    if result.available:  # pragma: no cover - only on a Lean-equipped runner
        print(f"  Lean kernel: available, verified={result.verified}")
    else:
        print("  Lean kernel: toolchain unavailable -> theorem_prover_verified stays False "
              "(sealed + digest-verified regardless)")

    # Tamper-evidence: mutate the sealed count and the sha256 digest no longer matches.
    forged = copy.deepcopy(sealed)
    forged["payload"]["rhs"] = exact + 1
    print(f"  tamper (rhs {exact} -> {exact + 1}) detected: {not verify_certificate_digest(forged)}")
    assert not verify_certificate_digest(forged)

    # UNSAT seals to a kernel-checkable #models = 0; a positive lower bound to a sat sign.
    unsat = model_count([[1], [-1]], n_vars=1)
    unsat_cert = seal_count_certificate(count_enclosure(unsat, order=2), problem=unsat)
    sat_sign = count_enclosure(model_count([[1, 2, 3, 4]], n_vars=6), order=1).seal()
    print(f"  UNSAT -> rhs {unsat_cert['payload']['rhs']} (certified 0);  positive lower bound -> "
          f"{sat_sign['meta']['obligation']} (certified satisfiable)")
    assert unsat_cert["payload"]["rhs"] == 0
    assert sat_sign["meta"]["obligation"] == "interval_sign"

    # The same certificate flows through the generic ProofMachine as a full Verdict: the
    # prover produces + seals it, then the machine layers schema + independent-oracle replay +
    # honesty + (opt-in) Lean gates. assert_theorem_prover=True *requires* a real kernel pass.
    verdict = prove_model_count(mc, claim="count", value=exact, lean_check=True)
    print(f"  ProofMachine: {verdict.status}  schema_ok={verdict.schema_ok}  "
          f"replay_ok={verdict.replay_ok}  theorem_prover_verified={verdict.theorem_prover_verified}")
    assert verdict.proved and verdict.schema_ok and verdict.replay_ok is True
    strict_unsat = prove_model_count(unsat, claim="unsat", strict=True)
    assert strict_unsat.proved  # tight [0, 0] + independent oracle replay survive strict mode
    blocked = prove_model_count(mc, claim="count", value=exact, assert_theorem_prover=True)
    assert blocked.blocked or blocked.theorem_prover_verified  # blocks unless a kernel really passed
    print("  Reading: the sound count becomes a portable, tamper-evident, kernel-checkable object\n"
          "  that adjudicates as PROVED / DISPROVED / BLOCKED through the shared ProofMachine.\n")


def approx_demo() -> None:
    print("=== 4. STATISTICAL layer -- omnibias.logic.approx -- NOT worst-case sound ===")
    print("  (probabilistic / coverage guarantees only; an ApproxCount is never a CountCertificate)")

    # (epsilon, delta) XOR-hashing estimate on a moderate instance.
    rng = np.random.default_rng(0)
    n = 9
    clauses = [
        [int(s * v) for s, v in zip(rng.choice([-1, 1], 2), rng.choice(np.arange(1, n + 1), 2, replace=False), strict=True)]
        for _ in range(6)
    ]
    mc = model_count(clauses, n_vars=n)
    exact = exact_model_count(mc)
    approx = approx_model_count(mc, epsilon=0.8, delta=0.2, seed=1)
    print(f"\n  hashing:  estimate {approx.estimate:.0f}  interval [{approx.lower:.0f}, "
          f"{approx.upper:.0f}]  (exact {exact:.0f})  worst_case_sound={approx.worst_case_sound}")
    assert isinstance(approx, ApproxCount) and not isinstance(approx, CountCertificate)
    assert approx.worst_case_sound is False
    assert approx.contains(exact)

    # Split-conformal coverage over a held-out split.
    alpha = 0.2
    cal_rng = np.random.default_rng(2024)

    def make(rng_: np.random.Generator) -> object:
        nn = int(rng_.integers(4, 7))
        cs = []
        for _ in range(int(rng_.integers(1, 4))):
            k = int(rng_.integers(1, 3))
            variables = rng_.choice(np.arange(1, nn + 1), size=k, replace=False)
            signs = rng_.choice([-1, 1], size=k)
            cs.append([int(s * v) for s, v in zip(signs, variables, strict=True)])
        return model_count(cs, n_vars=nn)

    problems = [make(cal_rng) for _ in range(120)]
    truths = [exact_model_count(p) for p in problems]
    conformal = ConformalCounter(alpha=alpha, seed=0, samples=1500)
    conformal.fit(problems[:80], truths[:80])
    covered = sum(conformal.predict(p).contains(t) for p, t in zip(problems[80:], truths[80:], strict=True))
    coverage = covered / len(problems[80:])
    print(f"  conformal: held-out coverage {coverage:.2f}  (target >= {1 - alpha:.2f}; marginal)")
    assert coverage >= 1.0 - alpha - 0.15  # generous finite-sample slack -> non-flaky
    print("\n  Reading: useful estimates, but explicitly outside the sound surface -- they carry\n"
          "  a probabilistic/coverage guarantee and a distinct, non-forgeable result type.\n")


def main() -> None:
    exact_agreement_demo()
    router_demo()
    certified_demo()
    approx_demo()
    print("OK: exact fast-paths agree with the oracle, the router tags every result sound, sound\n"
          "counts seal into tamper-evident Lean-checkable certificates, and the statistical layer\n"
          "stays type-quarantined from the certified counts.")


if __name__ == "__main__":
    main()
