# 07-08 Convergence ledgers: the finite spine of both parents

## 1. Thesis and status

Whether an infinite staged construction converges is, for two Clay-adjacent
parents, a finite rational inequality system: minima of affine forms, closed
by exact arithmetic. OpenAI's Navier–Stokes `ExponentLedger` decides the
packet-iteration ladder under `kappa <= 1/100000` and `sigma >= 1/5`. The
Kotecký–Preiss polymer-coordination majorants already locked in
`Check/Polymer.lean` decide the strong-coupling cluster expansion at
spacetime dimension four. omnibias already had the inequality engine and a
Mathlib-backed checker; it did not have a primitive that *is* that object.

- **Status**: shipped (G1–G5 CI; founding bias collapse, not temperature collapse; not Clay (A)/(B); not a continuum mass gap)
- **Depends on**: 01-11, 07-01, 09-30
- **Blocks**: none

## 2. Where it lands

`omnibias.core.proof.obligations.convergence_ledger` in `omnibias-core`,
alongside the existing rational-stencil obligation. No new package: the
object is a finite rational Lean obligation, the same domain, dependency
tier, and audience as spec 01-11. The Mathlib-backed lemmas live in
`formal/omnibias-analytic/OmnibiasAnalytic/Check/ConvergenceLedger.lean`.

## 3. Prior art in omnibias

The engines this spec consumes are shipped.

- `omnibias.core.proof.inequality` — propose / rationalize / check; a float
  residual is never an `ExactCheck`.
- `omnibias.core.proof.obligations.rational_stencil` — finite rational Lean
  obligations, kernel-tier `allRatEq` / `allIntGe`.
- `omnibias.core.proof.lean_check` — Mathlib-free kernel emitter.
- `omnibias.formal.mathlib_check` — Mathlib-backed `mathlib_verified` tier.
- `OmnibiasAnalytic.Check.Polymer` — locked `15 < 20` and `15 < 24`.
- `omnibias.core.proof.certificate` — hash-sealed v1 certificates;
  `theorem_prover_verified` is reserved.

**Confirmed gap.** There was no type for a stage-indexed affine budget, no
exact decision procedure for `< 0` of an affine residual on a possibly
unbounded rational interval, and parent honesty flags were hard-pinned
`False` rather than derived from a discharged empty-premise ledger.

## 4. Mathematics

A ledger is a stage-indexed budget:

- a stage map `s_{n+1} = s_n + step`, rational, with rational `s_0`
- named tracked quantities, each an affine form in the stage variable and
  fixed rational parameters
- gains, each a `min` of finitely many affine forms
- margin obligations `q(s_{n+1}) < q(s_n) + gain(s_n)` (or a
  cross-quantity comparison with a separate left-hand side)
- side conditions such as `kappa <= 1/100000`, `sigma >= 1/5`

The closure theorem is quantified over all `n`. Because the stage map is
affine and the side conditions are preserved, it collapses to a finite
check. Each margin residual is affine in the stage variable, so `< 0` on a
rational (possibly unbounded) feasible interval is decided exactly by
endpoint values plus slope sign. `BLOCKED` means infeasible or
out-of-fragment, never "the solver gave up."

The checker reports which part of each `min` is **binding** and the exact
tipping point of the governing parameter. For the Navier–Stokes instance
that number is `kappa < 1/200`, two orders of magnitude looser than the
manuscript's `1/100000`.

`external_premises` is the load-bearing honesty field: the named analytic
facts the ledger is conditional on. Parent flags become true only when
every margin discharges **and** that list is empty.

## 5. Worked example

OpenAI's `all_stage_arithmetic` at `sigma_0 = 1/5`, `step = 1/10`,
`kappa = 1/100000`. The wave exponent is `1/2 + sigma`; the mean exponent
is `1 + sigma`. Five margins plus the `.17` bar-residual gain all hold.
The binding threshold of the bar gain restates
`17/100 < 9/50 - 2 kappa` as `kappa < 1/200`.

The polymer instance is degenerate in the stage variable: `15 < 20` and
`15 < 24` at `d = 4`, matching `Check/Polymer.lean` bit for bit.

## 6. Proposed API

```
check_ledger(ledger) -> LedgerReport
stage_invariant_obligation(ledger) -> Obligation
ledger_to_inequality_system(ledger) -> InequalitySystem
seal_ledger_certificate(ledger, *, run_lean=True)
navier_stokes_exponent_ledger()
strong_coupling_polymer_ledger()
```

Types: `AffineForm`, `MinForm`, `StageMap`, `SideCondition`,
`MarginObligation`, `ConvergenceLedger`. All `Fraction`-exact.

## 7. Practical use cases

- Independently re-derive and stress-test the arithmetic of a published
  exponent ledger (binding thresholds, admissible `kappa`).
- Print the unmet analytic premises of a Yang–Mills cluster expansion as
  a sealed list rather than as rhetoric.
- Feed a discharged ledger into `solve_inequality` / `ProofMachine` so
  replay and sealing are inherited.

## 8. Acceptance gates

- **G1.** Curated ledgers discharge; kernel source is emitted; the NS
  instance recovers `kappa < 1/200`.
- **G2.** A deliberately failing margin returns `BLOCKED` naming that
  margin, and no kernel obligation is emitted.
- **G3.** No Lean toolchain: `theorem_prover_verified` stays false. A
  genuine `lake` pass sets it.
- **G4.** Editing a sealed field invalidates the digest.
- **G5.** Curated ledgers cannot produce a true parent flag; a
  hand-stamped `True` is refused; an empty-premise discharged toy ledger
  *does* flip the derived flag. `mathlib_verified` stays false on the
  kernel path.

## 9. Benchmark plan

`benchmarks/convergence_ledger.py` writes
`docs/benchmarks/convergence_ledger_smoke.json`. Algebra and honesty
gates run in CI; the kernel pass is the Lean job.

## 10. Honesty and scope

Lean certifies the **algebra**. It does not define analytic classes,
construct a correction, or prove a PDE. Clay unforced regularity
(A)/(B) stays an external obligation. Continuum / thermodynamic limit,
Osterwalder–Schrader reconstruction, and a spacing-uniform gap lower
bound stay an external obligation on the Yang–Mills side. The
forbidden phrase on the honesty register is unchanged.

Two collapse senses must not be conflated. This spec is finite
rational arithmetic in the founding **bias-collapse** register (exact
identities, no `delta -> 0` taken here). It is not a temperature
collapse (`beta -> inf`).

## 11. Open questions and risks

- **Premises dominate.** Discharging the arithmetic does not shorten
  the analytic construction. That is the point of `external_premises`.
- **Transcription drift.** The NS ledger is a faithful attributed
  transcription; a later manuscript revision must be re-read, not
  assumed.
- **Falsifier.** If a curated ledger's Python decision disagrees with
  the Lean instance, the spec is wrong and the Lean file is the
  independent arbiter.

## 12. Implementation checklist

- [x] `omnibias.core.proof.obligations.convergence_ledger`
- [x] `navier_stokes_exponent_ledger` / `strong_coupling_polymer_ledger`
- [x] Derived parent flags in `make_certificate` / `_merge_honesty`
- [x] Catalog entry + `ledger_to_inequality_system`
- [x] `OmnibiasAnalytic.Check.ConvergenceLedger`
- [x] Kernel `allRatLt` + Mathlib generator
- [x] `packages/omnibias-core/tests/test_convergence_ledger.py`
- [x] `benchmarks/convergence_ledger.py` plus smoke JSON
- [x] Docs page and nav entry
- [x] Index row in `theory/README.md`

## 13. Parent problem and the exact reason it stays an external obligation

**Parents: Navier-Stokes unforced regularity (Clay A/B) and Yang-Mills
existence and mass gap (Clay).**

The finite spine is not the parent. Clay alternatives (C) and (D)
(forced blowup on `R^3` and the torus) are resolved by a packet-iteration
construction whose *arithmetic* this ledger re-derives; unforced
regularity (A)/(B) is still open, and this module does not claim it.
Unforced 3D Euler blowup on `R^3` is likewise a resolved finite-time
singularity parent; a CCF residual hit is no longer novel against that
parent.

On the Yang-Mills side the polymer majorants `15 < 20` and `15 < 24`
are the same finite arithmetic already in `Check/Polymer.lean`. The
mass gap stays an external obligation because the continuum /
thermodynamic limit, Osterwalder–Schrader reconstruction, and a
spacing-uniform spectral-gap lower bound are not on the ledger. Those
premises are a printed list, not a speech.

A parent-level honesty flag flips true only when every margin
discharges and `external_premises` is empty. Both curated instances
ship with a nonempty list, so today's sealed outcome is
`CONDITIONAL` with a measured margin. The mechanism escalates on its
own the moment a premise is discharged; it cannot be stamped by hand.
