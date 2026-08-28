---
name: omnibias-formal-agent
description: Drive omnibias's Mathlib-backed formal loop — classify a certificate's finite obligation, discharge it over the rationals with drive_obligation / lake build, and attach the mathlib_verified verdict tier. Use when discharging rational / positive-definite / Newton-Kantorovich obligations or extending the omnibias-analytic checker.
---

# Mathlib-backed formal loop

This skill drives the Mathlib-backed formal register: a certificate's finite
obligation becomes Lean that Mathlib's kernel re-checks, reported on
`mathlib_verified`.

## Why nested AD fails

Autodiff cannot discharge an identity over Q. The Mathlib-free kernel is
intentionally small; without a classify-generate-build driver, larger rational
obligations stay unreplayed. Conflating `mathlib_verified` with
`theorem_prover_verified` loses the trust-base distinction.

## What only this tower unlocks

`drive_obligation`: classify → generate → `lake build` → `DriveReport` with a
rule-based `next_action`. Bridge re-derives each obligation exactly over Q
and emits only what it can confirm. `formal/omnibias-analytic` is sorry-free.
This is a larger, honestly labelled trust base over Q / R, distinct from the
ZInterval kernel.

Sibling: `omnibias-certificate-lean` owns `theorem_prover_verified` and never
touches `omnibias-analytic`.

## Use

- Driver: `omnibias.formal.drive.drive_obligation` (deterministic plumbing).
- Bridge: `omnibias.formal.mathlib_check` (`classify_obligation`,
  `generate_obligation`, `check_certificate`).
- Verdict tier: `omnibias.formal.augment.evaluate_with_mathlib` (core `Verdict`
  stays frozen).
- Lean project: `formal/omnibias-analytic` (non-blocking
  `.github/workflows/lean-analytic.yml`).

```bash
python -m pytest packages/omnibias-formal/tests -q
# where Mathlib is available:
cd formal/omnibias-analytic && lake build
```

## Extend

Source: `packages/omnibias-formal`. Compose with `omnibias-certificate-lean`,
`omnibias-verify`, `omnibias-sos`. New Check lemmas stay finite / rational.
Tests cover classify, generate, and the attached tier.

## Next invention

A new Check lemma (positive-definite inertia, Newton-Kantorovich radius, or
rational stencil poisedness) that `drive_obligation` classifies and Mathlib
re-checks in CI's analytic job.

## Further references

- `docs/scope-and-guarantees.md`
- `packages/omnibias-formal`
