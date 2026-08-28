---
name: omnibias-formal
description: Drive the Mathlib-backed formal checker: classify a certificate's finite rational obligation, discharge it with drive_obligation / lake build, and attach mathlib_verified. Use when inventing a new Check lemma or a new obligation class over Q.
---

# omnibias-formal

Mathlib-backed formal checker for omnibias certificates: drives the
formal/omnibias-analytic Lean project to discharge a certificate's rational/real finite
obligations, reporting a `mathlib_verified` tier distinct from (and never conflated
with) the Mathlib-free minimal kernel's `theorem_prover_verified`.

## Why nested AD fails

Autodiff cannot discharge a rational identity. A float residual is not a Lean proof.
Without a classify-generate-build loop, certificates stay JSON blobs with no kernel
replay.

## What only this tower unlocks

Mathlib-backed formal checker for omnibias certificates: drives the
formal/omnibias-analytic Lean project to discharge a certificate's rational/real finite
obligations, reporting a `mathlib_verified` tier distinct from (and never conflated
with) the Mathlib-free minimal kernel's `theorem_prover_verified`.

Coefficients live in `omnibias.core.polynomials` and are imported, never forked.
When torch and jax twins exist they stay bit-identical by construction. Tracked
files stay vendor-neutral.

## Use

Sibling of `omnibias-certificate-lean` (Mathlib-free `theorem_prover_verified`).
This package owns `formal/omnibias-analytic` and `mathlib_verified`.

| You want | Import |
| --- | --- |
| One-shot driver | `omnibias.formal.drive.drive_obligation` |
| Classify / generate | `omnibias.formal.mathlib_check` |
| Attach the tier | `omnibias.formal.augment.evaluate_with_mathlib` |

## Extend

- Source: [`packages/omnibias-formal`](../../../packages/omnibias-formal).
- Namespace: `omnibias.formal`. Inspect `__init__.py` and `docs/packages.md` before changing a public seam.
- Tests: `python -m pytest packages/omnibias-formal/tests -q`.
- Compose with `omnibias-formal-agent`, `omnibias-certificate-lean`, `omnibias-verify` by **new names**.
- New tensors follow the framework default dtype. Add a regression test for every behavioral change. Regenerate sorted `__all__` when a public symbol moves.
- Heavy compute follows the workspace rule; artifacts go to `$OMNIBIAS_SCRATCH`.

## Next invention

A new finite obligation class (e.g. rational stencil poisedness over a larger
multi-index) that `drive_obligation` classifies and Mathlib re-checks sorry-free.


## Further references

- Capability matrix: [`docs/operator-surface.md`](../../../docs/operator-surface.md)
- Package index: [`docs/packages.md`](../../../docs/packages.md)
- Repository map: [`AGENTS.md`](../../../AGENTS.md)
