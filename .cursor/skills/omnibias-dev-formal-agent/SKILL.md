---
name: omnibias-dev-formal-agent
description: Drive omnibias's Mathlib-backed formal loop -- classify a certificate's finite obligation, discharge it over the rationals with drive_obligation / lake build, and attach the mathlib_verified verdict tier. Use when driving formal proofs against Mathlib, discharging rational / positive-definite / Newton-Kantorovich obligations, or extending the omnibias-analytic checker; scoped to achievable finite obligations, never a proof of global regularity, and mathlib_verified is earned only by a genuine lake pass and never conflated with theorem_prover_verified. For contributors modifying omnibias itself, not for consumers using it.
---

# Driving the Mathlib-backed formal loop

omnibias runs one derivative tower in three registers -- differentiable,
rigorous, and **formal**. This skill drives the *Mathlib-backed* formal register:
it turns a certificate's finite obligation into Lean that Mathlib's kernel
re-checks, and reports the result on the `mathlib_verified` tier. Its credibility
depends entirely on never overstating what was proven, so it is scoped to
**achievable, finite** obligations -- never a proof of global regularity.

This is the sibling of `omnibias-dev-certificate-lean`; keep the two apart:

- `omnibias-dev-certificate-lean` owns the tiny **Mathlib-free** kernel and
  `theorem_prover_verified` (integer `ZInterval` arithmetic).
- **This skill** owns the **Mathlib-backed** project `formal/omnibias-analytic`
  and `mathlib_verified` -- a larger, honestly-labelled trust base over `ℚ` / `ℝ`.

## Where it lives

- Driver: `omnibias.formal.drive.drive_obligation` -- one deterministic pass
  (`classify -> generate -> lake build -> report`), returning a `DriveReport` with
  a rule-based `next_action`. Deterministic plumbing, **not** an autonomous prover.
- Bridge: `omnibias.formal.mathlib_check` (`classify_obligation`,
  `generate_obligation`, `check_certificate`) -- re-derives each obligation
  exactly over `ℚ` and refuses to emit anything it cannot itself confirm.
- Verdict tier: `omnibias.formal.augment.evaluate_with_mathlib` attaches the
  `mathlib_verified` tier consumer-side (the core `Verdict` is frozen / untouched).
- Lean project: `formal/omnibias-analytic` -- proven, `sorry`-free `Check/*`
  lemmas plus the bridge-overwritten `Generated.lean`. It builds on the
  non-blocking `.github/workflows/lean-analytic.yml` (Mathlib is off the fast path).

## The loop

1. Obtain / seal a certificate carrying a finite obligation.
2. `report = drive_obligation(cert)`. Read `report.obligation_class`,
   `report.verified`, `report.failure`, and follow `report.next_action`.
3. On failure the emitted `report.obligation` is the source of truth: fix the
   certificate's payload, or add / repair a **`sorry`-free** capability lemma in
   `OmnibiasAnalytic/Check/*` plus a matching generator in `mathlib_check.py`.
   **Never hand-edit `Generated.lean`** -- the bridge overwrites then restores it.
4. On a genuine pass, adjudicate with `evaluate_with_mathlib(...)` so the verdict
   carries `mathlib_verified` (asserting the claim without a pass is `BLOCKED`).

## Scoped task types

- **MAY attempt** -- finite, decidable obligations the checker (or a new `Check/*`
  lemma) can discharge over `ℚ` / `ℝ`: enclosed-quantity sign, positive-definite
  pivots, Newton-Kantorovich / Krawczyk contraction, PDE a-posteriori margins,
  Taylor-model centre values; and formalising sound sub-lemmas.
- **MUST NOT attempt** -- any infinite / analytic obligation: limits, continuum
  statements, asymptotics, or an open conjecture. They are out of scope and are
  not expressed in Lean; a green build never means such a theorem.

## The rules you must not break

- `mathlib_verified` is set **only** on a genuine `lake build` pass. It can never
  be forged or asserted into existence; a bare assertion without a pass must block
  the verdict (mirrors `theorem_prover_verified`).
- Never conflate `mathlib_verified` with `theorem_prover_verified`, and neither
  ever implies `unproven_claim`.
- Keep every `OmnibiasAnalytic` module `sorry`-free (CI audits this). An
  obligation you cannot discharge is out of scope -- do not admit it with a
  `sorry` to make the build green.
- The bridge only ever emits an obligation it re-derived exactly over `ℚ`; do not
  weaken that (never emit a fact you have not confirmed). Never edit
  `omnibias.core` to add the tier -- it is attached consumer-side.

## Checklist

- Any new / repaired capability lemma is `sorry`-free and has a regression test.
- Test both paths: obligation generation (fast, no `lake`) and graceful
  degradation when no Lean toolchain is present.

```bash
python -m pytest packages/omnibias-formal/tests -q
# Mathlib-backed build (only where a Lean toolchain exists):
cd formal/omnibias-analytic && lake exe cache get && lake build
# honesty audit: every module is sorry-free
```
