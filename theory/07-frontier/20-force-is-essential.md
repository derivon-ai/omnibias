# 07-20 Force is essential

## 1. Thesis and status

Treating the 07-13 concentrating field without the stress / force
correction as a putative unforced solution is `BLOCKED`. The
uncorrected axis jet excludes `{0}` over `Q`. Anisotropy
`ℓ_r / ℓ_z ≍ τ^h` thins as `τ` decreases. Core energy
`τ^{1/2-3h} → 0` while `‖u‖_∞` explodes. That is Clay (C)/(D)
geometry, not an unforced corollary. Setting `f = 0` is a different
PDE.

- **Status**: shipped (G1–G5 CI; founding bias collapse, not temperature collapse; leftover #58 `f0_not_a_corollary`; not Clay A/B; not a C/D reproof)
- **Depends on**: 07-13
- **Blocks**: none

## 2. Where it lands

`omnibias.pinn.certified.forced_flat` (`unforced_limit_of_forced_flat`)
in `omnibias-pinn`. No new package. Same domain and audience as 07-13.
Do not grow `omnibias.pinn.certified.navier_stokes`. Do not import that
module.

## 3. Prior art in omnibias

- `omnibias.pinn.certified.forced_flat.axis_T0` — uncorrected jet
  `(599/400, 0)`; corrected jet `(0, 0)` after `correct_axis_stress`.
- `SimilarityScales` / `LOCKED_TAU_LO` / `LOCKED_TAU_HI` /
  `core_energy_scale` — exact monomials `τ^{-A}` and `τ^{1/2-3h}`.
- Catalog `jet_flat_forced_blowup` / `pulse_family_composition` —
  parent C/D, `already_true`.

**Confirmed gap.** The `F = 0` idea was not recorded as a named
`BLOCKED` plant with leftover `#58`.

## 4. Mathematics

Locked `h = 1/200`, `A = 1/2 + h`. Uncorrected `F = 1 + X` has
regularized axis jet `T_{rθ}/X = 599/400 ≠ 0`. The 07-13 correction
sets `a = (1+h)/4` so that jet is `{0}`. Deleting the correction
restores the nonzero jet: force is part of the construction.

Anisotropy `ℓ_r / ℓ_z ≍ τ^h`. On the locked pair
`τ_lo = 1/4 < τ_hi = 1` and `h > 0`, so the ratio is strictly smaller
at `τ_lo` (the column thins). Core energy exponent `1/2 - 3h = 97/200
> 0`, so energy → 0 while `‖u‖_∞ ≍ τ^{-A} → ∞`. That is bounded-energy
blowup algebra for a *forced* field.

Founding bias-collapse arithmetic over `Q` and outward-rounded
`τ^h = exp(h ln τ)`. Not temperature collapse. Not a Dirac theorem.

## 5. Worked example

Uncorrected `axis_T0 = (599/400, 0)`, which excludes `{0}`. Corrected
`axis_T0 = (0, 0)`. `τ^h` at `τ = 1` is `1`; at `τ = 1/4` it is
`4^{-1/200} < 1`. The `f = 0` plant report is `BLOCKED` with reason
`force_is_part_of_the_construction`. Leftover `#58` is
`f0_not_a_corollary`. Parent flags stay false.

## 6. Proposed API

```
unforced_limit_of_forced_flat() -> report
F0_NOT_A_COROLLARY_LEFTOVER  # leftover_id 58
enclose_anisotropy_ratio(tau) -> Interval
```

Catalog kind `force_is_essential`, parent
`"Navier-Stokes forced blowup (Clay C/D)"`,
`parent_status="already_true"`.

## 7. Practical use cases

- Record that OpenAI Theorem 1.1 does not imply unforced NS by
  deleting `f`.
- Name C/D concentration geometry (anisotropy thinning, vanishing
  core energy) without calling it a Dirac / A/B theorem.
- Keep `NS_AB_EXTERNAL_PREMISES` untouched.

## 8. Acceptance gates

- **G1.** Uncorrected `axis_T0 ≠ (0, 0)`; the `f = 0` plant is
  `BLOCKED` with named reason `force_is_part_of_the_construction`.
- **G2.** Anisotropy ratio strictly decreases as `τ` decreases on the
  locked pair.
- **G3.** Corrected axis jet is `{0}` only after the 07-13 stress
  correction (force), not after deleting it.
- **G4.** Honesty: parent flags false; leftover `#58`
  `f0_not_a_corollary`; `NS_AB_EXTERNAL_PREMISES` unchanged.
- **G5.** Catalog kind `force_is_essential`, parent C/D,
  `parent_status="already_true"`.

## 9. Benchmark plan

`benchmarks/force_is_essential.py` writes
`docs/benchmarks/force_is_essential_smoke.json`. Algebra and honesty
run in CI. `--full` writes under `$OMNIBIAS_SCRATCH` (default
`artifacts/`).

## 10. Honesty and scope

Not Clay (A)/(B). Not a second proof of Clay (C)/(D). Not a uniform
unforced 3-D vorticity majorant. Not a published reduction
“delete `f` ⇒ A/B”. Forbidden: `navier_stokes_proof_claim`,
`continuum_navier_stokes_claim`, `forced_blowup_reproof_claim`.
`theorem_prover_verified` / `mathlib_verified` stay false unless a
genuine `lake build`.

## 11. Open questions and risks

- The C/D joining / pulse / `C^∞` through `t = 1` leftovers of 07-13
  stay leftover.
- Falsifier: uncorrected jet is `{0}`, anisotropy does not thin, or
  leftover `#58` dropped / parent flag stamped.

## 12. Implementation checklist

- [x] `unforced_limit_of_forced_flat` on `forced_flat`
- [x] Leftover `#58` `f0_not_a_corollary`
- [x] Tests + smoke JSON
- [x] Docs subsection on `forced_flat_blowup.md`
- [x] CI job
- [x] Index row in `theory/README.md`

## 13. Parent problem and the exact reason it stays an external obligation

**Parent: Navier-Stokes forced blowup (Clay C/D)**, already true in the
world. This fragment records that force is part of that construction.
It is not a second proof. Unforced Navier-Stokes regularity
(Clay A/B) stays an external obligation: `f = 0` is a different PDE,
not a corollary. The parent-level honesty flag stays false.
