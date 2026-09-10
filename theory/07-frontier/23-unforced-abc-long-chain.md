# 07-23 Unforced ABC long chain

## 1. Thesis and status

Four locked decaying ABC slabs `[0, 1/2]` through `[3/2, 2]` accept
under the same `BKM_BUDGET = 1` as 07-19 / 07-21. Cover end time is
`2`, not `∞`. Manufactured growth is `Halt` / `BLOCKED`. Empty
remaining budget is `search_incomplete`. Leftover `#57` is reused.

- **Status**: shipped (G1–G5 CI; founding bias collapse, not temperature collapse; leftover #57 reused; not Clay A/B)
- **Depends on**: 07-21
- **Blocks**: none

## 2. Where it lands

`omnibias.pinn.certified.unforced.locked_n_abc_slab_continuation` in
`omnibias-pinn`. No new package. Do not grow
`omnibias.pinn.certified.navier_stokes`. Do not import that module.

## 3. Prior art in omnibias

- 07-21 `locked_two_abc_slab_continuation` — two slabs, leftover `#57`.
- `BKM_BUDGET = 1`; decaying ABC integrals shrink in time.

**Confirmed gap.** The ABC continuation stopped at `n_slabs = 2`.

## 4. Mathematics

Same Beltrami BKM hull as 07-21. Later windows have smaller
`e^{-ν k² t0} - e^{-ν k² t1}` because the field has already decayed.
Four steps of horizon `H = 1/2` cover `[0, 2]`. Any finite `n_slabs`
is a finite union. `[0, ∞)` is not that union. Founding bias-collapse
enclosures, not temperature collapse.

## 5. Worked example

`n_slabs = 4`, `remaining_budget = 3`. Decaying ABC accepts four
windows; `cover_end = 2`. Growing ABC on the first window is
`BLOCKED` / `growing_vorticity`. A fifth request with empty budget
is `search_incomplete`. `covers_infinite_time` is false. Leftover
`#57`. Parent flags stay false.

## 6. Proposed API

```
locked_n_abc_slab_continuation(*, n_slabs=4) -> report
LOCKED_ABC_CHAIN_SLABS = 4
```

Catalog kind `unforced_abc_long_chain`, parent
`"Navier-Stokes unforced regularity (Clay A/B)"`,
`parent_status="open"`.

## 7. Practical use cases

- Lengthen the named ABC cover without claiming `[0, ∞)`.
- Keep leftover `#57` as one object for the same four A/B gaps.
- Halt on manufactured growth with the same budget.

## 8. Acceptance gates

- **G1.** Four decaying ABC slabs accept; cover end time is `2`,
  not `∞`.
- **G2.** Growing ABC still `Halt` / `BLOCKED` / `growing_vorticity`.
- **G3.** Empty remaining budget is `search_incomplete`.
- **G4.** `covers_infinite_time` false; leftover `#57` reused;
  `three_d_claim` false.
- **G5.** `navier_stokes_ab_architecture_ledger` stays `CONDITIONAL`;
  do not delete `"[0, infinity) is not a finite union of CI slabs"`.

## 9. Benchmark plan

`benchmarks/unforced_abc_long_chain.py` writes
`docs/benchmarks/unforced_abc_long_chain_smoke.json`. Algebra and
honesty run in CI. `--full` writes under `$OMNIBIAS_SCRATCH` (default
`artifacts/`).

## 10. Honesty and scope

Not Clay (A)/(B). Four finite slabs are not `[0, ∞)`. Not all data.
Not a bridge theorem. Forbidden: `navier_stokes_proof_claim`,
`continuum_navier_stokes_claim`, `forced_blowup_reproof_claim`.
`theorem_prover_verified` / `mathlib_verified` stay false unless a
genuine `lake build`.

## 11. Open questions and risks

- A uniform unforced 3-D majorant stays leftover `#57`.
- Falsifier: four decaying slabs rejected, growing vorticity accepted,
  `covers_infinite_time` true, or leftover `#57` dropped.

## 12. Implementation checklist

- [x] `locked_n_abc_slab_continuation(n_slabs=4)`
- [x] Leftover `#57` reused
- [x] Tests + smoke JSON
- [x] Docs page extension and mkdocs nav
- [x] CI job
- [x] Index row in `theory/README.md`

## 13. Parent problem and the exact reason it stays an external obligation

**Parent: Navier-Stokes unforced regularity (Clay A/B)**, open. This
fragment continues one named 3-D ABC plant across four finite
horizons, or halts. A finite union of CI slabs is not `[0, ∞)`, not
all smooth data, not the Clay 3-D quantifier, and not a bridge
theorem. The parent-level honesty flag stays false.
