# 07-22 Unforced 3-D Taylor–Green IC

## 1. Thesis and status

Classical 3-D Taylor–Green is a force-free initial condition on the
3-torus, not an exact decaying Navier–Stokes solution. Exact curl at
`t = 0` has a sound Euclidean hull `|A| √6`. Continuation that reuses
the ABC / 2-D TG exponential is `Halt` / `BLOCKED` /
`three_d_tg_not_closed_form`. Vortex stretching has no `e^{-ν k² t}`
law.

- **Status**: shipped (G1–G5 CI; founding bias collapse, not temperature collapse; leftover #59 `three_d_tg_not_exact`; not Clay A/B)
- **Depends on**: 07-18, 07-21
- **Blocks**: none

## 2. Where it lands

`omnibias.pinn.certified.fluid_fixtures.taylor_green_vortex_3d` and
`omnibias.pinn.certified.unforced` in `omnibias-pinn`. No new package.
Do not grow `omnibias.pinn.certified.navier_stokes`. Do not import that
module. Kolmogorov is forced and is not this plant.

## 3. Prior art in omnibias

- `taylor_green_vortex` — exact 2-D decaying TG, `exact_solution: True`.
- `beltrami_abc_flow` — exact 3-D decaying Beltrami, `exact_solution: True`.
- 07-21 `try_continue_abc_slab` — ABC exponential BKM.

**Confirmed gap.** The 3-D TG IC was not named as a non-exact plant
with a Halt on fake decay.

## 4. Mathematics

On `[0, 2π)³` at `t = 0`

```
u = A (sin x cos y cos z, -cos x sin y cos z, 0)
ω = A (-cos x sin y sin z, -sin x cos y sin z, 2 sin x sin y cos z).
```

Component hulls `|ω_x| ≤ |A|`, `|ω_y| ≤ |A|`, `|ω_z| ≤ 2|A|`, so
`‖ω₀‖_∞ ≤ |A| √6`. The field is divergence-free. The nonlinear term
is not a pure gradient: this is not Beltrami, not a Stokes eigenfunction
with a closed-form viscous decay. Founding bias-collapse arithmetic
on the `t = 0` hull. Not temperature collapse. Not a continuum NS
theorem.

## 5. Worked example

Locked `A = 1`, `n = 16`. Fixture `dimension = 3`, force zero,
`exact_solution` false. The `√6` hull contains a grid and a random
sample of `|ω|`. `try_continue_tg3d_slab` with remaining budget `1`
returns `Halt` / `BLOCKED` / `three_d_tg_not_closed_form`. Leftover
`#59`. Parent flags stay false.

## 6. Proposed API

```
taylor_green_vortex_3d(n, *, viscosity, amplitude=1, time=0) -> sample
enclose_tg3d_omega0(amplitude) -> Interval
tg3d_omega0_contains_grid_and_sample(sample) -> bool
try_continue_tg3d_slab(...) -> Halt
force_free_tg3d_ic() -> report
THREE_D_TG_NOT_EXACT_LEFTOVER  # leftover_id 59
```

Catalog kind `unforced_tg3d_ic`, parent
`"Navier-Stokes unforced regularity (Clay A/B)"`,
`parent_status="open"`.

## 7. Practical use cases

- Name the 3-D TG temptation without treating it as ABC.
- Keep the 2-D TG / ABC exact plants distinct from this IC.
- Refuse a fake exponential majorant.

## 8. Acceptance gates

- **G1.** Fixture is 3-D, force array zero, `exact_solution` false.
- **G2.** `t = 0` vorticity enclosure contains a grid + sample of `|ω|`.
- **G3.** Continuation with the 07-19 budget is `Halt` / `BLOCKED` /
  `three_d_tg_not_closed_form`.
- **G4.** Honesty: parent flags false; `three_d_claim` false;
  leftover `#59` `three_d_tg_not_exact`; leftover `#57` untouched.
- **G5.** Catalog `unforced_tg3d_ic`, parent A/B, `open`. The A/B
  3-D premise stays.

## 9. Benchmark plan

`benchmarks/unforced_tg3d_ic.py` writes
`docs/benchmarks/unforced_tg3d_ic_smoke.json`. Algebra and honesty
run in CI. `--full` writes under `$OMNIBIAS_SCRATCH` (default
`artifacts/`).

## 10. Honesty and scope

Not Clay (A)/(B). Not an exact 3-D decaying plant. Not all data.
Not `[0, ∞)`. Not a bridge theorem. Forbidden:
`navier_stokes_proof_claim`, `continuum_navier_stokes_claim`,
`forced_blowup_reproof_claim`. `theorem_prover_verified` /
`mathlib_verified` stay false unless a genuine `lake build`.

## 11. Open questions and risks

- Ethier–Steinman would be a later exact 3-D decaying plant, not this
  IC.
- Falsifier: `exact_solution` true, a decaying exponential BKM, or
  leftover `#59` dropped.

## 12. Implementation checklist

- [x] `taylor_green_vortex_3d` fixture
- [x] `t = 0` ω hull + Halt continuation
- [x] Tests + smoke JSON
- [x] Docs page extension and mkdocs nav
- [x] CI job
- [x] Index row in `theory/README.md`

## 13. Parent problem and the exact reason it stays an external obligation

**Parent: Navier-Stokes unforced regularity (Clay A/B)**, open. This
fragment names one force-free 3-D initial condition whose evolution
is not closed form. An IC is not all smooth data, not `[0, ∞)`, not
a uniform 3-D vorticity majorant, and not a bridge theorem. The
parent-level honesty flag stays false.
