# 07-21 Unforced ABC slab

## 1. Thesis and status

One periodic 3-torus, one horizon `1/2`, `f = 0`. The plant is the
exact decaying 3-D ABC flow. Because ABC is Beltrami,
`‖ω‖_∞(t) = k ‖u‖_∞(t) = k F(t) ‖u_0‖_∞` with
`F = e^{-ν k² t}`. A sound Euclidean hull of the component bounds
encloses the BKM integral as an `Interval`. Two locked decaying slabs
accept under the 07-19 continuation budget; manufactured growth is
`Halt` / `BLOCKED`. That is one named 3-D exact plant, not Clay (A)/(B).

- **Status**: shipped (G1–G5 CI; founding bias collapse, not temperature collapse; leftover #57 reused; not Clay A/B)
- **Depends on**: 07-18
- **Blocks**: none

## 2. Where it lands

`omnibias.pinn.certified.unforced` in `omnibias-pinn`. No new package.
Same domain and audience as 07-18. Do not grow
`omnibias.pinn.certified.navier_stokes`. Do not import that module.

## 3. Prior art in omnibias

- `omnibias.pinn.certified.fluid_fixtures.beltrami_abc_flow` — exact
  decaying 3-D ABC, `forcing` identically zero, `forced: False`,
  Beltrami `∇×u = k u`.
- 07-18 / 07-19 `enclose_bkm_integral` / `try_continue_slab` on
  Taylor–Green; `BKM_BUDGET = 1`; leftover `#57`.
- `navier_stokes_ab_architecture_ledger` — nonempty
  `NS_AB_EXTERNAL_PREMISES`, including
  `"three-dimensional unforced NS, not 2-D Taylor-Green"`.

**Confirmed gap.** The force-free BKM slab was 2-D Taylor–Green only.

## 4. Mathematics

ABC on the 3-torus: `u = F u_0` with `F = e^{-ν k² t}` and

```
u_0 = (A sin kz + C cos ky, B sin kx + A cos kz, C sin ky + B cos kx).
```

Beltrami: `∇×u = k u`, so `‖ω‖_∞(t) = k ‖u‖_∞(t)`. A sound bound:

```
‖u_0‖_∞ ≤ √((|A|+|C|)² + (|B|+|A|)² + (|C|+|B|)²).
```

The decaying BKM integral is at most
`‖u_0‖_∞ / (ν k) (e^{-ν k² t0} - e^{-ν k² t1})`, enclosed by
outward-rounded `exp`. Manufactured growth uses a faster positive
exponent so the same budget returns `BLOCKED`. Founding bias-collapse
enclosures, not temperature collapse.

One exact ABC solution does not discharge the Clay quantifier “all
smooth finite-energy 3-D data”. The 3-D Taylor–Green premise stays.

## 5. Worked example

Locked `A = B = C = 1/2`, `k = 1`, `ν = 1/10`, `H = 1/2`. The
Euclidean hull is `√3`. The decaying integral is about `0.845 < 1`.
Two slabs `[0, 1/2]` then `[1/2, 1]` accept. Fixture force is the
zero array; `dimension = 3`. `three_d_claim` stays false;
`exact_3d_abc_plant` is true. Leftover `#57` is reused.

## 6. Proposed API

```
locked_force_free_abc_slab() -> AbcSlab
force_free_abc_bkm_slab() -> report
enclose_abc_bkm_integral(slab) -> Interval
try_continue_abc_slab(...) -> Continue | Halt
locked_two_abc_slab_continuation() -> report
```

Catalog kind `unforced_abc_slab`, parent
`"Navier-Stokes unforced regularity (Clay A/B)"`,
`parent_status="open"`, `navier_stokes_proof_claim: False`.

## 7. Practical use cases

- Climb from 2-D TG to one named 3-D force-free exact plant.
- Keep the Clay 3-D quantifier as an external premise.
- Reuse leftover `#57` rather than minting a second id for the same
  four A/B gaps.

## 8. Acceptance gates

- **G1.** Fixture is 3-D, `forced: False`, force array zero.
- **G2.** BKM integral enclosure bounded, contains a quadrature /
  grid+sample of the integrand.
- **G3.** Two locked decaying slabs accept under the same
  continuation budget as 07-19; manufactured growth still
  `Halt` / `BLOCKED`.
- **G4.** Honesty: `three_d_claim` false; `exact_3d_abc_plant` true;
  `all_data_leftover` / `infinite_time_leftover` /
  `bridge_theorem_leftover` / `unforced_majorant_leftover` true;
  leftover `#57` reused.
- **G5.** `navier_stokes_ab_architecture_ledger` stays `CONDITIONAL`;
  do not delete premise
  `"three-dimensional unforced NS, not 2-D Taylor-Green"`.

## 9. Benchmark plan

`benchmarks/unforced_abc_slab.py` writes
`docs/benchmarks/unforced_abc_slab_smoke.json`. Algebra and honesty
run in CI. `--full` writes under `$OMNIBIAS_SCRATCH` (default
`artifacts/`).

## 10. Honesty and scope

Not Clay (A)/(B). Not all data. Not `[0, ∞)`. Not a bridge theorem.
Not a uniform unforced 3-D vorticity majorant. One ABC solution is
not the Clay 3-D quantifier. Forbidden: `navier_stokes_proof_claim`,
`continuum_navier_stokes_claim`, `forced_blowup_reproof_claim`.
`theorem_prover_verified` / `mathlib_verified` stay false unless a
genuine `lake build`.

## 11. Open questions and risks

- A 3-D unforced majorant for all data stays leftover `#57`.
- Falsifier: nonzero forcing, a 2-D fixture, deleting the 3-D
  premise, stamping `three_d_claim`, or minting a second leftover id
  for the same four gaps.

## 12. Implementation checklist

- [x] `omnibias.pinn.certified.unforced` ABC BKM slab + continuation
- [x] Interval BKM integral on a sound `‖u_0‖_∞` hull
- [x] Tests + smoke JSON
- [x] Docs page extension and mkdocs nav
- [x] CI job
- [x] Index row in `theory/README.md`

## 13. Parent problem and the exact reason it stays an external obligation

**Parent: Navier-Stokes unforced regularity (Clay A/B)**, open. This
fragment encloses one force-free 3-D ABC plant on two finite
horizons. A named exact solution is not all smooth data, not
`[0, ∞)`, not a uniform 3-D vorticity majorant, and not a bridge
theorem. The parent-level honesty flag stays false.
