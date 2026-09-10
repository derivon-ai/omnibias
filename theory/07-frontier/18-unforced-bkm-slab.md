# 07-18 Unforced BKM slab

## 1. Thesis and status

One periodic box, one horizon, `f = 0`. The exact 2-D Taylor–Green
vortex has `||ω||_∞(t) = 2 |A| e^{-2ν t}`. The BKM time integral of
that quantity is an outward-rounded `Interval`. The 07-02 weak residual
covers the same box. That is a force-free slab, not Clay (A)/(B).

- **Status**: shipped (G1–G5 CI; founding bias collapse, not temperature collapse; not Clay A/B)
- **Depends on**: 07-02, 07-08
- **Blocks**: 07-19, 07-21

## 2. Where it lands

`omnibias.pinn.certified.unforced` in `omnibias-pinn`. No new package.
Same domain and audience as 07-02. Do not grow
`omnibias.pinn.certified.navier_stokes`. Do not import that module.

## 3. Prior art in omnibias

- `omnibias.pinn.certified.fluid_fixtures.taylor_green_vortex` — exact
  2-D decaying TG, `forcing` identically zero, `forced: False`.
- `omnibias.pinn.certified.weak_form.certified_weak_residual` /
  `enclosure_covers` — 07-02 box `[0, 2π]²`, horizon `0.5`.
- `omnibias.core.verified.interval.Interval` plus
  `omnibias.core.verified.transcend.exp_iv`.
- `navier_stokes_ab_architecture_ledger` — finite architecture
  arithmetic with nonempty `NS_AB_EXTERNAL_PREMISES`.

**Confirmed gap.** Unforced regularity was not well-typed as a
force-free enclosed BKM slab. C/D construction ledgers stay a different
parent.

## 4. Mathematics

Taylor–Green on the torus: `u = A F (sin x cos y, -cos x sin y)`,
`F = e^{-2ν t}`, `f = 0`. Vorticity
`ω = ∂x u_y - ∂y u_x = 2 A F sin x sin y`, so
`||ω||_∞(t) = 2 |A| F`. For `ν > 0`

```
∫_0^H ||ω||_∞ dt = |A|/ν (1 - e^{-2ν H}).
```

The enclosure is that closed form with outward-rounded `exp`. The weak
residual of the exact vortex is `{0}` up to the 07-02 width report.
Founding bias-collapse jets on the weak form, not temperature collapse.

## 5. Worked example

Locked `A = 1`, `ν = 1/10`, `H = 1/2`. The integral is
`10 (1 - e^{-1/10}) ≈ 0.951626`, strictly below the continuation
budget `1`. Fixture force is the zero array. Weak-form coverage
misses are `0`.

## 6. Proposed API

```
locked_force_free_slab() -> UnforcedSlab
force_free_bkm_slab() -> report
enclose_bkm_integral(slab) -> Interval
honesty_payload() -> dict
```

Catalog kind `unforced_bkm_slab`, parent
`"Navier-Stokes unforced regularity (Clay A/B)"`,
`parent_status="open"`, `navier_stokes_proof_claim: False`.

## 7. Practical use cases

- Name a force-free plant before any continuation machine.
- Keep C/D exponent premises on the construction spine.
- Feed 07-19 a slab whose BKM integral is a sound `Interval`.

## 8. Acceptance gates

- **G1.** Fixture force is the zero array; plant is unforced.
- **G2.** Weak-form certificate covers the box (`enclosure_covers`
  misses `0`; residual enclosure contains `0`).
- **G3.** BKM integral enclosure is bounded, contains the exact value,
  and contains a grid + random sample of the integrand.
- **G4.** Honesty: parent flags false; `three_d_claim` false;
  `infinite_time_leftover` / `all_data_leftover` /
  `bridge_theorem_leftover` / `unforced_majorant_leftover` true.
- **G5.** `navier_stokes_ab_architecture_ledger` stays `CONDITIONAL`
  with nonempty premises.

## 9. Benchmark plan

`benchmarks/unforced_bkm_slab.py` writes
`docs/benchmarks/unforced_bkm_slab_smoke.json`. Algebra and honesty
run in CI. `--full` writes under `$OMNIBIAS_SCRATCH` (default
`artifacts/`).

## 10. Honesty and scope

Not Clay (A)/(B). Not 3-D. Not all data. Not `[0, ∞)`. Not a bridge
theorem. Not a uniform unforced 3-D vorticity majorant. Forbidden:
`navier_stokes_proof_claim`, `continuum_navier_stokes_claim`,
`forced_blowup_reproof_claim`. `theorem_prover_verified` /
`mathlib_verified` stay false unless a genuine `lake build`.

## 11. Open questions and risks

- The missing bridge (finite enclosed checks imply A/B) stays leftover.
- A 3-D unforced majorant is not this slab.
- Falsifier: nonzero forcing, a BKM enclosure that misses the exact
  integral, or a stamped parent flag.

## 12. Implementation checklist

- [x] `omnibias.pinn.certified.unforced` force-free TG slab
- [x] Interval BKM integral + 07-02 weak residual
- [x] Tests + smoke JSON
- [x] Docs page and mkdocs nav entry
- [x] CI job
- [x] Index row in `theory/README.md`

## 13. Parent problem and the exact reason it stays an external obligation

**Parent: Navier-Stokes unforced regularity (Clay A/B)**, open. This
fragment encloses one force-free 2-D Taylor–Green slab. A finite slab
does not imply global regularity, all smooth data, three-dimensional
unforced Navier-Stokes, or a bridge theorem. The parent-level honesty
flag stays false. C/D stays a different parent.
