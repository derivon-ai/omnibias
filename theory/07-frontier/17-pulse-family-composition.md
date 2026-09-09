# 07-17 Pulse-family composition

## 1. Thesis and status

A locked occupancy pulse `P = s t` from 07-12 composes into the 07-13
forced field: `P'` matches the sigmoid tower on a finite rational grid,
the corrected axis jet times `P` is `{0}`, and `P` times the
uncorrected jet is enclosed (grid + random sample). That is one named
pulse family, not cutoff summation and not joining.

- **Status**: shipped (G1–G5 CI; founding bias collapse, not temperature collapse; not a forced-blowup reproof)
- **Depends on**: 07-12, 07-13
- **Blocks**: none

## 2. Where it lands

`omnibias.core.pulse_envelope` (grid + Interval enclosure) and
`omnibias.pinn.certified.forced_flat` (`compose_locked_pulse_family`).
No new package: same domain and audience as 07-12 / 07-13. Do not grow
`omnibias.pinn.certified.navier_stokes`.

## 3. Prior art in omnibias

- `omnibias.core.pulse_envelope` — `P = s t`, exact `P'` from the
  Riccati / tower identity; `cutoff_tail_external` stays true.
- `omnibias.core.mollifier.tail_bound` (01-05).
- `omnibias.pinn.certified.forced_flat` — axis-flat `T_0`, core energy,
  mollifier cutoff, from-rest ramp. `pulses_leftover` stays true on the
  07-13 honesty payload (full OpenAI pulse cycle).

**Confirmed gap.** The locked pulse was not composed into the forced
field as a named identity with a grid-and-sample enclosure.

## 4. Mathematics

Occupancy `s = σ(v)`, `t = σ(L-v)` with locked `L = 2`.
`P = s t`, `P' = s(1-s)t - s t(1-t)` from `σ' = σ(1-σ)`, matching
`P_1` of the sigmoid tower. The axis jet `T_0` of 07-13 is independent
of occupancy, so `(P T_0)' = P' T_0`. On the corrected profile
`T_0 = (0,0)` and `P T_0 = 0`. On the uncorrected plant
`T_{rθ} = 599/400`; `P T_{rθ}` is enclosed by the product of the
occupancy box `[1/8, 7/8]^2` with that rational. Growth / decay rates
`±1/2` compose as `P' T = (±1/2) P T`.

Founding bias-collapse arithmetic, not temperature collapse.

## 5. Worked example

Growth `(s,t) = (1/4, 3/4)`: `P = 3/16`, `P' = 3/32 = (1/2) P`.
Corrected `P T_0 = (0,0)`. Uncorrected
`P T_{rθ} = (3/16)(599/400) = 1797/6400`. The box enclosure of `P`
contains every locked-grid value and a seeded random sample.

## 6. Proposed API

```
locked_pulse_grid() -> tuple[PulseEnvelope, ...]
locked_pulse_grid_matches_tower() -> bool
enclose_pulse_on_box() -> Interval
pulse_product_contains_grid_and_sample(coeff) -> bool
compose_locked_pulse_family() -> report
```

Honesty on the 07-17 report: `pulses_leftover=False` only if the
identity holds; `joining_leftover`, `uniqueness_leftover`, and
`c_infinity_through_t1_leftover` stay true. Parent flags stay false.
Catalog kind `pulse_family_composition`, parent
`"Navier-Stokes forced blowup (Clay C/D)"`,
`parent_status="already_true"`.

## 7. Practical use cases

- Lock one pulse family before any WKB PDE or cutoff summation.
- Keep the 07-13 leftover list honest: the full pulse cycle stays
  leftover on `honesty_payload()`.
- Feed a later joining / moment table a named occupancy multiplier.

## 8. Acceptance gates

- **G1.** `P'` matches the tower on the locked rational occupancy grid.
- **G2.** Corrected `P T_0 = (0,0)` over `Q`; growth / decay product
  rule `P' T = (±1/2) P T` on the uncorrected jet.
- **G3.** Interval enclosure of `P T_{rθ}` (uncorrected) contains the
  locked grid and a random sample.
- **G4.** Honesty: parent flags false; 07-17 `pulses_leftover` false
  iff the identity holds; joining / uniqueness / `C^infty` through
  `t=1` stay leftover.
- **G5.** `NS_SCALE_EXTERNAL_PREMISES` still names
  `"construction of each pulse family and the cutoff summation"`.

Leftover-record `#56` only if the composition identity fails.

## 9. Benchmark plan

`benchmarks/pulse_family_composition.py` writes
`docs/benchmarks/pulse_family_composition_smoke.json`. Algebra and
honesty run in CI. `--full` writes under `$OMNIBIAS_SCRATCH` (default
`artifacts/`).

## 10. Honesty and scope

Not Clay (C)/(D). Not a second proof of the OpenAI theorem. One locked
pulse is not each pulse family plus cutoff summation. Unforced (A)/(B)
stays external. `navier_stokes_proof_claim` and
`forced_blowup_reproof_claim` stay false. Do not trim scale-ledger
premises.

## 11. Open questions and risks

- Cutoff summation over many pulses stays external.
- Joining / moment matching stays leftover.
- Falsifier: a locked-grid `P'` disagrees with the tower, or the
  enclosure misses a sample.

## 12. Implementation checklist

- [x] `omnibias.core.pulse_envelope` grid + enclosure
- [x] `compose_locked_pulse_family` on `forced_flat`
- [x] Tests + smoke JSON
- [x] Docs page and mkdocs nav entry
- [x] CI job
- [x] Index row in `theory/README.md`

## 13. Parent problem and the exact reason it stays an external obligation

**Parent: Navier-Stokes forced blowup (Clay C/D)**, already true in the
world. This fragment composes one locked occupancy pulse into the
07-13 field. It does not claim a second proof. Unforced Navier-Stokes
regularity (Clay A/B) stays an external obligation. The parent-level
honesty flag stays false.
