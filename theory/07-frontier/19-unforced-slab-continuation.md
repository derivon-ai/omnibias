# 07-19 Unforced slab continuation

## 1. Thesis and status

A continuation machine on 07-18 slabs: `try_continue_slab` accepts the
next force-free Taylor–Green window when the enclosed BKM integral is
strictly below a named rational threshold, and otherwise returns
`Halt`. Two locked slabs `[0, 1/2]` then `[1/2, 1]` succeed on decaying
TG. Growing vorticity is `BLOCKED`. Empty remaining budget is
`search_incomplete`. Finite `n_slabs` does not cover `[0, ∞)`.

- **Status**: shipped (G1–G5 CI; founding bias collapse, not temperature collapse; leftover #57; not Clay A/B)
- **Depends on**: 07-18
- **Blocks**: none

## 2. Where it lands

`omnibias.pinn.certified.unforced` (`try_continue_slab`,
`locked_two_slab_continuation`). No new package. Do not grow
`omnibias.pinn.certified.navier_stokes`. Do not couple 07-06 as an NS
field.

## 3. Prior art in omnibias

- 07-18 `force_free_bkm_slab` / `enclose_bkm_integral`.
- `omnibias.core.verified.lohner.lohner_step` — available for a 1-D
  decay ODE; this pass uses the exact TG exponential instead.
- Discovery `budget == 0` → `search_incomplete` (not a parent theorem).

**Confirmed gap.** There was no accept-or-halt continuation on a
force-free enclosed BKM integral.

## 4. Mathematics

On decaying TG, `||ω||_∞` is decreasing. The integral on
`[t0, t0+H]` is `|A|/ν (e^{-2ν t0} - e^{-2ν (t0+H)})`. If that
enclosure lies strictly below `BKM_BUDGET = 1`, emit the next window
of length `H`. A manufactured growing law `||ω||_∞(t) = 2 |A| e^{+2ν t}`
has integral `|A|/ν (e^{2ν (t0+H)} - e^{2ν t0})`, which exceeds the
budget on the locked slab and returns `BLOCKED`. An empty remaining
budget is `search_incomplete` and never flips
`navier_stokes_proof_claim`.

Any finite `n_slabs` is a finite union. `[0, ∞)` is not that union.
Leftover `#57` records infinite time, all data, 3-D, and the bridge
theorem as one leftover object.

## 5. Worked example

Locked decaying slabs: `[0, 1/2]` integral `≈ 0.9516 < 1` → `Continue`
with `[1/2, 1]`. Second-slab integral `≈ 0.861 < 1` as arithmetic,
but remaining budget `0` after the first accept is `search_incomplete`.
Growing TG on `[0, 1/2]` integral `≈ 1.0517 > 1` → `Halt` /
`BLOCKED` / `growing_vorticity`.

## 6. Proposed API

```
try_continue_slab(slab, *, remaining_budget, next_horizon) -> Continue | Halt
locked_two_slab_continuation() -> report
UNFORCED_CONTINUATION_LEFTOVER  # leftover_id 57
```

Catalog kind `unforced_slab_continuation`, parent
`"Navier-Stokes unforced regularity (Clay A/B)"`,
`parent_status="open"`.

## 7. Practical use cases

- Accept a decaying TG horizon without forging A/B.
- Halt on growing vorticity with a named reason.
- Keep empty budget as `search_incomplete`, never a parent flag.

## 8. Acceptance gates

- **G1.** Two locked decaying TG slabs `[0, 1/2]` then `[1/2, 1]`
  accept.
- **G2.** Manufactured growing vorticity returns `Halt` / `BLOCKED`
  with named reason `growing_vorticity`.
- **G3.** Empty remaining budget is `search_incomplete`; honesty
  parent flags stay false.
- **G4.** Leftover `#57` records infinite-time / all-data / 3-D /
  bridge as one object; `covers_infinite_time` is false.
- **G5.** `navier_stokes_ab_architecture_ledger` stays `CONDITIONAL`
  with nonempty premises.

## 9. Benchmark plan

`benchmarks/unforced_slab_continuation.py` writes
`docs/benchmarks/unforced_slab_continuation_smoke.json`. Algebra and
honesty run in CI. `--full` writes under `$OMNIBIAS_SCRATCH` (default
`artifacts/`).

## 10. Honesty and scope

Not Clay (A)/(B). Not a published reduction “these slabs ⇒ A/B”.
Not a 3-D unforced majorant. Not exhausting initial-data space.
Forbidden: `navier_stokes_proof_claim`,
`continuum_navier_stokes_claim`, `forced_blowup_reproof_claim`.
`theorem_prover_verified` / `mathlib_verified` stay false unless a
genuine `lake build`.

## 11. Open questions and risks

- A non-exploding 3-D majorant stays leftover.
- Falsifier: decaying TG rejected, growing vorticity accepted, or
  leftover `#57` dropped.

## 12. Implementation checklist

- [x] `try_continue_slab` accept / Halt
- [x] Leftover `#57`
- [x] Tests + smoke JSON
- [x] Docs page (continuation section) and mkdocs nav entry
- [x] CI job
- [x] Index row in `theory/README.md`

## 13. Parent problem and the exact reason it stays an external obligation

**Parent: Navier-Stokes unforced regularity (Clay A/B)**, open. This
fragment continues one named 2-D Taylor–Green plant across two finite
horizons, or halts. A finite union of CI slabs is not `[0, ∞)`, not
all smooth data, not three-dimensional unforced Navier-Stokes, and not
a bridge theorem. The parent-level honesty flag stays false.
