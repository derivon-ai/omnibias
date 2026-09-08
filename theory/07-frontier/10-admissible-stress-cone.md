# 07-10 Admissible stress cone and scale ledger

## 1. Thesis and status

Lemma 4.5 / Appendix C is finite linear algebra: `T` is in the interior
of `cone(v1, v2)` iff `det(v1, v2) ≠ 0` and both Cramer weights are
strictly positive. Separately, the paper has rational constraints that
07-08 does not record: `0 < h < 1/100`, energy `h < 1/6`,
`κ_s ≤ 10^{-5}`. Those are a second `ConvergenceLedger`, not a new
kind.

- **Status**: shipped (G1–G5 CI; founding bias collapse, not temperature collapse; not Clay (A)/(B); not a forced-blowup reproof)
- **Depends on**: 07-08, 01-11
- **Blocks**: none

## 2. Where it lands

`omnibias.core.proof.obligations.stress_cone` beside the stencil and
ledger obligations, in `omnibias-core`. The scale ledger is
`navier_stokes_scale_ledger()` in
`omnibias.core.proof.obligations.convergence_ledger`. Mathlib lemmas
live in `Check/StressCone.lean` and `Check/ConvergenceLedger.lean`.
No new package.

## 3. Prior art in omnibias

- `omnibias.core.proof.obligations.convergence_ledger` — stage-indexed
  affine budgets; 07-08 NS exponent / YM polymer instances.
- `omnibias.core.proof.lean_check` — `allRatLt`.
- `omnibias.core.proof.inequality` — not an LP solver; this cone is 2×2.

**Confirmed gap.** Exact cone membership and the `(h, κ_s)` scale
ledger were not recorded.

## 4. Mathematics

```
λ1 = det(T, v2) / det(v1, v2)
λ2 = det(v1, T) / det(v1, v2)
```

Interior is `λi > 0`; closed cone is `λi ≥ 0`. `det = 0` is
`BLOCKED` and named. Scale ledger side conditions are affine in the
parameters; the binding upper bound on `h` is the stricter of `1/100`
and `1/6`. Founding bias-collapse arithmetic, not temperature collapse.

## 5. Worked example

`v1 = e1`, `v2 = e2`, `T = (1, 1)` gives `λi = 1`. Parallel generators
`(1,0)`, `(2,0)` are `degenerate_generators`. Opposite `T = (-1,-1)` is
`opposite_cone`. Scale ledger at `h = 1/200`, `κ_s = 1/100000`
discharges; binding is `h < 1/100`.

## 6. Proposed API

```
check_cone(query) -> ConeReport
seal_cone_certificate(query, *, run_lean=True)
locked_interior_cone()
navier_stokes_scale_ledger()
```

No LP solver. Kernel emission reuses `allRatLt`.

## 7. Practical use cases

- Independently re-derive Appendix C on a locked pair.
- Print unmet analytic premises of the scale constraints.
- Feed a discharged cone into the same seal / replay path as 07-08.

## 8. Acceptance gates

- **G1.** Locked cone discharges with exact `λi`.
- **G2.** Parallel / opposite cases `BLOCKED` and named.
- **G3.** Scale ledger discharges; binding `h < 1/100`.
- **G4.** Seal / digest / replay; no parent flag.
- **G5.** Kernel `allRatLt` on a discharged cone.

## 9. Benchmark plan

`benchmarks/stress_cone.py` writes
`docs/benchmarks/stress_cone_smoke.json`. The existing
`benchmarks/convergence_ledger.py` iterates
`curated_convergence_ledgers()` and therefore covers the scale ledger.

## 10. Honesty and scope

`external_premises` stay nonempty (analytic classes, pulse
construction, PDE estimates). Parent flags remain false. Not a Clay
(C)/(D) reproof. Unforced (A)/(B) stays external.

## 11. Open questions and risks

- Higher-dimensional cones stay out of fragment.
- Falsifier: Python `λi` disagrees with the Lean instance.

## 12. Implementation checklist

- [x] `omnibias.core.proof.obligations.stress_cone`
- [x] `navier_stokes_scale_ledger`
- [x] Kernel `allRatLt` + Mathlib `Check/StressCone.lean`
- [x] Tests + smokes
- [x] Docs page and nav entry
- [x] Index row in `theory/README.md`

## 13. Parent problem and the exact reason it stays an external obligation

**Parent: Navier-Stokes forced blowup (Clay C/D)**, already true in the
world. A 2×2 cone membership and a scale budget are not a second proof.
Unforced Navier-Stokes regularity (Clay A/B) stays an external
obligation. Parent flags stay false because `external_premises` is
nonempty and is never stamped by hand.
