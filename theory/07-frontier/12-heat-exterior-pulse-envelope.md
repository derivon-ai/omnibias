# 07-12 Heat exterior identity and pulse envelope

## 1. Thesis and status

The exterior swirl `K = r^{-1-2h} H_ext(τ/r²)` solves the radial
`m = 1` heat equation exactly. That is D-finite in the similarity
variable, not a PINN residual. The pulse envelope `P(v)` grows then
viscously decays; if `P` is an activation, `P'` is closed-form.

- **Status**: shipped (G1–G6 CI; founding bias collapse, not temperature collapse; `h = 0` is the CI identity; paper `h > 0` leftover-recorded; not a forced-blowup reproof)
- **Depends on**: 01-05, 09-12
- **Blocks**: none

## 2. Where it lands

Heat checker: `omnibias.core.verified.swirl_heat` in `omnibias-core`.
Holonomic identity: `omnibias.holonomic.swirl_heat`, following the
Jacobian Case A pattern (`named_*_generators`, `prove("identity")`).
Pulse: `omnibias.core.pulse_envelope` using the existing sigmoid
tower. No new package. Do not claim a 3-D heat theorem.

## 3. Prior art in omnibias

- `omnibias.core.collapse.identity` / `prove("identity")`
- `omnibias.holonomic.jacobian_n2_case_a` — named generators plus
  identity payloads.
- `omnibias.core.mollifier.tail_bound` (01-05)
- `omnibias.core.verified.coeffs.sigmoid_poly_coeffs_exact`

**Confirmed gap.** The radial `m = 1` swirl-heat identity and an
exact-`D` pulse envelope were not locked.

## 4. Mathematics

For `h = 0`, `H ≡ 1`, `K = 1/r`. Then
`(∂_r² + r^{-1}∂_r - r^{-2}) K = 0` and `∂_τ K = 0`. In
`ξ = 1/r`, `K = ξ` is annihilated by `-ξ d/dξ + 1`. The paper uses
`h > 0`; that scaling leaves the integer-power `Q` fragment and is
leftover-recorded.

Pulse: `P = s t` with `s = σ(v)`, `t = σ(L-v)`,
`P' = s(1-s)t - s t(1-t)` from the Riccati identity, matching
`P_1` of the sigmoid tower. Cutoff tails at `τ = 0` stay an external
premise; the certified tail is `mollifier.tail_bound`.

Founding bias-collapse arithmetic, not temperature collapse.

## 5. Worked example

`r = 1`, `τ = 1`, `h = 0`: residual `{0}`. Identity payload
`left = [0]`, `right = [0]` proves via `prove("identity")`. Growth
half `(s, t) = (1/4, 3/4)` has `P' = (1/2) P`; decay
`(3/4, 1/4)` has `P' = (-1/2) P`. Logistic mollifier tail at
half-width `3` contains `true_outside_mass`.

## 6. Proposed API

```
swirl_heat_residual(r, tau, *, h=0)
prove_swirl_heat_identity()
locked_growth_envelope() / locked_decay_envelope()
mollifier_tail_contains_truth()
```

## 7. Practical use cases

- Lock the radial operator before any anisotropic scaling.
- Match pulse `P'` to the tower instead of a finite difference.
- Bound cutoff tails with the existing 01-05 mollifier.

## 8. Acceptance gates

- **G1.** Swirl-heat residual is `{0}` (interval) at locked points.
- **G2.** Holonomic identity payload proves via `prove("identity")`.
- **G3.** No-toolchain degrades (Lean optional).
- **G4.** Pulse `P'` matches the tower, not FD.
- **G5.** Mollifier tail contains `true_outside_mass`.
- **G6.** Honesty.

## 9. Benchmark plan

`benchmarks/swirl_heat_pulse.py` writes
`docs/benchmarks/swirl_heat_pulse_smoke.json`.

## 10. Honesty and scope

Not a 3-D heat theorem. Not Osterwalder–Schrader. Not a continuum
mass gap. YM is untouched. Paper uses `h > 0`; the CI identity is
`h = 0`. Cutoff summation to a `C^∞_c` force stays external. Not a
Clay (C)/(D) reproof. Unforced (A)/(B) stays external.

## 11. Open questions and risks

- A D-finite plant at `h = 1/200` would need a different annihilator.
- Falsifier: residual at a locked rational point is nonzero.

## 12. Implementation checklist

- [x] `omnibias.core.verified.swirl_heat`
- [x] `omnibias.holonomic.swirl_heat`
- [x] `omnibias.core.pulse_envelope`
- [x] Tests + smoke JSON
- [x] Docs page and nav entry
- [x] Index row in `theory/README.md`

## 13. Parent problem and the exact reason it stays an external obligation

**Parent: Navier-Stokes forced blowup (Clay C/D)**, already true in the
world. A radial `m = 1` identity and a 1-D envelope derivative are
not a second proof and are not Osterwalder–Schrader. Unforced
Navier-Stokes regularity (Clay A/B) stays an external obligation.
The Yang-Mills mass gap is untouched.
