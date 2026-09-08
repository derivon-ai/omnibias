# 07-11 Weighted coefficient class (finite-order)

## 1. Thesis and status

Definitions 6.4–6.5 are seminorms `W_α`, `M_α`, `S_α` with weights
`ζ`, `δ`, envelope `P(v)`, and `ε = Q^h`.
`omnibias.core.verified.sequence_space` is geometric / analytic
(`ν^k`), not Gevrey (`k!^s`). The honest primitive is a **fixed-order**
weighted bound plus an optional finite-order Gevrey majorant.
Infinite-class membership stays out of Lean.

- **Status**: shipped (G1–G5 CI; founding bias collapse, not temperature collapse; not a Gevrey theorem; not a forced-blowup reproof)
- **Depends on**: 07-08
- **Blocks**: none

## 2. Where it lands

`omnibias.core.verified.weighted_class` in `omnibias-core`, next to
`sequence_space`. No new package: same domain, dependency tier, and
audience as the verified sequence-space primitive.

## 3. Prior art in omnibias

- `omnibias.core.verified.sequence_space.ValidatedSeries` — geometric
  `ν^k` tails. Kept; not replaced.
- `omnibias.core.verified.interval.Interval`

**Confirmed gap.** No fixed-order `W/M/S` bound and no finite-order
Gevrey majorant that refuses `k > k_max`.

## 4. Mathematics

A `WeightedBound` is `C ε^α` times a stored envelope factor, with
integer `α` so the comparison stays in `Q`.
`gevrey_majorant` checks `|a_k| ≤ C k!^s ρ^k` only for `k ≤ k_max`
and integer `s`. A longer coefficient list is `out-of-fragment`.
Founding bias-collapse arithmetic, not temperature collapse.

## 5. Worked example

Locked samples `{1/2, 1/3, 2/5}` against `C = 1`, `α = 0`, `ε = 1`
all lie in the bound. A sample `2` is named `too_large`. Geometric
coefficients `(1, 1/2, 1/4)` satisfy the majorant at `s = 0`,
`ρ = 1/2`, `k_max = 2`; a fourth coefficient is refused.

## 6. Proposed API

```
EdgeWeights(zeta, delta)
PulseWeight(L_s, coeffs)
WeightedBound(alpha, eps, S_star, C, b, d)
check_pointwise_bound(samples, bound) -> BoundReport
gevrey_majorant(coeffs, s, rho, k_max)
```

## 7. Practical use cases

- Lock a hand-checkable envelope before any WKB pulse PDE.
- Name a violating sample instead of returning a silent false.
- Keep the geometric `ValidatedSeries` path for analytic data.

## 8. Acceptance gates

- **G1.** Locked samples lie in the enclosure.
- **G2.** A violating sample is named.
- **G3.** `gevrey_majorant` holds up to `k_max` and refuses `k > k_max`.
- **G4.** Geometric `ValidatedSeries` path still works.
- **G5.** Honesty: no continuum Gevrey / no NS class-membership claim.

## 9. Benchmark plan

`benchmarks/weighted_class.py` writes
`docs/benchmarks/weighted_class_smoke.json`.

## 10. Honesty and scope

Not a Gevrey-class theorem. Not a continuum NS class-membership claim.
Not a Clay (C)/(D) reproof. Unforced (A)/(B) stays external.

## 11. Open questions and risks

- Non-integer `s` stays out of fragment.
- Falsifier: a locked sample is excluded by outward rounding. Grid +
  random coverage is required.

## 12. Implementation checklist

- [x] `omnibias.core.verified.weighted_class`
- [x] `packages/omnibias-core/tests/test_weighted_class.py`
- [x] Smoke JSON
- [x] Docs page and nav entry
- [x] Index row in `theory/README.md`

## 13. Parent problem and the exact reason it stays an external obligation

**Parent: Navier-Stokes forced blowup (Clay C/D)**, already true in the
world. A finite-order seminorm bound is not membership of an infinite
Gevrey class and is not a second proof. Unforced Navier-Stokes
regularity (Clay A/B) stays an external obligation.
