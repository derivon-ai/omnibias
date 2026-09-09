# 07-14 Similarity profile search

## 1. Thesis and status

07-09 evaluates Lemma 4.1 on a **locked** polynomial. Its leading
residual is the independent plant `R = r^2`, so a discovery loop
cannot find a profile. This spec ships a **finite family of
axis-regular jets** whose score is the real similarity residual
`T_b(F) - 1` and whose accept/reject includes the 07-10 cone.

- **Status**: shipped (G1–G5 CI; founding bias collapse, not temperature collapse; not a forced-blowup reproof)
- **Depends on**: 07-09, 07-10
- **Blocks**: 07-15

## 2. Where it lands

`omnibias.pinn.certified.anisotropic` (pure `Fraction` / `Interval`
family + check) and `omnibias.pinn.jax.discovery.ns_core` (the
`run_discovery` harness). A `ProblemAdapter` lives on
`omnibias.pinn.jax.discovery.pipeline`. No new package: same domain
and audience as 07-09. Do not grow
`omnibias.pinn.certified.navier_stokes`.

## 3. Prior art in omnibias

- `omnibias.pinn.certified.anisotropic` — locked `F = c(1 + a X)`,
  Lemma 4.1, axis germ `{0}`. Residual plant is `R = r^2`.
- `omnibias.pinn.jax.discovery.euler3d_axisym` — consumes that locked
  residual; cannot discover a profile.
- `omnibias.core.proof.obligations.stress_cone.check_cone` — exact
  2x2 interior-cone membership.
- `omnibias.core.proof.discovery.run_discovery` — finite family +
  `ExactCheck`.

**Confirmed gap.** A finite family whose score is the profile PDE
residual, not `R = r^2`.

## 4. Mathematics

`F = c (1 + a X + b X^2)` so `E / sqrt(2X)` stays polynomial and the
axis TM remainder is `{0}`. `U = eta X (1 + a X + b X^2)`; `Pi` is
the exact antiderivative of `F^2`; `V_0 = -X U_X`. At the named
point `(X, eta, operator-b) = (1, 0, 0)`,

```
T_0 F = X F_X / L = c (a + 2b).
```

The profile residual is `T_0 F - 1`. Implied stress for the cone is
the coefficient pair `T = (c, a)` against `e1, e2`. Origin
`(c, a, b) = (1, 1, 0)` still discharges. A second witness
`(1, 3, -1)` also has `T_0 F = 1` and interior cone. Opposite
`(-1, -1, 0)` is named `BLOCKED`. This is founding **bias-collapse**
arithmetic. It is not temperature collapse.

## 5. Worked example

Origin `(1, 1, 0)`: `F = 1 + X`, `T_0 F = 1`, cone `(1, 1)`
interior. Neighbor `(1, 3, -1)`: `F = 1 + 3X - X^2`,
`F_X(1) = 1`, `T_0 F = 1`, cone `(1, 3)` interior. Opposite
`(-1, -1, 0)`: residual `0` but `check_cone` returns
`opposite_cone`. `budget == 0` is `search_incomplete`.

## 6. Proposed API

```
NSCoreProfileFamily
check_ns_core_candidate((c, a, b)) -> ExactCheck | None
run_ns_core_discovery(budget=..., collect=...)
NSCoreAdapter
```

Catalog kind `ns_core_profile_search`, mode `exact_search`. Honesty:
`forced_blowup_reproof_claim=False`, `navier_stokes_proof_claim=False`.
No torch/jax in the family module.

## 7. Practical use cases

- Search a named jet box instead of scoring `R = r^2`.
- Reject cone-opposite coefficients by name.
- Feed a second witness into later joining (joining stays external).
- Pipeline-stage the exact-Q search beside IPM / Boussinesq CAP.

## 8. Acceptance gates

- **G1.** Locked origin still discharges (`T_b F = 1`, cone interior).
- **G2.** Cone-opposite coefficient is named `BLOCKED`.
- **G3.** `run_discovery` with `budget == 0` is `search_incomplete`.
- **G4.** A non-origin witness in the box (or leftover-recorded empty box).
- **G5.** Pipeline adapter + honesty flags false.

## 9. Benchmark plan

`benchmarks/ns_core_search.py` writes
`docs/benchmarks/ns_core_search_smoke.json`. Algebra and honesty run
in CI. `--full` writes `$OMNIBIAS_SCRATCH/training/ns_core_search/`.

## 10. Honesty and scope

Not Theorem 1.1. Pulses / joining / cutoff stay `external_premises`.
Not a second proof of Clay (C)/(D). Unforced (A)/(B) stays external.
`navier_stokes_proof_claim` is never set by hand. Forbidden: “we
prove global regularity”, “we reproduce the OpenAI proof”.

## 11. Open questions and risks

- A hit that is only the origin is a unit test, not discovery.
- The true continuum residual is not this short `Fraction` identity.
- Falsifier: origin fails Lemma 4.1, or the box has no second witness
  and that miss is claimed as discovery.

## 12. Implementation checklist

- [x] `omnibias.pinn.certified.anisotropic` family + residual
- [x] `omnibias.pinn.jax.discovery.ns_core`
- [x] `NSCoreAdapter` on `pipeline.py`
- [x] `packages/omnibias-pinn/tests/certified/test_ns_core_search.py`
- [x] `benchmarks/ns_core_search.py` plus smoke JSON
- [x] Docs page and nav entry
- [x] Index row in `theory/README.md`

## 13. Parent problem and the exact reason it stays an external obligation

**Parent: Navier-Stokes forced blowup (Clay C/D)**, already true in the
world. This fragment searches a finite axis-regular jet box. It does
not claim a second proof. Unforced Navier-Stokes regularity
(Clay A/B) stays an external obligation. The parent-level honesty
flag stays false.
