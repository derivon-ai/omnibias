# 07-16 Boussinesq remainder CAP

## 1. Thesis and status

The IPM remainder loop on Boussinesq: CubicGN discovery, a named
smoke-grid residual hull that contains a truth sample, and leftover
#54 that the full streamfunction-Poisson remainder stays external.
`lambda_n` stays an empirical hypothesis.

- **Status**: shipped (G1–G5 CI; leftover #54 recorded; not Navier-Stokes)
- **Depends on**: 07-15
- **Blocks**: none

## 2. Where it lands

`omnibias.pinn.jax.discovery.boussinesq` and
`omnibias.pinn.certified.boussinesq`. `BoussinesqAdapter` on
`omnibias.pinn.jax.discovery.pipeline`. Cookbook
`docs/cookbook/boussinesq-singularity.md` stays a consumer. No new
package.

## 3. Prior art in omnibias

- `run_boussinesq_discovery` was an Adam smoke scaffold.
- 07-15 is the IPM twin (CubicGN + leftover #53).
- `lambda_n` formula in `lambda_laws` is `empirical_hypothesis_not_theorem`.

**Confirmed gap.** Earn-path optimizer was Adam; CAP had no named
remainder leftover.

## 4. Mathematics

Gaussian-poly or compactified affine-hat jets. CubicGN on
`(R_omega, R_theta, R_psi)`. Interval hull of the computed grid
contains an independently recomputed sample. Not a continuum CAP.
Not temperature collapse. Not Navier-Stokes.

## 5. Worked example

`n=6`, `steps=2`, CubicGN. Adapter refuses Adam.
`enclose_boussinesq_grid_residual` has `contains_truth_sample=True`
and `full_boussinesq_proved=False`. `lambda_n_hypothesis_is_theorem`
stays false.

## 6. Proposed API

```
run_boussinesq_discovery(BoussinesqDiscoveryConfig(method="cubic"))
enclose_boussinesq_grid_residual(discovery)
BoussinesqAdapter.discover(optimizer="cubic_gn")
```

## 7. Practical use cases

- Earn-path Boussinesq ticks without Adam.
- Keep `lambda_n` labeled as a hypothesis.
- Pair with 07-15 as the remainder-CAP lane, not CCF.

## 8. Acceptance gates

- **G1.** CubicGN path; Adam raises.
- **G2.** Named enclosure contains a truth sample.
- **G3.** `full_boussinesq_proved` is false.
- **G4.** Leftover #54 recorded; `lambda_n` is not a theorem.
- **G5.** Honesty: `navier_stokes_proof_claim=False`.

## 9. Benchmark plan

`benchmarks/boussinesq_remainder_cap.py` writes
`docs/benchmarks/boussinesq_remainder_cap_smoke.json`. CI. `--full`
under `$OMNIBIAS_SCRATCH/training/boussinesq_remainder_cap/`.

## 10. Honesty and scope

Not Navier-Stokes. Not Euler blowup. `lambda_n` stays a hypothesis.
Forbidden: “we prove global regularity”, stamping
`navier_stokes_proof_claim`.

## 11. Open questions and risks

- Compactified hats may stay far from a true profile.
- A float residual is not a proof.
- Falsifier: Adam on the earn path, or `lambda_n` stamped a theorem.

## 12. Implementation checklist

- [x] CubicGN in `run_boussinesq_discovery`
- [x] `enclose_boussinesq_grid_residual` leftover #54
- [x] `BoussinesqAdapter` forbids Adam
- [x] tests + smoke JSON
- [x] Index row in `theory/README.md`

## 13. Parent problem and the exact reason it stays an external obligation

**Parent: finite-time singularity of 3D Euler / Navier-Stokes.**
Boussinesq is a different model equation. A smoke-grid residual hull
is not Euler or Navier-Stokes blowup evidence. The parent stays
external. The never-write line is: our residual is not evidence for
Euler or Navier-Stokes blowup.
