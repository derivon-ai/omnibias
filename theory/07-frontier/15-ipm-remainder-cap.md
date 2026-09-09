# 07-15 IPM remainder CAP

## 1. Thesis and status

Lift IPM discovery off Adam onto CubicGN and name a remainder: an
interval hull of the smoke-grid residual that contains a truth sample,
while the full streamfunction-Poisson operator stays external. The
object that can close is still `ipm_banded_toy_radii`.

- **Status**: shipped (G1–G5 CI; leftover #53 recorded; not Navier-Stokes)
- **Depends on**: 07-14, 07-03
- **Blocks**: 07-16

## 2. Where it lands

`omnibias.pinn.jax.discovery.ipm` (CubicGN) and
`omnibias.pinn.certified.ipm` (`enclose_ipm_grid_residual`).
`IPMAdapter` on `omnibias.pinn.jax.discovery.pipeline`. No new
package. Not an extension of `navier_stokes.py`.

## 3. Prior art in omnibias

- `run_ipm_discovery` was an Adam smoke scaffold
  (`optimizer: adam_smoke_scaffold`).
- `train_gn.gauss_newton_minimize` already ships CubicGN.
- `build_ipm_cap_bundle` is JSON packing, not a remainder.
- `ipm_banded_toy_radii` is the named toy that can close.
- `full_ipm_proved` stays false.

**Confirmed gap.** Earn-path optimizer was Adam; CAP had no named
enclosure leftover.

## 4. Mathematics

Gaussian-poly jets for `(Theta, Psi)` are closed form. CubicGN
minimises `||(R_theta, R_psi)||^2`. The smoke-grid residual hull is
an `Interval` of computed floats that contains an independently
recomputed sample at index 0. That hull is not a remainder for the
nonlocal Poisson solve. Founding bias collapse is not in play on the
Adam lift; the enclosure is a sound hull of a finite grid. Not
temperature collapse. Not Navier-Stokes.

## 5. Worked example

`n=8`, `steps=5`, CubicGN. Adapter refuses `optimizer="adam"`.
`enclose_ipm_grid_residual` returns `contains_truth_sample=True` and
`full_ipm_proved=False`. `ipm_banded_toy_radii()["proved"]` may be
true; leftover #53 names that as the only closing object.

## 6. Proposed API

```
run_ipm_discovery(IPMDiscoveryConfig(method="cubic"))
enclose_ipm_grid_residual(discovery) -> hull + leftover #53
IPMAdapter.discover(optimizer="cubic_gn")  # adam raises
```

`full_ipm_proved=False`. Scaffold residual floor `2.0` is not
promoted.

## 7. Practical use cases

- Earn-path IPM ticks without Adam.
- Name what the CAP does and does not enclose.
- Keep the toy radii as the only closing sub-case.

## 8. Acceptance gates

- **G1.** CubicGN path; Adam raises.
- **G2.** Named enclosure contains a truth sample.
- **G3.** `full_ipm_proved` is false.
- **G4.** Leftover #53: only `ipm_banded_toy_radii` closes.
- **G5.** Honesty: `navier_stokes_proof_claim=False`.

## 9. Benchmark plan

`benchmarks/ipm_remainder_cap.py` writes
`docs/benchmarks/ipm_remainder_cap_smoke.json`. CI algebra / leftover.
`--full` under `$OMNIBIAS_SCRATCH/training/ipm_remainder_cap/`.

## 10. Honesty and scope

Not Navier-Stokes. Not Euler blowup. `full_ipm_proved` stays false.
Forbidden: “we prove global regularity”, stamping
`navier_stokes_proof_claim`. Scaffold floor `2.0` is not a physical
remainder gate.

## 11. Open questions and risks

- CubicGN on a two-amplitude Gaussian may not beat the scaffold floor
  in a physically meaningful way.
- A float residual is not a proof.
- Falsifier: Adam still on the earn path, or `full_ipm_proved=True`.

## 12. Implementation checklist

- [x] CubicGN in `run_ipm_discovery`
- [x] `enclose_ipm_grid_residual` leftover #53
- [x] `IPMAdapter` forbids Adam
- [x] tests + smoke JSON
- [x] Index row in `theory/README.md`

## 13. Parent problem and the exact reason it stays an external obligation

**Parent: finite-time singularity of 3D Euler / Navier-Stokes.** IPM
is a different model equation. A smoke-grid residual hull is not
Euler or Navier-Stokes blowup evidence. The parent stays external.
The never-write line is: our residual is not evidence for Euler or
Navier-Stokes blowup.
