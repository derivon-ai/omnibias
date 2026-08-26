# 09-29 Plant PID layer

## 1. Thesis and status

A SISO plant controller whose measurement is
`y(t) = sigma(alpha t + beta)` and whose I / D terms are the closed-form
antiderivative and derivative tower, not a discrete sum or a finite
difference.

- **Status**: shipped (G1–G4 CI; activation-of-time plant; not 08-10)
- **Depends on**: 09-01, 09-03
- **Blocks**: none

### Operator card

- **Benefit.** `I` is FTC of `r - sigma` via `S' = sigma`. `D` is
  `-alpha sigma'`. Both spend the tower / `integral` role.
- **How it works.** Evaluate `e = r - y`, `I` by the exact formula,
  `D` by the Riccati identity, `u = kp e + ki I + kd D`.
- **Strength.** 1-D tracking of an activation-shaped measurement.
- **When to use.** As a nominal plant controller next to
  `omnibias.control` CBF. Not as a `theta` trainer (that is 08-10).
- **When not.** As cruise-control SOTA. As 08-10. As CCF stretch.
- **Accuracy floor.** The measurement *is* `sigma(alpha t + beta)`.

## 2. Where it lands

`omnibias.core.pid_layer` plus thin
`omnibias.{torch,jax}.plant_pid` twins. No new package. Reuses
`omnibias.core.ftc` `sigmoid` / `softplus`.

## 3. Prior art in omnibias

- Spec 09-03 — `integral` cell `S(z+b_hi)-S(z+b_lo)`. This spec
  spends `S` on a tracking error, not an FTC-Net hidden cell.
- Spec 08-10 — jet-PID *trainer* on `L(theta + s d)`. Distinct.
- `omnibias.control` CBF-QP — wraps a nominal `a_nom`; this layer
  can supply `a_nom`.
- Activation Riccati — used as `sigma'`, not as an LQR gain.

**Confirmed gap.** No plant controller whose I term is the activation
antiderivative of a tracking error.

## 4. Mathematics

`z(t) = alpha t + beta`, `y = sigma(z)`, `e = r - y`.

```
P = e(t)
D = -alpha sigma'(z)
I = r (t-t0) - (S(z(t)) - S(z(t0))) / alpha     (alpha != 0)
I = e (t-t0)                                      (alpha = 0)
u = kp P + ki I + kd D
```

`S' = sigma` (`softplus` for sigmoid, `log cosh` for tanh). Founding
**bias collapse** (`delta -> 0`) supplies `sigma'`. No temperature
collapse (`beta -> inf`, feasibility). This is not a discrete running
sum.

## 5. Worked example

Sigmoid, `alpha=1`, `beta=0`, `r=0.5`, `t0=-1`, `t=0`.
`y(0)=1/2`, `e=0`, `D=-1/4`. `I` matches a fine trapezoid of `e` to
`1e-8`. `ki=1`, `kd=1` yields `u = I + D`.

## 6. Proposed API

Shipped.

```python
from omnibias.core.pid_layer import PlantPIDConfig, plant_pid

report = plant_pid(0.0, config=PlantPIDConfig(ki=1.0, kd=1.0, t0=-1.0))
```

## 7. Practical use cases

1. **Exact I vs trapezoid** on a sigmoid window.
2. **Nominal `a_nom`** into `cbf_filter`.
3. **Not** 08-10. **Not** CCF Hilbert.

## 8. Acceptance gates

- **G1 FTC / D.** Trapezoid residual of `I` `< 1e-8`. `D` matches
  `-sigma'(0)` at the named point.
- **G2 skill.** Random windows, both families: median FTC residual
  `< 1e-8`. Unknown family raises.
- **G3 honesty.** Payload forbids trainer claim, cruise SOTA, discrete
  sum I, stretch, skipped chain rule.
- **G4 parity.** Torch / jax bit-identical on G1.

## 9. Benchmark plan

- `benchmarks/plant_pid.py`
- Smoke: `docs/benchmarks/plant_pid_smoke.json`
- `--full`: `$OMNIBIAS_SCRATCH/inventions/plant_pid/`
- CI smoke after implementation.

## 10. Honesty and scope

- Not a global minimum of a deep nest. Not a `theta` trainer (08-10).
- Hilbert × dictionary remains `~1e-1`; this layer does not set or
  weaken `CCF_STRETCH_RESIDUAL_GATE`.
- The plant is `sigma(alpha t + beta)`, not an identified MIMO model.
- Not cruise / process-control SOTA.
- Bias collapse supplies `sigma'`; the `integral` role supplies `S`.

## 11. Open questions and risks

- **A textbook PID is prior art.** Only exact I / D on an activation
  measurement is new.
- **CBF wrapper is leftover.** Feeding `u` into `cbf_filter` as
  `a_nom` is a consumer, not this ship. MIMO / identified plants stay
  leftover.
- **Falsifier.** If G1 FTC residual misses `1e-8`, the antiderivative
  is wrong; do not ship.

## 12. Implementation checklist

- [x] `plant_pid` + FTC / Riccati helpers
- [x] Torch / jax twins
- [x] Tests: G1–G3, `alpha=0`, tanh, parity
- [x] `benchmarks/plant_pid.py` plus smoke JSON
- [x] Index row in `theory/README.md`

---

## Repo invariants this spec must respect

Check these before submitting an implementation.

- **Pure core**: no torch, jax, tensorflow or keras imports from
  `omnibias.core`. Pure-Python math lives there and every backend imports it.
- **Bit-identical twins**: torch and jax implementations must agree exactly.
  Polynomial coefficients come from `omnibias.core.polynomials`; never fork them
  per backend.
- **Default dtype**: use the framework default
  (`torch.get_default_dtype()` / `keras.config.floatx()`), never a hardcoded
  `float32`.
- **Vendor-neutral language**: no scheduler commands, vendor names, internal
  hostnames, usernames, or absolute local paths in tracked files. Artifacts go
  to `$OMNIBIAS_SCRATCH`, defaulting to a repo-relative `artifacts/`.
  `packages/omnibias-core/tests/test_no_leakage.py` enforces this across the
  whole readable surface.
- **Terminology**: distinguish the founding bias collapse (`delta -> 0`, yields a
  derivative) from temperature collapse (`beta -> inf`, yields a 0/1 step). The
  older penalty phrasings are retired and guarded by `tests/test_terminology.py`.
  Every new `**/relaxation.py` must carry the cross-reference note and be listed
  in `PENALTY_FILES` in
  `packages/omnibias-core/tests/test_concept_terminology.py`.
- **Executable docs**: if any part of this lands in `docs/` or a package README,
  every fenced Python block is executed by `tests/test_docs_snippets.py`. Verify
  calls against real signatures; opting out needs a directive with a stated
  reason.
- **Earned flags**: `theorem_prover_verified` is set only by a genuine
  `lake build` pass of the Mathlib-free kernel, and `mathlib_verified` only by a
  genuine pass of the Mathlib-backed project. Asserting either without a pass
  blocks the verdict.
- **Typing tier**: the strict CI gate covers `core`, `torch`, `jax` and
  `ferminet`. Newly authored modules should be written strict-clean regardless
  of tier; curated beta modules can be added to
  `scripts/mypy_strict_allowlist.txt` once
  `mypy --strict --follow-imports=silent <file>` is clean.
