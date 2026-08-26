# 08-10 Jet-PID optimizer

## 1. Thesis and status

A directional Taylor jet of `phi(s) = L(theta + s d)` turns PID into
an exact (on the model) P / I / D triple: `P = phi'(0)`,
`I` is the fundamental-theorem increment of the Taylor polynomial, and
`D = phi''(0)`.

- **Status**: shipped (G1–G4 CI; local 1-D restriction; not a plant PID)
- **Depends on**: 08-01, 03-12
- **Blocks**: none

### Operator card

- **Benefit.** The I term is an exact FTC of the jet, not a leaked
  running sum of past gradients. D is `phi''`, not a finite difference.
- **How it works.** Form the restriction jet (bias collapse). Evaluate
  `u = kp P + ki I + kd D` and step `s* = clip(-u, -s_max, s_max)`
  along `d`.
- **Strength.** 1-D bowls and other jets whose model is faithful on
  the step.
- **When to use.** After 03-12 when a damped first-order step on the
  same restriction is wanted. Optional in the 08-01 stack.
- **When not.** As a plant PID. As LQR / MPC. As a global solver.
  Not CCF stretch.
- **Accuracy floor.** Jet truncation `R_N`. Optional Lagrange refuse.

## 2. Where it lands

`omnibias.core.control_pid` plus thin
`omnibias.{torch,jax}.optim_pid` twins. No new package. The polynomial
solve is the 03-12 host-side restriction; tensor jets reuse
`directional_derivatives`.

## 3. Prior art in omnibias

- Spec 03-12 — exact jet line search on the same `phi(s)`.
- `omnibias.core.line_search` — `taylor_coeffs_from_derivatives`,
  `poly_eval`, `lagrange_remainder_bound`.
- `omnibias.{torch,jax}.optim` — Newton family. This spec is PID on
  the jet, not another Newton rename.
- Spec 08-09 — I/O filter after a step. Do not merge.
- `omnibias.control` CBF-QP — plant safety filter, not a `theta`
  trainer.

**Confirmed gap.** No optimizer whose I term is the FTC of a Taylor
model of `L` along `d`.

## 4. Mathematics

Let `phi(s) = L(theta + s d)` and let `p` be its degree-`N` Taylor
model at `0` (coefficients `a_k = phi^(k)(0) / k!`). The tracking
error is `e(s) = phi'(s)`. On the model:

```
P = p'(0) = phi'(0)
I_window = p(h) - p(0)          # exact FTC of p' on [0, h]
I_running += p(s*) - p(0)       # exact FTC of the applied step
D = p''(0) = phi''(0)
u = kp P + ki I + kd D
s* = clip(-u, -s_max, s_max)
```

The founding **bias collapse** (`delta -> 0`) supplies `phi^(k)` and
the jets. No temperature collapse (`beta -> inf`, feasibility).
Faà di Bruno is the chain rule that forms the jet. The I term is
exact on `p`; `|phi(h) - p(h)|` is the Lagrange remainder, which may
refuse a step. Activation Riccati is not a gain formula.

## 5. Worked example

`phi(s) = (s-1)^2`, start at `s=0`: `phi=1`, `phi'=-2`, `phi''=2`.
`kp=0.5`, `ki=0`, `kd=0` yields `u=-1`, `s*=1` (the minimizer).
`kp=1`, `kd=0` overshoots (`s*=2`). `kp=1`, `kd=0.5` restores the
Newton step `s*=-phi'/phi''=1`.

## 6. Proposed API

Shipped.

```python
from omnibias.core.control_pid import JetPIDConfig, pid_step_from_derivatives

report = pid_step_from_derivatives(
    (1.0, -2.0, 2.0),
    config=JetPIDConfig(kp=0.5, ki=0.0, kd=0.0, s_max=2.0),
)
```

Torch / jax: `jet_pid_step(loss_fn, params, direction, config=...)`.
Eager; do not `jit` / `torch.compile` the driver.

## 7. Practical use cases

1. **Quadratic bowl.** P-only with `kp=0.5` is exact.
2. **D-damped overshoot** on the same bowl.
3. **Not** plant cruise control. **Not** CCF Hilbert.

## 8. Acceptance gates

- **G1 bowl.** On `phi(s)=(s-1)^2` from `0`, `kp=0.5` steps to `1`
  within `1e-15`. D-damping restores the Newton step.
- **G2 skill.** Random bowls `(s-c)^2` with `c in (0.25, 0.75)`:
  median `|s*-c| < 1e-12`. `kd` without `phi''` raises.
- **G3 honesty.** Payload forbids global min, plant PID, LQR, MPC,
  algebraic Riccati, stretch, skipped chain rule.
- **G4 parity.** Torch / jax bit-identical on G1.

## 9. Benchmark plan

- `benchmarks/jet_pid.py`
- Smoke: `docs/benchmarks/jet_pid_smoke.json`
- `--full`: `$OMNIBIAS_SCRATCH/training/jet_pid/`
- CI smoke after implementation.

## 10. Honesty and scope

- Not a global minimum of a deep nest. Faà di Bruno is the chain rule.
- Hilbert × dictionary remains `~1e-1`; this trainer does not set or
  weaken `CCF_STRETCH_RESIDUAL_GATE`.
- Not a plant PID, not LQR, not MPC.
- I is exact on the Taylor *model*. Remainder is real.
- Bias collapse supplies the jet; it does not certify a continuum
  solution.
- Distinct from 03-12 (root of the model, not PID gains).

## 11. Open questions and risks

- **A generic discrete PID optimizer is prior art.** Only the FTC-I /
  exact-D reading is new. Tests must exercise the FTC identity.
- **Falsifier.** If G1 misses `s*=1`, the sign convention is wrong;
  do not ship.

## 12. Implementation checklist

- [x] `pid_step_from_derivatives` with model-window and running I
- [x] Torch / jax `jet_pid_step` twins
- [x] Tests: G1–G3, FTC identity, remainder refuse, parity
- [x] `benchmarks/jet_pid.py` plus smoke JSON
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
