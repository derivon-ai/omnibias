# 08-11 Jet-LQR optimizer

## 1. Thesis and status

A directional jet of `phi(s) = L(theta + s d)` supplies the linear
error dynamics `e+ = e + phi''(0) u`. Finite-horizon discrete LQR
on that scalar plant is the first control `u = -K_0 phi'(0)`.

- **Status**: shipped (G1–G4 CI; scalar discrete Riccati; not DARE)
- **Depends on**: 08-01, 03-12, 08-10
- **Blocks**: none

### Operator card

- **Benefit.** Explicit `(Q, R, Qf)` effort vs error on the same
  restriction as 03-12 / 08-10. `R=0`, `N=1`, `Qf=1` is Newton.
- **How it works.** Backward scalar Riccati on `A=1`, `B=phi''(0)`.
  Step `s* = clip(-K_0 phi'(0), -s_max, s_max)`.
- **Strength.** Quadratic bowls and locally quadratic restrictions.
- **When to use.** After 08-10 when a quadratic regulator (not PID
  gains) is the right knob. 08-12 recedes this.
- **When not.** As plant MIMO LQR. As the algebraic Riccati / DARE.
  As the activation Riccati. As a global solver. Not CCF stretch.
- **Accuracy floor.** Quadratic model; `phi''=0` and `R=0` is singular.

## 2. Where it lands

`omnibias.core.control_lqr` plus thin
`omnibias.{torch,jax}.optim_lqr` twins. No new package.
`scalar_finite_horizon_lqr` is the reusable Riccati for 08-12.

## 3. Prior art in omnibias

- Spec 03-12 — same restriction, root of the model.
- Spec 08-10 — PID gains on the same jet. This spec is LQR.
- `omnibias.{torch,jax}.optim` — Newton family. Newton is the
  `R=0, N=1, Qf=1` corner, not a rename of CubicNewton.
- Activation Riccati (`omnibias.core.polynomials`, 09-10) — a
  different equation. Do not merge.

**Confirmed gap.** No optimizer whose step is a finite-horizon
discrete Riccati on the jet of `L`.

## 4. Mathematics

Let `e_0 = phi'(0)`, `H = phi''(0)`. The model of the directional
derivative along a step `u` is `e+ = e + H u` (`A=1`, `B=H`).
Finite-horizon LQR with costs `Q, R, Qf` and horizon `N` solves

```
P_N = Qf
K_k = (R + B^T P_{k+1} B)^{-1} B^T P_{k+1} A
P_k = Q + A^T P_{k+1} A - K_k^T (R + B^T P_{k+1} B) K_k
u_0 = -K_0 e_0
```

This is a **scalar discrete Riccati recursion**. It is not the
infinite-horizon algebraic Riccati equation (DARE) and not
`sigma' = s(1-s)`. Founding **bias collapse** (`delta -> 0`) supplies
`phi^(k)`. No temperature collapse (`beta -> inf`, feasibility).
Faà di Bruno is the chain rule that forms the jet.

When `R=0`, `N=1`, `Qf=1`, and `H != 0`:
`K_0 = H / H^2 = 1/H`, `u_0 = -phi'/phi''` (Newton).

## 5. Worked example

`phi(s)=(s-1)^2`, start at `0`: `phi'=-2`, `phi''=2`.
`R=0`, `N=1`, `Qf=1` yields `K_0=1/2`, `s*=1`.
`R=4` yields `s*=1/2` (effort shrinks the Newton step).

## 6. Proposed API

Shipped.

```python
from omnibias.core.control_lqr import JetLQRConfig, lqr_step_from_derivatives

report = lqr_step_from_derivatives(
    (1.0, -2.0, 2.0),
    config=JetLQRConfig(q=0.0, r=0.0, qf=1.0, horizon=1),
)
```

Torch / jax: `jet_lqr_step(loss_fn, params, direction, config=...)`.
Eager; do not `jit` / `torch.compile` the driver.

## 7. Practical use cases

1. **Newton corner** on a bowl.
2. **Effort-regularized** Newton via `R>0`.
3. **Not** plant cruise / quadrotor LQR. **Not** CCF Hilbert.

## 8. Acceptance gates

- **G1 Newton.** On `phi(s)=(s-1)^2` from `0`, `R=0` steps to `1`
  within `1e-15`. `R=4` shrinks to `0.5`.
- **G2 skill.** Random bowls: median `|s*-c| < 1e-12`. `R=0` and
  `phi''=0` raises singular.
- **G3 honesty.** Payload forbids global min, plant LQR, DARE,
  activation-Riccati-as-gain, MPC, stretch, skipped chain rule.
- **G4 parity.** Torch / jax bit-identical on G1.

## 9. Benchmark plan

- `benchmarks/jet_lqr.py`
- Smoke: `docs/benchmarks/jet_lqr_smoke.json`
- `--full`: `$OMNIBIAS_SCRATCH/training/jet_lqr/`
- CI smoke after implementation.

## 10. Honesty and scope

- Not a global minimum of a deep nest. Faà di Bruno is the chain rule.
- Hilbert × dictionary remains `~1e-1`; this trainer does not set or
  weaken `CCF_STRETCH_RESIDUAL_GATE`.
- Not a plant LQR. Not the algebraic Riccati / DARE. Not the
  activation Riccati. Not MPC (receding + constraints is 08-12).
- Bias collapse supplies the jet; it does not certify a continuum
  solution.
- Distinct from 08-10 (PID gains, not a Riccati).

## 11. Open questions and risks

- **A textbook discrete LQR is prior art.** The new object is LQR on
  the *jet of L*, with Newton as a named corner.
- **Falsifier.** If G1 misses `s*=1`, the `(A,B,x)` convention is
  wrong; do not ship.

## 12. Implementation checklist

- [x] `scalar_finite_horizon_lqr` plus `lqr_step_from_derivatives`
- [x] Torch / jax `jet_lqr_step` twins
- [x] Tests: G1–G3, Newton identity, singular refuse, parity
- [x] `benchmarks/jet_lqr.py` plus smoke JSON
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
