# 10-02 Jet-adjoint policy optimization

## 1. Thesis and status

A discrete adjoint (costate) policy-gradient stack for a declared
`DifferentiableEnvironment`, where every Jacobian the recursion needs
(`dpi/dy`, environment `A_k`/`B_k`, `dpi/dtheta`) is exact -- `dpi/dy` from
the closed-form directional jet tower, the rest from ordinary reverse-mode
autodiff -- plus a PEARL-style learned-terminal-adjoint bootstrap
(`PSDTerminalHead`) for short-window training.

- **Status**: shipped (G1/G2/G9 CI-earned; G3/G4/G8 measured-baseline
  benchmarks are the pending `benchmarks/` deliverable, leftover-recorded
  here until that lands)
- **Depends on**: 10-01 (ledger)
- **Blocks**: 10-03 (certified horizon / gradient-bias consume this
  module's Jacobians), 10-04 (contact environments plug into this seam)

## 2. Where it lands

No new package. Pure-Python recursion algebra in
`omnibias.core.adjoint` (mirrors the small-matrix style of
`omnibias.core.control_lqr.scalar_finite_horizon_lqr`, spec 08-11, of
which `discrete_riccati_sweep` here is the matrix generalisation).
Backend twins and the containers they close over:

- `omnibias.control.ocp` -- `DifferentiableEnvironment` protocol, `OCPSpec`,
  `Rollout`, `rollout_generic` (pure containers, no tensor ops).
- `omnibias.control.{jax,torch}.adjoint` -- closed-loop Jacobian assembly
  (`policy_jacobian_dy` via `mlp_jet` / `layer_jet` / `compose_jet`;
  environment and parameter Jacobians via ordinary `jax.jacrev` /
  `torch.func.jacrev`), and `actor_adjoint_gradient`.
- `omnibias.control.{jax,torch}.policy` -- the five trainer steps
  (`bptt_step`, `truncated_bptt_step`, `zero_order_step`,
  `actor_adjoint_step`, `actor_adjoint_jet_step`) plus `PSDTerminalHead`
  and `fit_terminal_head`.
- `omnibias.control.{jax,torch}.envs` -- `DoubleGyrePointMass` (2-D ODE)
  and `AdvectionDiffusionGrid` (PDE-grid, on
  `omnibias.pinn.solver.{jax,torch}.evolution.advection_diffusion_semidiscrete`
  / `spectral.SpectralGrid1D`, both existing).
- `omnibias.control.robotics` -- `DifferentiableEnvironment` adapters for
  Brax / MuJoCo-MJX (Phase 5; see section 7 and honesty).

## 3. Prior art in omnibias

- `omnibias.core.control_lqr.scalar_finite_horizon_lqr` (08-11) -- scalar
  directional Riccati sweep for a *training* step; this spec's
  `discrete_riccati_sweep` is the stated matrix generalisation, applied to
  a genuine state-space plant, not a loss-along-a-direction scalar.
- `omnibias.{torch,jax}.jet.{mlp_jet,layer_jet,compose_jet}` -- the exact
  directional-jet kernels this spec reuses verbatim for `dpi/dy`; no new
  jet math.
- `omnibias.core.verified.kantorovich.kantorovich_accept_step` -- the plan
  originally named this for terminal-head trust-region acceptance; shipped
  code instead validates the head by direct held-out error measurement
  (`fit_terminal_head` regression plus the `head_error` diagnostic), which
  is simpler and equally checkable for the PSD-quadratic head's low
  parameter count. Kantorovich acceptance stays a documented extension
  point, not silently claimed as shipped.
- **Confirmed gap.** No existing module computed a multi-step,
  multi-dimensional policy-gradient adjoint with an exact `dpi/dy` from the
  tower; see 10-01 section 3 for the full disambiguation against
  08-10/08-11/09-29.

## 4. Mathematics

Locally linearised closed-loop transition
`y_{k+1} ~= A_k y_k + B_k u_k`, state-feedback policy `u_k = pi(y_k; theta)`.
The discrete Pontryagin / adjoint (costate) recursion:

```
M_k = A_k + B_k @ dpi/dy_k
c_k = dL/dy_k + dL/du_k @ dpi/dy_k
lambda_k = M_k^T lambda_{k+1} + c_k,          lambda_H = dPhi/dy_H
grad_theta J = sum_k dpi/dtheta_k^T (dL/du_k + B_k^T lambda_{k+1})
```

This is the classical costate recursion underlying reverse-mode BPTT and
the DDP / iLQR backward pass -- **not a new algorithm**. The delta this
spec ships is where `M_k`'s `dpi/dy_k` factor comes from: the founding
**bias collapse** (`delta -> 0`) closed-form derivative tower
(`mlp_jet` contracted along the adjoint direction), not a finite
difference and not truncated unrolled autodiff. No `beta -> inf`
temperature collapse appears anywhere in this module; do not conflate the
two.

`discrete_riccati_sweep` is the time-varying finite-horizon LQR Riccati
backward sweep (`P_H = Qf`, `K_k` the optimal gain); for a problem run
through it, `lambda_k = P_k y_k` exactly (`lqr_costate_from_riccati`) --
the analytic reference for gate G2.

`n_step_adjoint_bootstrap` / `td_lambda_mix` implement the TD(`lambda`)
geometric blend PEARL uses to learn a *terminal* adjoint correction instead
of a value function: an `n`-step target propagates a learned terminal
estimate backward through `n` exact steps, and

```
lambda_bar = (1-lam) sum_{n=1}^{N-1} lam^{n-1} G^(n) + lam^{N-1} G^(N)
```

`PSDTerminalHead.value_gradient(y) = (L L^T)(y - target)` is the gradient
of the PSD quadratic `V(y) = 0.5 (y-target)^T (L L^T) (y-target)` -- PSD
for any real `L`, so this is a closed-form Riccati-style terminal value,
not a generic critic network.

## 5. Worked example

`omnibias.core.adjoint.worked_example()`: a fixed 2-state, 6-step LQR
problem. `discrete_riccati_sweep` gives `P_k`/`K_k`; rolling the optimal
closed loop forward and running `adjoint_recursion` backward on the same
trajectory reproduces `lambda_k = P_k y_k` to `max_abs_err = 1.78e-15`
(gate G2, threshold `< 1e-10`).

## 6. Proposed API

Shipped.

```python
from omnibias.core.adjoint import discrete_riccati_sweep, adjoint_recursion, policy_gradient

# also: omnibias.control.jax.policy.{bptt_step, truncated_bptt_step,
# zero_order_step, actor_adjoint_step, actor_adjoint_jet_step, PSDTerminalHead}
# and the omnibias.control.torch twins (bit-identical call signatures,
# torch.Tensor in place of jax.Array).
```

Dtype policy: `omnibias.core.adjoint` uses `numpy.float64` throughout (a
pure-Python/numpy module, no framework default to defer to); the
`{jax,torch}.adjoint` / `.policy` twins pass through whatever dtype the
caller's `layers` / environment already use (framework default unless the
caller overrides it), never a hardcoded `float32`.

## 7. Practical use cases

1. **Exact-vs-truncated diagnosis.** `bptt_step` vs `actor_adjoint_step`
   on the same problem must agree to float64 round-off (gate G1); any
   divergence is a bug, not noise -- useful for regression-testing a new
   environment's Jacobians.
2. **Short-horizon training with a certified escape hatch.**
   `actor_adjoint_jet_step` trains on a short `window` (cheap) while
   `omnibias.control.certified.gradient_bias` (10-03) bounds exactly what
   that shortcut cost.
3. **PDE-scale environments.** `AdvectionDiffusionGrid` is the environment
   used to *measure* the `O(state_dim)` cost of one directional jet per
   state dimension in `policy_jacobian_dy`, rather than assuming it away.
4. **Robotics via a wrapped engine.** `omnibias.control.robotics` lets the
   same five trainers run on Brax / MuJoCo-MJX once installed, without a
   native rigid-body implementation in this repository.

## 8. Acceptance gates

- **G1 exactness/parity.** `policy_jacobian_dy` (closed-form) matches
  `jax.jacrev` / `torch.func.jacrev` through the same forward to
  `atol=1e-10`; `actor_adjoint_gradient` matches full BPTT
  (`jax.grad` / `torch.autograd`) to `atol=1e-8`. **Earned**
  (`test_control_adjoint_backend.py`).
- **G2 analytic reference.** Finite-horizon discrete LQR: adjoint equals
  `P_k y_k` to `< 1e-10`. **Earned** (`max_abs_err = 1.78e-15`,
  `omnibias.core.adjoint.worked_example` / `adjoint_skill`).
- **G3 control performance.** `require_skill` vs zero-control, plus a
  Phase-A-measured tracking-distance threshold, worst of `>= 5` seeds.
  **Not yet measured against a committed baseline curve** -- the unit
  tests exercise correctness, not a skill benchmark; this is the pending
  `benchmarks/actor_adjoint_control.py` deliverable. Leftover-recorded,
  not claimed earned.
- **G4 sample efficiency.** Interactions to reach a problem-fixed return
  `R*` vs PPO/TD3/BPTT/tBPTT/SHAC/PEARL baselines. **Leftover-recorded**
  for the same reason as G3; `zero_order_step` / `truncated_bptt_step`
  exist as the comparison arms but no committed curve exists yet.
- **G8 cost.** `require_cost_parity` vs SHAC on one hardware class.
  **Leftover-recorded**: neither `brax` nor `mujoco` was installed while
  developing this spec (see `omnibias.control.robotics` module docstring),
  so no real robotics cost measurement exists; the adapter *seam* is
  tested against `FakeArticulatedEnvironment` instead.
- **G9 full-pipeline bit-parity.** `omnibias.core.adjoint` is backend-free
  pure Python; `omnibias.control.jax.adjoint` and
  `omnibias.control.torch.adjoint` both call the identical module object
  (`jax_adjoint.adjoint_recursion is torch_adjoint.adjoint_recursion`).
  **Earned structurally** (`test_control_adjoint_backend.py`), not merely
  measured to a tolerance.

## 9. Benchmark plan

- `benchmarks/adjoint_exactness.py` (G1/G2/G9) -- pending, per the
  `benchmarks` to-do; unit-test coverage of the same claims already exists
  in `packages/omnibias-control/tests/test_control_adjoint_backend.py` and
  `packages/omnibias-core/tests/test_adjoint.py`.
- `benchmarks/actor_adjoint_control.py` (G3/G4) -- pending; needs a
  Phase-A measured curve before any threshold is set.
- Smoke JSON target: `docs/benchmarks/adjoint_exactness_smoke.json`.
- `--full`: `$OMNIBIAS_SCRATCH/control/adjoint/`.

## 10. Honesty and scope

- Not a solved Hamilton-Jacobi-Bellman equation. Not the infinite-horizon
  algebraic Riccati equation / DARE. Policy-gradient training finds a
  stationary point of `J(theta)`, never asserted as a global optimum.
- `actor_adjoint_jet_step`'s gradient is biased whenever the learned head
  has not converged; that bias is exactly what 10-03's
  `omnibias.control.certified.gradient_bias` bounds, conditionally, never
  asserted zero here.
- Two founding limits appear across this group and must not be conflated:
  `omnibias.control.__init__` now declares `__lineage__ = "both"` --
  the CBF-QP safety filter is temperature collapse (`beta -> inf`,
  feasibility); this module's `dpi/dy` is founding bias collapse
  (`delta -> 0`, the closed-form tower). Neither substitutes for the
  other.
- `theorem_prover_verified` is not asserted anywhere in this module.
- G3/G4/G8 are explicitly not claimed earned (section 8); no prose in
  this codebase should read otherwise until a committed benchmark JSON
  exists.

## 11. Open questions and risks

- **Cost at PDE scale** is the primary technical risk the plan named.
  `AdvectionDiffusionGrid` exists specifically to measure it; no
  microbenchmark JSON is committed yet (part of the pending `benchmarks`
  to-do).
- **Second-order adjoints are fragile.** `discrete_riccati_sweep` raises
  `ValueError` on a singular `R + B^T P B` rather than returning a
  plausible-looking but wrong gain; DDP-style divergence on poor curvature
  is a known failure mode of the class, mitigated by the PSD
  parameterization of the terminal head, not eliminated.
- **Kantorovich trust-region acceptance was replaced.** The plan named
  `kantorovich_accept_step` for the terminal head; shipped code instead
  validates by direct held-out residual (`head_error`). If a future spec
  wants the trust-region-certified version, it is additive, not a
  correction of a wrong claim (none was made).
- **Falsifier.** If `test_control_adjoint_backend.py`'s G1 comparison
  ever needs `atol` looser than `1e-8`, the closed-form `dpi/dy` path has
  a bug -- do not raise the tolerance to make it pass.

## 12. Implementation checklist

- [x] `omnibias.core.adjoint` (recursion, Riccati sweep, TD(lambda) mix,
      worked example, skill)
- [x] `omnibias.control.ocp` (`DifferentiableEnvironment`, `OCPSpec`,
      `Rollout`, `rollout_generic`)
- [x] `omnibias.control.{jax,torch}.adjoint` (closed-loop Jacobian
      assembly, `actor_adjoint_gradient`)
- [x] `omnibias.control.{jax,torch}.envs` (`DoubleGyrePointMass`,
      `AdvectionDiffusionGrid`)
- [x] `omnibias.control.{jax,torch}.policy` (five trainer steps,
      `PSDTerminalHead`, `fit_terminal_head`)
- [x] `omnibias.control.robotics` (Brax / MuJoCo-MJX adapters,
      `FakeArticulatedEnvironment`)
- [x] Tests: `test_adjoint.py`, `test_ocp.py`, `test_control_envs.py`,
      `test_control_adjoint_backend.py`, `test_control_policy.py`,
      `test_control_robotics.py`
- [x] Torch/jax parity tests (`test_control_adjoint_backend.py` G9)
- [ ] `benchmarks/adjoint_exactness.py`, `benchmarks/actor_adjoint_control.py`
      (pending; smoke JSON not yet committed)
- [ ] Docs page + mkdocs nav entry (pending `docs-and-wiring` to-do)
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
  (`torch.get_default_dtype()` / `keras.config.floatx()`), never a
  hardcoded `float32`.
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
