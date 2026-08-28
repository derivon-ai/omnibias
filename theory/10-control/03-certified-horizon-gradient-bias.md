# 10-03 Certified horizon and policy-gradient-bias enclosure

## 1. Thesis and status

Two sound (interval-arithmetic) enclosures for the two things every
short-horizon adjoint trainer (10-02) currently picks by hand: a
*certified truncation horizon* derived from the enclosed closed-loop
discrete monodromy, and a *sound, conditional* bound on the resulting
policy-gradient bias `||grad_theta J - grad_theta J_h||`. Both compose
into the proof-carrying `ControllerBundle`.

- **Status**: shipped (module code + unit-test coverage; the >= 1000-trial
  full-acceptance run named in the original plan is a smoke-scale subset
  today -- section 8 states the honest gap)
- **Depends on**: 10-02 (consumes `closed_loop_jacobian` and the
  environment/policy Jacobians it assembles)
- **Blocks**: none

## 2. Where it lands

No new package. `omnibias.control.horizon` (certified horizon) and
`omnibias.control.certified.gradient_bias` (gradient-bias enclosure), plus
`omnibias.control.bundle` (`ControllerBundle`, the Phase-3 proof-carrying
container). The second module lives under `certified/`; claims follow
`.cursor/rules/omnibias.md` (Frontier program).

## 3. Prior art in omnibias

- `omnibias.dynamics.spectral_radius_bound` / `omnibias.dynamics._core.
  variational.variational_step` (spec 07-06) -- `spectral_radius_bound` is
  reused **verbatim**, not reimplemented, for the interval spectral bound.
  `variational_flow` / `monodromy_matrix` in that same module are
  continuous-time ODE flow enclosures (Lohner-stepped fundamental
  matrices) and do **not** apply here: the adjoint recursion's `M_j`
  sequence is already discrete, so there is no ODE to step. This spec's
  `monodromy_product` instead composes the interval matrix product
  directly with `omnibias.core.verified.linalg.matmul`, the identical
  operation `variational_step` uses to accumulate its fundamental matrix.
- `omnibias.verify.lipschitz_bound` -- reused verbatim as the sole source
  of the terminal-adjoint-error Lipschitz constant; this spec adds no new
  Lipschitz-bound algorithm.
- `omnibias.core.verified.interval.Interval` / `omnibias.core.verified.
  linalg.{matmul, identity_matrix}` -- the outward-rounded interval
  primitives this spec's balls and products are built from.
- `omnibias.control.problem.RecoverableCertificate` /
  `omnibias.control.certify.certify_recoverable` -- unchanged; `bundle.py`
  composes with these rather than replacing them.
- **Confirmed gap.** No existing module derived a training-horizon length
  from a certificate instead of a hand-picked constant, and no module
  bounded the specific bias a learned terminal-adjoint substitution
  introduces into a policy gradient.

## 4. Mathematics

### Certified horizon

Closed-loop discrete monodromy over a window of length `h`:

```
Phi_h = M_{k+h-1} @ ... @ M_k,    M_j = A_j + B_j @ dpi/dy_j
```

Given a caller-declared *sound* radius `r` around each point Jacobian
`M_j` (a ball, not derived here), `enclose_jacobian_ball` builds the
elementwise interval box `[M_ij - r, M_ij + r]`, and
`monodromy_product` composes those boxes left-to-right via interval
matrix multiplication -- a rigorous enclosure of `Phi_h` for *every*
trajectory whose true per-step Jacobian stays inside the declared ball.
`spectral_radius_bound` brackets that enclosure's spectral radius; if the
upper bound is `<= tol`, the window is *provably* contracting and
`certified_horizon` returns the smallest such `h` (never a fabricated one
-- `certified=False` if none is found within `max_horizon`).

This is **founding bias collapse** (`delta -> 0`), not temperature
collapse (`beta -> inf`, the feasibility sense): the certificate is a
sound enclosure of a fixed finite product, not a sharpening gate. Do not
conflate the two. **Coverage is conditional on the declared ball**: if the
true per-step Jacobian variation exceeds the caller's declared radius,
the certificate is unsound by construction. Deriving that radius
automatically from a Lipschitz bound on `dpi/dy` and the environment
Jacobian is a named leftover (section 11), not shipped in this pass.

### Gradient-bias enclosure

A short-horizon trainer substitutes a learned `lambda_pred_h` for the true
continuation adjoint `lambda_true_h` at truncation. Every per-step
Jacobian for the first `h` steps is exact; linearising the costate
recursion in that one substitution gives an exact first-order bias:

```
||grad_theta J - grad_theta J_h|| <=
    ( sum_{k=0}^{h-1} ||dpi/dtheta_k|| * ||B_k|| * ||Phi_{k+1}|| ) * ||lambda_pred_h - lambda_true_h||
```

with `Phi_{k+1} = M_{h-1} ... M_{k+1}` (identity at `k = h-1`). The
operator norms of `M_k`/`B_k`/`dpi/dtheta_k` are **ordinary
floating-point** computations (`_op_norm_sym`, `max(||A||_inf, ||A||_1)`,
a sound bound on both `||A||_inf` and `||A^T||_inf` so no transpose
direction needs separate tracking) -- these matrices are point values from
autodiff, not themselves enclosed. The genuinely certified link in the
chain is `terminal_adjoint_error_bound`: a sound, outward-rounded
Lipschitz-extension bound

```
sup_box ||lambda_pred(y) - lambda_true(y)|| <= |e(y0)| + L * radius(box)
```

built by composing `omnibias.verify.lipschitz_bound` (verbatim) with a
caller-measured point residual `|e(y0)|` at the box centre.

This is **founding bias collapse** (`delta -> 0`), not temperature
collapse (`beta -> inf`, the feasibility sense); nothing here sharpens a
`beta -> inf` gate. Do not conflate the two. The overall bound is
**conditional**: sound given (1) the supplied `error_net` genuinely
represents `lambda_pred - lambda_true` over the box (a caller modelling
choice, unverified here) and (2) the point residual is itself accurate.
Neither condition is checked by this module.

## 5. Worked example

`omnibias.control.horizon.worked_example()`: a diagonal contracting system
`diag(0.5, 0.6)`, declared radius `0.01`, `tol = 0.05`. Certified horizon
`h = 7` against an analytic point-matrix reference `ceil(log(0.05) /
log(0.6)) = 6` (the certificate is one step more conservative than the
point trajectory, as expected of a sound outer enclosure). `g5_earned`
checks `horizon <= analytic_h + 2`.

`omnibias.control.certified.gradient_bias.worked_example()`: a linear
error net with weight `diag(0.5, 0.5)` over box `[-0.1, 0.1]^2` and point
residual `0.01` gives `terminal_adjoint_error_bound = 0.01 + 0.5*0.1 =
0.06` exactly; feeding that into `truncation_bias_bound` over a 2-step
`M = diag(0.8, 0.8)` system gives `gradient_bias_bound = 0.0324`.

## 6. Proposed API

Shipped.

```python
from omnibias.control.horizon import certified_horizon
from omnibias.control.certified.gradient_bias import (
    truncation_bias_bound,
    terminal_adjoint_error_bound,
)
from omnibias.control.bundle import build_bundle, bundle_status
```

Both `horizon.py` and `certified/gradient_bias.py` use `numpy.float64`
internally (interval arithmetic is defined over Python floats /
`omnibias.core.verified.interval.Interval`, not a tensor framework); there
is no jax/torch twin for either -- they consume the *outputs* of the
`{jax,torch}.adjoint` Jacobian assembly, not tensors themselves.

## 7. Practical use cases

1. **Derive, don't guess, `h`.** Replace a hand-picked SHAC/PEARL window
   length with `certified_horizon`'s smallest provably-contracting window.
2. **Know what a shortcut costs.** Before shipping an
   `actor_adjoint_jet_step`-trained policy, compute
   `truncation_bias_bound` from the training-time Jacobians and the
   measured terminal-head error to get an honest number, not a hope.
3. **One bundle to ship.** `build_bundle` composes both certificates with
   the pre-existing `RecoverableCertificate` so a deployed policy carries
   all three proofs (or an honest `"partial"` / `"uncertified"` verdict).

## 8. Acceptance gates

- **G5 certified horizon.** Plan target: `require_enclosure_coverage`
  exactly `1.0` over `>= 1000` trials; certified `h` matches or beats
  hand-tuned `h=16`. **Partially earned**: `horizon_skill` (default
  `n=200`, CI-run at `n=40`) gives coverage `1.0` on every sampled
  contracting diagonal system, and never violates the true point-matrix
  trajectory (asserted, not just measured) -- but the CI run is `n=40`,
  not `>= 1000`. The `>= 1000`-trial full run is the pending `benchmarks`
  deliverable; do not read the current CI pass as satisfying the full
  plan threshold.
- **G6 gradient-bias enclosure.** Plan target: coverage exactly `1.0` over
  `>= 1000` random `(theta, y, mu)` draws; width reported, never
  co-gated. **Partially earned** the same way: `gradient_bias_skill`
  (CI-run at `n=150`) gives coverage `1.0` against a closed-form realized
  bias on every draw; the `>= 1000`-trial run is pending.
- **Recoverable-set G7** (pre-existing, unchanged): `certify_disc_recoverable`
  still returns `certified=True` over its declared box; this spec does
  not touch that certificate's logic.

## 9. Benchmark plan

- `benchmarks/certified_horizon.py` (G5) -- pending; would run
  `horizon_skill(n=1000)` (or the full sweep it approximates) and commit
  `docs/benchmarks/certified_horizon_smoke.json`.
- `benchmarks/gradient_bias_enclosure.py` (G6) -- pending; would run
  `gradient_bias_skill(n=1000)` and commit
  `docs/benchmarks/gradient_bias_enclosure_smoke.json`.
- `--full`: `$OMNIBIAS_SCRATCH/control/certified/`.

## 10. Honesty and scope

- The horizon certificate is sound **only within the caller-declared
  per-step Jacobian radius** -- it is not an unconditional guarantee about
  the true nonlinear system, and this module does not derive that radius
  from a Lipschitz bound automatically (section 11).
- The gradient-bias bound is **conditional** on the caller's `error_net`
  and point-residual measurement genuinely modelling the true terminal
  error; neither is checked here. Callers must record which case applies.
- `theorem_prover_verified` is never asserted by either module. The
  interval spectral bound stacking into a Lean-checkable statement is a
  plausible rung-3 stretch, per `docs/honesty.md` -- leftover-recorded,
  not assumed.
- `ControllerBundle`'s verdict is a plain conjunction over whichever
  certificates are present; a missing slot downgrades to `"partial"`,
  never silently counts as passed. `bundle.summary` never emits the
  phrase "certified safe control" -- only "model-relative"; that exact
  phrase is reserved for `omnibias.control.__init__`.

## 11. Open questions and risks

- **Auto-deriving the per-step ball radius** from a Lipschitz bound on
  `dpi/dy` and the environment Jacobian (rather than requiring the caller
  to declare one) is the natural next step and is not shipped here.
- **Interval arithmetic will not scale** to large state dimensions --
  this spec's monodromy products are sized for the same reduced-order
  regime spec 07-06 already scopes its claims to.
- **Falsifier.** If `horizon_skill`'s assertion
  (`actual_point_rho <= tol` whenever `certified=True`) ever fails on a
  random seed, the enclosure is unsound -- this is checked by raising
  `AssertionError` inline in the skill function itself, not only in a
  test.

## 12. Implementation checklist

- [x] `omnibias.control.horizon` (`enclose_jacobian_ball`,
      `monodromy_product`, `certified_horizon`, worked example, skill)
- [x] `omnibias.control.certified.gradient_bias`
      (`truncation_bias_bound`, `terminal_adjoint_error_bound`, worked
      example, skill)
- [x] `omnibias.control.bundle` (`ControllerBundle`, `bundle_status`,
      `build_bundle`, `summary`)
- [x] Tests: `test_control_horizon.py`, `test_control_gradient_bias.py`,
      `test_control_bundle.py`
- [x] No torch/jax split needed (interval-arithmetic modules, backend-free)
- [ ] `benchmarks/certified_horizon.py`, `benchmarks/gradient_bias_enclosure.py`
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
