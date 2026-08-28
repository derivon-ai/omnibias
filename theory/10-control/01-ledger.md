# 10-01 Control-systems optimization ledger

## 1. Thesis and status

This is the assignment table for Group 10: it records what already existed
in `omnibias.control` / `omnibias.core` before this group (and is *not*
plant control), and maps the three new specs (10-02, 10-03, 10-04) onto
their exact homes so nobody re-derives the "is this a plant controller"
question per file.

- **Status**: shipped (index + assignment table; no code of its own)
- **Depends on**: none
- **Blocks**: 10-02, 10-03, 10-04 (this ledger is read first)

## 2. Where it lands

This document only. No module, no package. Prose home is
`theory/10-control/`; the code it indexes lands in `omnibias.core`,
`omnibias.control`, `omnibias.control.{jax,torch}`, and
`omnibias.control.certified`, all existing packages.

## 3. Prior art in omnibias

Three trainer/plant modules already exist and are frequently confused with
"optimal control of a dynamical system." None of them is:

- **`omnibias.core.control_lqr.scalar_finite_horizon_lqr`** (spec 08-11) --
  a **1-D directional** discrete Riccati sweep used to set a *training*
  step size `theta_{k+1} = theta_k - lr_k grad_k` along one direction. It
  is a training-loop optimizer, not a controller for a state-space plant.
  10-02's `omnibias.core.adjoint.discrete_riccati_sweep` is the genuine
  **matrix** generalisation this spec's docstring already says
  `scalar_finite_horizon_lqr` is a restriction of.
- **`omnibias.core.control_pid` / `omnibias.{torch,jax}.optim_pid`**
  (spec 08-10) -- a jet-PID **optimizer** that treats the loss-along-a-
  direction `L(theta + s d)` as the "plant" and steps `s`. Not a plant
  PID.
- **`omnibias.core.pid_layer` / `omnibias.{torch,jax}.plant_pid`**
  (spec 09-29) -- a genuine plant controller, but a scalar one: the
  "plant" is the closed-form curve `y(t) = sigma(alpha t + beta)`, tracked
  by exact activation calculus (`S' = sigma`), not a discrete-time
  state-feedback policy over a declared `(A, B)` or a general nonlinear
  `f(y, u)`.

**Confirmed gap** (before this group): no module in omnibias computed an
exact policy-gradient adjoint over a genuinely multi-step, multi-dimensional
`DifferentiableEnvironment`, and no module certified anything about that
gradient (a truncation horizon, or a bias bound). Groups 08/09 are
training-loop and single-plant work; this group is the first to treat
*optimal control of a declared dynamical system* as the object.

**Existing, reused (not duplicated) by this group:**

- `omnibias.control.problem` / `omnibias.control.certify` -- the CBF-QP
  safety filter and `RecoverableCertificate`. Unchanged; 10-03's proof-
  carrying bundle composes with it rather than replacing it.
- `omnibias.dynamics.spectral_radius_bound` (spec 07-06) -- reused
  verbatim by 10-03 for the certified-horizon spectral bound.
- `omnibias.verify.lipschitz_bound` -- reused verbatim by 10-03 for the
  terminal-adjoint error bound.
- `omnibias.core.polynomials.sigmoid_polynomial_coeffs` -- reused verbatim
  by 10-04's contact-force tower.
- `omnibias.core.parameter_jets` (spec 09-27) -- named as the mechanism
  for Phase-4 scenario generalisation (`dlambda/dmu`); not itself modified
  by this group.

## 4. Mathematics

None new in this ledger; see 10-02/10-03/10-04. This document only
disambiguates: the discrete Pontryagin / costate recursion in 10-02 is
classical (it is reverse-mode backpropagation-through-time restated in
control language, per Bradbury/Betts-style adjoint methods); what is new
is sourcing every Jacobian in it from the closed-form derivative tower
(founding bias collapse, `delta -> 0`) rather than a finite difference.

## 5. Worked example

N/A (ledger; no computation).

## 6. Proposed API

N/A. This document indexes existing and new APIs; it defines none.

## 7. Practical use cases

1. **Onboarding.** Anyone extending Group 10 reads this first to avoid
   re-litigating "is 08-10/08-11/09-29 already the thing I'm building."
2. **Guarding scope creep.** Future specs citing "control" must state
   which of the four homes (08-10/08-11 trainer, 09-29 scalar plant,
   10-02+ multi-step adjoint) they extend.

## 8. Acceptance gates

N/A (no falsifiable claim; this is an index).

## 9. Benchmark plan

N/A.

## 10. Honesty and scope

- Does **not** claim 08-10/08-11/09-29 are wrong or superseded -- they
  solve different, narrower problems (a training step size; a scalar
  plant) and stay exactly as scoped.
- Does **not** claim this group is the first differentiable-control work
  anywhere; it claims a specific, checkable delta over what already
  shipped in this repository (section 3, "Confirmed gap").

## 11. Open questions and risks

- If a future spec needs a *general* nonlinear plant PID (MIMO, identified
  model), it is a new spec, not a silent extension of 09-29.

## 12. Implementation checklist

- [x] This document.
- [x] Index rows in `theory/README.md` for 10-01..10-04.

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
