# 10-04 Certified contact-force smoothing

## 1. Thesis and status

A smoothed one-sided normal-contact force law whose exact derivative tower
comes from the founding bias collapse (the same sigmoid Eulerian recurrence
every other activation derivative in this library uses) *and* whose
hardening error as the smoothing sharpens (temperature collapse) carries a
sound, one-sided enclosure -- rather than the usual single informal
smoothing with no bound on how wrong it is near contact.

- **Status**: shipped (1-D point-mass validation; G1/G2 CI-earned; 2-D
  Coulomb block is leftover-recorded, not shipped)
- **Depends on**: 10-02 (the contact environments implement
  `DifferentiableEnvironment` and plug into the same trainers)
- **Blocks**: none

## 2. Where it lands

No new package. Pure-Python force law and bias bound in
`omnibias.core.contact_smoothing` (reuses
`omnibias.core.polynomials.sigmoid_polynomial_coeffs`,
`omnibias.core.ftc.sigmoid`, and
`omnibias.core.verified.{interval.Interval, transcend.exp_iv}`, no new
primitives). Environments:
`omnibias.control.{jax,torch}.contact.ContactPointMass1D` plus
`contact_certificate`.

## 3. Prior art in omnibias

- `omnibias.core.polynomials.sigmoid_polynomial_coeffs` -- the exact
  Eulerian-polynomial recurrence for `sigma^(n)`; this spec's
  `contact_force_tower` spends it on `f_smooth(z) = f_max * sigma(-beta z)`
  rather than a new derivative algorithm.
- `omnibias.core.verified.transcend.exp_iv` -- the rigorous interval
  exponential used (verbatim) inside `contact_smoothing_bias_bound`.
- `omnibias.shape.morphology` / other `beta -> inf` (temperature-collapse)
  specs already sharpen a soft gate into a 0/1 indicator; what is new here
  is pairing that sharpening with a **sound enclosure** of its own error,
  rather than asserting "sharp enough" informally.
- **Confirmed gap.** No existing contact/complementarity primitive in
  omnibias carried both an exact derivative tower *and* a certified bound
  on the smoothing-vs-hard discrepancy.

## 4. Mathematics

Signorini normal-contact complementarity `z >= 0, f >= 0, z f = 0` has no
gradient at the boundary. This spec's smooth force:

```
f_smooth(z) = f_max * sigmoid(-beta * z)
f_hard(z)   = f_max if z <= 0 else 0        (the beta -> inf limit)
```

**Two distinct limits, deliberately not conflated:**

1. **The smoothed force and every derivative are closed form** (founding
   **bias collapse**, `delta -> 0`): with `u = -beta z`, the chain rule
   through the affine map contributes `(-beta)^n` at order `n`, and
   `sigma^(n)(u)` comes from `sigmoid_polynomial_coeffs` -- available for
   *any* order `n`, no finite difference, no truncated series.
2. **The smoothing itself is temperature collapse** (`beta -> inf`): as
   `beta` grows, `f_smooth` sharpens toward `f_hard`. This spec supplies a
   *sound* one-sided enclosure of that hardening error rather than only
   asserting it informally:

```
|f_smooth(z) - f_hard(z)| <= f_max * exp(-beta * z_min)      for |z| >= z_min > 0
```

using the elementary tail bound `sigma(x) <= exp(x)` for `x <= 0`
(immediate from `sigma(x) = e^x / (1+e^x) <= e^x`). The bound **excludes a
neighbourhood of `z = 0`**: for `z_min <= 0` the module returns
`certified=False` ("Inconclusive"), because the two force laws disagree
most exactly at contact and no finite bound holds there -- this excluded
band is the honest price of certification, not a hidden limitation.

Neither limit is the third one in this codebase, **Enclosure Collapse**
(`width -> 0` of a sound enclosure, a point plus a proof): the enclosure
width here is fixed by the caller's `(z_min, beta)`, not driven to zero by
this module.

## 5. Worked example

`omnibias.core.contact_smoothing.worked_example()`: `beta=8`, `f_max=3`,
`z0=0.7`. The exact-tower first derivative matches a central finite
difference (`h=1e-4`) to residual `9.19e-9`. `contact_smoothing_bias_bound
(z_min=0.5, beta=8, f_max=3)` gives `bound.hi = 0.0549`, and the bound
holds (`grid_ok=True`) at every test point with `|z| >= 0.5`.

## 6. Proposed API

Shipped.

```python
from omnibias.core.contact_smoothing import (
    contact_force_smooth,
    contact_force_hard,
    contact_force_tower,
    contact_smoothing_bias_bound,
)
from omnibias.control.jax.contact import ContactPointMass1D, contact_certificate
# omnibias.control.torch.contact.ContactPointMass1D is the bit-identical twin
```

`omnibias.core.contact_smoothing` is plain Python floats (no tensor
framework); `{jax,torch}.contact.ContactPointMass1D` implement
`DifferentiableEnvironment` for their respective backends and are checked
against the core module and each other for exact numeric agreement (not
approximate) at shared inputs.

## 7. Practical use cases

1. **Differentiable bounce with a stated error.** Train a policy through
   `ContactPointMass1D` via `actor_adjoint_step` while knowing exactly how
   far the smoothed dynamics can differ from the true Signorini law away
   from contact.
2. **Choosing `beta`.** `contact_smoothing_bias_bound` turns "pick a large
   enough hardness" into a number: the smallest `beta` for a target bound
   at a target `z_min`.
3. **Not** a rigid-body engine or a 2-D friction model -- see section 10.

## 8. Acceptance gates

- **G1 tower exactness.** First-derivative tower matches a finite
  difference to residual `< 1e-7` (measured `9.19e-9`, well inside).
  **Earned** (`test_contact_smoothing.py`).
- **G2 skill / coverage.** Random `(z_min, beta, f_max)` draws with
  `|z| >= z_min`: the realized smoothing discrepancy never exceeds the
  certified bound; `z_min <= 0` always reports `Inconclusive`.
  **Earned** at CI scale (`contact_skill`, `n=200`, coverage `1.0`).
- **Environment parity.** `ContactPointMass1D` jax/torch outputs and the
  core force law agree to `< 1e-10` at shared inputs. **Earned**
  (`test_control_contact.py`).
- **End-to-end differentiability.** `actor_adjoint_step` on
  `ContactPointMass1D` produces a finite gradient. **Earned**
  (`test_control_contact.py`).

## 9. Benchmark plan

- `benchmarks/contact_smoothing.py` -- pending (the `benchmarks` to-do);
  would formalise `worked_example` / `contact_skill` into the
  `provenance` + `write_json` + `gates_block` template and commit
  `docs/benchmarks/contact_smoothing_smoke.json`.
- `--full`: `$OMNIBIAS_SCRATCH/control/contact/`.

## 10. Honesty and scope

- **Not** a rigid-body physics engine. **Not** a 2-D Coulomb-friction
  contact model -- validated only on a 1-D bouncing point mass; a 2-D
  block with friction is explicitly leftover-recorded, not shipped.
- The bias bound is **one-sided** (`f_smooth` vs `f_hard`, not a two-sided
  Signorini complementarity certificate) and holds only for `|z| >= z_min`.
  `z_min <= 0` is `Inconclusive`, never a fabricated number.
- Two collapse senses coexist in this module by design and are each named
  explicitly at every use (`contact_smoothing.py`'s docstring, listed in
  `PENALTY_FILES`): bias collapse never appears on the `beta` axis;
  temperature collapse never appears on the `sigma^(n)` axis.
- `theorem_prover_verified` is not asserted.

## 11. Open questions and risks

- **2-D Coulomb friction** is the natural next step and is explicitly
  deferred, per the plan's "1-D impact, then a 2-D block before any
  robotics claim" staging.
- **Native rigid-body contact** (multiple simultaneous contacts, complementarity
  LCP solves) is out of scope for this spec; `omnibias.control.robotics`
  (10-02) is where a wrapped engine's own contact model would be used
  instead.
- **Falsifier.** If `contact_skill`'s coverage ever drops below `1.0` on a
  declared `z_min > 0`, the tail bound derivation
  (`sigma(x) <= exp(x)` for `x <= 0`) has been misapplied -- do not loosen
  the bound to compensate; find the error.

## 12. Implementation checklist

- [x] `omnibias.core.contact_smoothing` (force law, hard limit, tower,
      bias bound, worked example, skill)
- [x] `omnibias.control.{jax,torch}.contact` (`ContactPointMass1D`,
      `contact_certificate`)
- [x] Tests: `test_contact_smoothing.py`, `test_control_contact.py`
- [x] Torch/jax parity test (`test_control_contact.py`)
- [ ] `benchmarks/contact_smoothing.py` (pending; smoke JSON not yet
      committed)
- [ ] Docs page + mkdocs nav entry (pending `docs-and-wiring` to-do)
- [x] `contact_smoothing.py` added to `PENALTY_FILES` in
      `test_concept_terminology.py`
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
