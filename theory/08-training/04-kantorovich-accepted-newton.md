# 08-04 Kantorovich-accepted Newton

## 1. Thesis and status

A Gauss–Newton or cubic-Newton step is **legal** only when
`radii_polynomial_certificate` returns a nonempty unique-zero ball around
the trial point, so the optimizer can refuse a step that is not a
certified isolated root of the residual map.

- **Status**: designed
- **Depends on**: 08-01, 03-12
- **Blocks**: none

### Operator card

- **Benefit.** Local uniqueness, not a faster step: after accept, there
  is a true zero of the finite residual map in `B(theta_bar, r)`.
- **How it works.** Compute a trial `theta'` by existing `GaussNewton` /
  `CubicNewton`. Evaluate `Y0, Z0, Z1, Z2` for `F(theta) = residual` (or
  `grad L`) via `newton_kantorovich_bounds`. Accept iff
  `radii_polynomial_certificate(...)` is not `None`. Otherwise reject and
  shrink (cubic `lambda`, trust radius, or 03-12 step).
- **Strength.** Toy residuals and small PINNs already near a root, where
  a paper wants a sealed critical point.
- **When to use.** After a residual is already small (`~1e-8` or better)
  and the map is finite-dimensional. Stack: direction (GN) → length
  (03-12) → this accept/reject.
- **When not.** CCF stretch (Hilbert error makes `Y0` huge); claiming
  continuum PDE existence; as a replacement for Adam in early training
  (the ball will be empty and every step rejects).
- **Accuracy floor.** Enclosure quality and Lipschitz of `DF`. Empty
  ball ⇒ no step, not "the method failed to train."

## 2. Where it lands

A wrapper around shipped `GaussNewton` / `CubicNewton` in
`omnibias.{torch,jax}.optim`, calling
`omnibias.core.verified.kantorovich.radii_polynomial_certificate`.
No new package. The bounds assembly may live in
`omnibias.core.verified` if it is backend-free (residual as a Python
callable on intervals); tensor trial steps stay in torch/jax.

## 3. Prior art in omnibias

- `omnibias.core.verified.kantorovich` —
  `radii_polynomial_certificate`, `newton_kantorovich_bounds`,
  `krawczyk_certificate`, `RadiiCertificate`.
- `omnibias.core.verified.pde_certificate.radii_polynomial_residual_certificate`
  — residual form already exists for certified PDEs; this spec is the
  **optimizer policy** that refuses updates without a ball, not a new
  radii formula.
- `omnibias.{torch,jax}.optim.GaussNewton`, `CubicNewton`.
- Spec 08-09 certifies an **input-output** Lipschitz / box after a step.
  This spec certifies a **root** of `F`. Do not merge them.

**Confirmed gap.** No optimizer consults `radii_polynomial_certificate`
before applying a step.

## 4. Mathematics

For an approximate zero `x_bar` of a finite map `F` and an approximate
inverse `A ≈ DF(x_bar)^{-1}`, the radii polynomial

```
p(r) = Z2 r^2 - (1 - Z0 - Z1) r + Y0
```

has a unique zero of `F` in `B(x_bar, r)` when some `r > 0` satisfies
`p(r) < 0` and `Z0 + Z1 + 2 Z2 r < 1` (see the module docstring of
`kantorovich.py`). `Y0, Z0, Z1, Z2` are rigorous **upper** bounds.

The founding **bias collapse** enters only insofar as `DF` uses exact
`sigma'` (smaller `Y0` than finite-difference Jacobian). No temperature
collapse.

**Scope.** `F` is the discrete residual (collocation vector) or a finite
parameter gradient. The ball is in parameter space or in the residual's
domain as declared in the certificate `claim` string. It is not a ball
in a function space of continuum PDE solutions.

## 5. Worked example

**Scalar quadratic.** `F(x) = x^2 - 2`, `x_bar = 1.5`, `F(1.5) = 0.25`,
`F' = 2x`, `A = 1/3`. Then `Y0 >= |A F| = 1/12 ≈ 0.0833`. On
`B(1.5, r)` with `r = 0.2`, `F'` ranges in `[2.6, 3.4]`,
`|1 - A F'| <= max(|1 - 2.6/3|, |1 - 3.4/3|) ≈ 0.133`, which feeds
`Z0, Z1`. For this polynomial `Z2` is controlled by `|A| * 2` (Lipschitz
of `F'`). The implementer must call `newton_kantorovich_bounds` and
obtain a nonempty `RadiiCertificate` for `r` near `0.1` (true root
`sqrt(2) ≈ 1.414` lies in `B(1.5, 0.1)`). A trial point `x_bar = 3.0`
(`F = 7`) must return `None`.

Expected: accept rate `1.0` on the good ball, `0.0` on the far point.

## 6. Proposed API

Does not exist yet.

```python
# omnibias/torch/optim.py  (and jax twin) — proposed
@dataclass(frozen=True)
class KantorovichAccept:
    accepted: bool
    certificate: object | None    # RadiiCertificate
    reason: str                   # "ball" | "empty" | "bounds_failed"

def kantorovich_accept_step(
    residual_iv,     # interval residual map (core, no torch)
    jac_iv,
    A,
    trial_params,
    *,
    r_max: float,
) -> KantorovichAccept:
    """Assemble NK bounds and consult radii_polynomial_certificate.
    Does not exist yet."""
```

The interval maps are pure Python (`omnibias.core.verified`). Torch/jax
only produce `trial_params` and a float `A` estimate; converting `A` to
intervals is the wrapper's job. Default dtype for the trial step; the
certificate is rational/interval, not float32.

## 7. Practical use cases

1. **Seal a 1-D residual root** after GN has reached `1e-10`, for a
   methods paper that wants "unique critical point in this ball."
2. **Reject an overshooting cubic step** whose `Y0` explodes, forcing
   03-12 or a smaller trust radius.
3. **Not** the DeepMind CCF campaign: Hilbert error dominates `Y0`.

## 8. Acceptance gates

- **G1 accept/reject.** On `F(x) = x^2 - 2` (or an equivalent isolated
  polynomial root): 100% accept inside a named ball that contains the
  true root; 0% accept at a named far point. Five seeds if any
  randomization is used (none needed for this `F`).
- **G2 never continuum.** The sealed payload contains
  `continuum_pde_claim: false` (or equivalent) and the claim string
  names a finite map.
- **G3 no forge.** `theorem_prover_verified` is absent unless a genuine
  kernel pass is wired later; the radii certificate is the sound-
  enclosure tier.

## 9. Benchmark plan

- `benchmarks/kantorovich_newton.py`
- Smoke: `docs/benchmarks/kantorovich_newton_smoke.json`
- `--full`: `$OMNIBIAS_SCRATCH/training/kantorovich_newton/`
- CI smoke after implementation.

## 10. Honesty and scope

- Sound enclosure tier. Not Mathlib, not Clay, not NS regularity.
- Empty ball is a **valid outcome** (reject), not a bug.
- Bias collapse may tighten `DF`; it does not certify a continuum
  solution.
- Distinct from 08-09 (I/O Lipschitz).

## 11. Open questions and risks

- **Interval residual of a deep net is expensive** and may be vacuous.
  G1 is a polynomial map on purpose. A PINN G4 is optional and must not
  block G1.
- **Approximate `A`.** A bad inverse yields empty balls (safe) or, if
  bounds are wrong, an unsound accept (forbidden). Bounds must be
  upper bounds; tests must try to break soundness.
- **Falsifier.** If G1 accept/reject is not 100%/0%, the wrapper is
  wrong; do not ship.

## 12. Implementation checklist

- [ ] `kantorovich_accept_step` wrapping
      `radii_polynomial_certificate`
- [ ] `continuum_pde_claim: false` in the artifact
- [ ] Tests: G1 polynomial, soundness (grid + random sample of `F`)
- [ ] `benchmarks/kantorovich_newton.py` plus smoke JSON
- [ ] `__all__` update
- [ ] Index row in `theory/README.md`

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
