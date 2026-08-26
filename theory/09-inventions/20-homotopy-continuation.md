# 09-20 Homotopy continuation trainer

## 1. Thesis and status

A **path of problems** (linear model to full nonlinear nest). Each
step is taken only if the Kantorovich ball is nonempty (08-04).
Training is a certified homotopy, not epoch soup.

- **Status**: shipped (G1–G4 CI; G4 torch/jax parity on G1; empty ball is a halt)
- **Depends on**: 08-04, 09-01
- **Blocks**: none

### Operator card

- **Benefit.** A schedule of problems with a unique-zero filter on
  every accepted step.
- **How it works.** `H(theta, tau) = (1-tau) H_linear + tau H_full`,
  `tau: 0 -> 1`. At each `tau`, one 08-04-accepted GN/cubic step (or
  halt).
- **Strength.** Roots of residual maps that start easy (linear
  Poisson) and add nonlinearity.
- **When to use.** After 08-04 exists as code.
- **When not.** As a rewrite of one-step 08-04. Not CCF stretch
  (empty ball early).
- **Accuracy floor.** Empty ball => halt; that is success of the
  filter, not a failed train.

## 2. Where it lands

`omnibias.{torch,jax}.optim` beside 08-04. No new package.

## 3. Prior art in omnibias

- Spec 08-04 — `radii_polynomial_certificate` on *one* step.
- `omnibias.core.verified.kantorovich.radii_polynomial_certificate`,
  `newton_kantorovich_bounds`, `krawczyk_certificate`
- Spec 01-08 — tropical-log *homotopy* (different axis)

**Confirmed gap.** No `tau`-schedule that calls 08-04 at each knot.

## 4. Mathematics

`H(·,0)` is linear (unique root, large ball). `H(·,1)` is the target
residual. Continuation is standard; acceptance is 08-04. Jets /
`sigma''` from **bias collapse**. No temperature collapse.

## 5. Worked example

Scalar `H(theta, tau) = theta - 1 - tau theta^2`. At `tau=0`, root
`1`. At small `tau=0.1`, Newton from `1` with
`H'=1-0.2 theta` stays near the analytic root
`(1-sqrt(1-0.4))/0.2` if we take the branch near 1
(`≈ 1.127`). Implementer: 08-04 accept is `true` at `tau=0.1` from
`theta=1`; at `tau=1`, `H=theta-1-theta^2` may have an empty ball
from a far start — halt is allowed and must be recorded.

## 6. Proposed API

Does not exist yet. Bit-identical torch / jax twins; default dtype.

```python
# omnibias/torch/optim_homotopy.py  (and jax twin) — proposed
@dataclass(frozen=True)
class HomotopyConfig:
    n_knots: int = 8
    require_ball: bool = True

def homotopy_train(H_fn, theta0, *, config: HomotopyConfig) -> dict:
    """Returns {theta, tau_final, halted, balls}.
    halted True if 08-04 rejects."""
```

## 7. Practical use cases

1. **Scalar quadratic** (worked example).
2. **1-D semilinear** `-u'' + tau u^3 = f`.
3. **Not** CCF `tau`-continuation as stretch.

## 8. Acceptance gates

- **G1.** Worked `tau=0.1` step is 08-04 accepted and
  `|H(theta',0.1)| < 1e-10`.
- **G2 skill.** Semilinear smoke: `tau_final == 1` on at least 3 of 5
  seeds with residual `< 1e-6`, **or** honest `halted` with
  `tau_final` recorded (G2 passes if the artifact never claims a root
  after a reject).
- **G3 honesty.** `stretch_claim` false; empty ball is not a forge.
- **G4 parity.** torch / jax bit-identical on G1.

## 9. Benchmark plan

- `benchmarks/homotopy_continuation.py`
- Smoke: `docs/benchmarks/homotopy_continuation_smoke.json`
- `--full`: `$OMNIBIAS_SCRATCH/inventions/homotopy/`

## 10. Honesty and scope

- Sound enclosure on each accepted step (08-04). Not a continuum
  theorem. Not stretch. `theorem_prover_verified` not asserted.
- Certificate tier: sound enclosure (per step) + empirical path.

## 11. Open questions and risks

- **Too few knots** => reject. Increase `n_knots` before claiming
  failure of the method.
- **Falsifier.** G1 reject on the easy quadratic.

## 12. Implementation checklist

- [x] Homotopy wrapper calling 08-04
- [x] Halt-recording tests
- [x] `benchmarks/homotopy_continuation.py` plus smoke JSON
- [x] `__all__` update
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
