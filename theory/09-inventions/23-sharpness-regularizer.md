# 09-23 Sharpness regularizer

## 1. Thesis and status

Add exact `lambda_max(H)` or `Tr(H)` to the **loss**, using exact
HVPs — distinct from 08-06, which only *schedules* the cubic step
size.

- **Status**: shipped (G1–G4 CI; G4 torch/jax parity on G1; not 08-06 schedule)
- **Depends on**: 08-06, 09-01
- **Blocks**: none

### Operator card

- **Benefit.** SAM-like sharpness control without a two-step
  perturbation dance, because HVPs are exact.
- **How it works.** `L_sharp = L + mu * ritz_lambda_max(H)` or
  `mu * exact_trace` on a tower loss. `H` via `hvp` / Lanczos as in
  08-06, but the scalar enters `L`.
- **Strength.** Ill-conditioned PINN losses that overfit residual
  then rebound (documented FNO/PINO caveat).
- **When to use.** After 08-06 exists as code.
- **When not.** As a rewrite of 08-06. Not CCF. Not ImageNet SAM SOTA.
- **Accuracy floor.** Ritz underestimates `lambda_max`.

## 2. Where it lands

`omnibias.{torch,jax}.optim` beside 08-06 / `hvp`. No new package.

## 3. Prior art in omnibias

- Spec 08-06 — `lambda_max` sets cubic `sigma` / lr (schedule)
- `omnibias.{torch,jax}.optim.hvp`, Lanczos helpers
- FNO/PINO early-stopping caveat in `docs/benchmarks.md`

**Confirmed gap.** Nothing adds exact `lambda_max` to `L`.

## 4. Mathematics

`H = D^2 L` from exact HVPs (**bias collapse** `sigma''`). Lanczos
Ritz value `λ_R <= λ_max`. Regularizer uses `λ_R` and records the
gap honesty. No temperature collapse.

## 5. Worked example

`L(theta) = 5 theta^2`. `H=10`, `lambda_max=10`, `Tr=10`.
`L_sharp = 5 theta^2 + 0.1 * 10 = 5 theta^2 + 1`. At `theta=0`,
`L_sharp=1`. Implementer: `hvp` of this quadratic matches `10` to
`1e-12`; adding `mu=0.1` shifts the scalar loss by `1`.

## 6. Proposed API

Does not exist yet. Bit-identical torch / jax twins; default dtype.

```python
# omnibias/torch/optim_sharp_loss.py  (and jax twin) — proposed
@dataclass(frozen=True)
class SharpnessLossConfig:
    mu: float = 0.1
    kind: str = "lambda_max"   # lambda_max | trace
    lanczos_k: int = 4

def sharpness_augmented_loss(loss_fn, theta, *, config: SharpnessLossConfig):
    """L + mu * ritz. Artifact must set schedule_only=false."""
```

## 7. Practical use cases

1. **Quadratic sanity** (worked example).
2. **1-D PINN** that overfits then rebounds: report best-checkpoint
   residual vs 08-06 schedule-only (no required win).
3. **Not** SAM on ImageNet.

## 8. Acceptance gates

- **G1.** Worked `H=10` to `1e-12`; augmented loss at `0` is `1`.
- **G2 skill.** On a named 1-D Poisson, five seeds: training
  **finishes finite** (100% ; same spirit as 08-06 G1) and skill vs
  `u=0` positive. A win vs 08-06 is reported, not required.
- **G3 honesty.** `is_08_06_schedule` is `false`.
- **G4 parity.** torch / jax bit-identical on G1.

## 9. Benchmark plan

- `benchmarks/sharpness_regularizer.py`
- Smoke: `docs/benchmarks/sharpness_regularizer_smoke.json`
- `--full`: `$OMNIBIAS_SCRATCH/inventions/sharp_loss/`

## 10. Honesty and scope

- Distinct from 08-06. Not stretch. Ritz is a lower estimate.
- Certificate tier: empirical.

## 11. Open questions and risks

- **`mu` too large** freezes the quadratic term. Sweep `mu` in
  `--full`.
- **Falsifier.** G1 HVP is wrong (then this is not exact).

## 12. Implementation checklist

- [x] Sharpness-augmented loss twins
- [x] G1 quadratic test
- [x] `benchmarks/sharpness_regularizer.py` plus smoke JSON
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
