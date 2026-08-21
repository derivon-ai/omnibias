# 08-06 Sharpness-scheduled curvature step

## 1. Thesis and status

A few exact Hessian-vector products give `lambda_max` of the loss
Hessian (or GN matrix) in closed form, so the cubic penalty or learning
rate can be set from **measured sharpness** instead of a fixed
hyperparameter or a Hutchinson estimate.

- **Status**: designed
- **Depends on**: 08-01
- **Blocks**: none

### Operator card

- **Benefit.** Fewer diverging cubic / Newton steps on stiff residuals
  without a hand-tuned `lambda`.
- **How it works.** Each step: `k` Lanczos HVPs via existing `hvp` /
  `lanczos_tridiag` (exact `sigma''` through the tower). Set cubic
  `sigma` (regularization) or lr to a named function of
  `lambda_max` (default `max(lambda_max, 0)` scaled by a constant
  recorded in the artifact).
- **Strength.** Stiff quadratics and 1-D PINNs where a fixed cubic
  `lambda` either crawls or explodes.
- **When to use.** As a hook on `CubicNewton` / `CubicRegularizedNewton`
  after a cold start is already finite.
- **When not.** As a CCF stretch plan; as a claim that sharpness *is*
  generalization (it is a step-size signal only).
- **Accuracy floor.** HVP budget (`k`). Not Hilbert.

## 2. Where it lands

A hook on `CubicNewton` in
`packages/omnibias-torch/src/omnibias/torch/optim.py` and the jax twin.
Not a package.

## 3. Prior art in omnibias

- `omnibias.{torch,jax}.optim.hvp`, `lanczos_tridiag`,
  `cubic_regularized_newton_step`, `CubicNewton`.
- Curvature / Fisher surfaces in `omnibias-curvature` and spec 04-01
  (pack Fisher is a different tensor; do not reuse G2 of 04-01 as this
  gate).

**Confirmed gap.** `CubicNewton` takes a scheduled or fixed cubic
coefficient; nothing sets it from a measured `lambda_max` each step.

## 4. Mathematics

For a twice-differentiable `L`, Lanczos on `v |-> Hess L[v]` with exact
HVPs (bias-collapse `sigma''`) yields Ritz values; the largest Ritz
value `ell_k` converges to `lambda_max` from below for the Krylov
subspace. The cubic model

```
m(s) = L + g·s + (1/2) s^T H s + (sigma/6) ||s||^3
```

is stable when `sigma` is large enough relative to the Lipschitz of
`H`. The heuristic this spec **gates** (not proves) is
`sigma = c * max(ell_k, ell_min)` with `c` and `ell_min` named in the
config and written to the artifact.

No temperature collapse. Error term: `lambda_max - ell_k >= 0` (Ritz);
underestimating sharpness can still diverge — G1 requires **finite**
finishes, not an optimal `c`.

## 5. Worked example

**Stiff quadratic.** `L(x, y) = (1/2)(x^2 + 10^4 y^2)`,
`lambda_max = 10^4`. One HVP along `(0,1)` already returns `10^4`.
A fixed cubic `sigma = 1` from `(1, 1)` can overshoot `y`; 
`sigma = c * 10^4` with `c = 1` damps the `y` step. The implementer
checks that 20 scheduled steps from `(1, 1)` stay finite and reach
`L < 1e-8`, while a named too-small fixed `sigma` diverges on at least
one of five seeds (or is documented as "fixed also finite" — then G1
must still show 100% finite for scheduled).

## 6. Proposed API

Does not exist yet.

```python
# omnibias/torch/optim.py  (and jax twin) — proposed
@dataclass(frozen=True)
class SharpnessSchedule:
    n_lanczos: int = 4
    c: float = 1.0
    ell_min: float = 1e-6
    target: str = "cubic_sigma"   # cubic_sigma | lr

def sharpness_lambda_max(loss_fn, params, *, n_lanczos: int) -> float:
    """Largest Ritz value from exact HVPs. Does not exist yet as a
    public helper (lanczos_tridiag already exists internally)."""
```

Default dtype; jax: `loss_fn` must be jit-compatible.

## 7. Practical use cases

1. **Stiff quadratic / 1-D PINN** — the gate.
2. **Pair with 03-12.** Sharpness sets the cubic; the jet sets the
   length along the cubic direction.
3. **Not** a generalization certificate (no "flatter is better" claim).

## 8. Acceptance gates

- **G1 finite finishes.** On a stiff quadratic **or** a 1-D PINN, five
  seeds: 100% of sharpness-scheduled cubic runs finish with finite
  params and finite loss. Report wall time vs Adam (no required win).
- **G2 named schedule.** Artifact records `c`, `n_lanczos`,
  `ell_min`, and per-step `ell_k` (smoke may keep only min/max).
- **G3 parity.** torch / jax `lambda_max` on the stiff quadratic
  agree to `1e-10` relative.

## 9. Benchmark plan

- `benchmarks/sharpness_schedule.py`
- Smoke: `docs/benchmarks/sharpness_schedule_smoke.json`
- `--full`: `$OMNIBIAS_SCRATCH/training/sharpness/`
- CI smoke after implementation.

## 10. Honesty and scope

- Empirical tier. Sharpness is a **step-size** signal, not a
  generalization or Clay claim.
- Does not clear CCF stretch.
- Bias collapse makes HVPs exact; Hutchinson is not the method.

## 11. Open questions and risks

- **Ritz underestimates `lambda_max`.** Too-small `k` can still
  diverge; G1 uses `n_lanczos >= 4` on a 2-D quadratic where `k=1`
  already suffices — PINN full must use the configured `k`.
- **Falsifier.** Any NaN/Inf on G1 seeds fails the spec. Do not
  weaken to "fewer divergences on average."

## 12. Implementation checklist

- [ ] Public `sharpness_lambda_max` + CubicNewton hook
- [ ] Tests: stiff quadratic, G1 five seeds, G3 parity
- [ ] `benchmarks/sharpness_schedule.py` plus smoke JSON
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
