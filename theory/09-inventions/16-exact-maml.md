# 09-16 Exact MAML

## 1. Thesis and status

Model-agnostic meta-learning whose **inner step uses exact HVPs /
Gauss–Newton** and whose **meta-gradient is IFT through the inner KKT
system**, not a finite-difference unrolling of Adam.

- **Status**: shipped (G1–G4 CI; G4 torch/jax parity on G1; not ImageNet)
- **Depends on**: 08-01, 09-01
- **Blocks**: none

### Operator card

- **Benefit.** Few-shot PINN adaptation that is actually Newton.
- **How it works.** Inner loss `L_task(theta)`; inner update is
  `GaussNewton` / `hvp` Newton. Meta-grad `d theta'/d phi` from the
  implicit function theorem on the inner stationarity condition,
  matching the convex KKT pattern.
- **Strength.** Families of 1-D Poisson / Helmholtz with a varying
  source.
- **When to use.** First-bet cluster.
- **When not.** As ImageNet few-shot SOTA. Not CCF stretch. Not "we
  skip the chain rule."
- **Accuracy floor.** Same operator floor as the inner task.

## 2. Where it lands

`omnibias.{torch,jax}.optim`. Fails the package test.

## 3. Prior art in omnibias

- `omnibias.{torch,jax}.optim.GaussNewton`, `hvp`,
  `gauss_newton_direction`
- `omnibias.convex.{torch,jax}.layer` — KKT implicit-function
  gradients for LP/QP
- Spec 08-08 — IFT for DEQ, not a meta-loop

**Confirmed gap.** No MAML loop with exact inner curvature and IFT
meta-grad.

## 4. Mathematics

Inner: `G(theta; D_i) = 0` (GN residual or `grad L = 0`). Then
`d theta / d phi = - (dG/d theta)^{-1} (dG/d phi)`. `dG/d theta`
uses exact HVPs (**bias collapse** `sigma''` / GN `J^T J`). This
**is** the chain rule (IFT). No temperature collapse.

## 5. Worked example

Quadratic `L = 0.5 (theta - alpha)^2` with inner exact Newton:
`theta' = alpha` in one step. Meta-loss `0.5 (theta' - alpha_star)^2`
has gradient `0` when `alpha` is a parameter that Newton already
sets to the task target. Implementer: one-step inner Newton on this
quadratic reaches `|theta'-alpha|<1e-12`; IFT meta-grad matches
autodiff through the closed-form step to `1e-10`.

## 6. Proposed API

Does not exist yet. Bit-identical torch / jax twins; default dtype.

```python
# omnibias/torch/optim_maml.py  (and jax twin) — proposed
@dataclass(frozen=True)
class ExactMAMLConfig:
    inner_steps: int = 1
    solver: str = "gn"   # gn | newton_hvp

def exact_maml_meta_step(tasks, params, *, config: ExactMAMLConfig):
    """Inner exact GN; meta-grad via IFT. No Adam inner default."""
```

JAX: inner loop is `lax.scan`; IFT is one linear solve.

## 7. Practical use cases

1. **Poisson family** `u'' = s_i(x)` with five source tasks; adapt in
   one GN step.
2. **Compare** to five inner Adam steps (must win iso-inner-eval or
   G2 fails).
3. **Not** MiniImageNet.

## 8. Acceptance gates

- **G1 IFT.** Quadratic worked example: inner error `< 1e-12`, IFT vs
  closed-form meta-grad `< 1e-10`.
- **G2 skill.** 1-D Poisson family, five seeds: post-adapt residual
  `< 1e-4` and strictly below 5-step inner Adam (median). Skill vs
  `u=0` positive.
- **G3 honesty.** `imagenet_claim` false; `stretch_claim` false.
- **G4 parity.** torch / jax bit-identical on G1.

## 9. Benchmark plan

- `benchmarks/exact_maml.py`
- Smoke: `docs/benchmarks/exact_maml_smoke.json`
- `--full`: `$OMNIBIAS_SCRATCH/inventions/exact_maml/`

## 10. Honesty and scope

- IFT is the chain rule. Not a global min. Not CCF stretch.
- Certificate tier: empirical.

## 11. Open questions and risks

- **Singular inner `J`.** Damping required; record `mu`.
- **Falsifier.** G2: Adam inner wins iso-eval.

## 12. Implementation checklist

- [x] `omnibias.{torch,jax}.optim` exact MAML
- [x] Quadratic IFT test
- [x] `benchmarks/exact_maml.py` plus smoke JSON
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
