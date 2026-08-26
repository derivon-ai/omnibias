# 09-21 Exact score matching

## 1. Thesis and status

Hyvärinen / denoising score matching on an **OMBU score** whose
**Hessian trace is closed form**. This is a *training objective*, not
a claim that omnibias lacks divergence.

- **Status**: shipped (G1–G4 CI; G4 torch/jax parity on G1; CNF exact div is prior art)
- **Depends on**: 09-01
- **Blocks**: none

### Operator card

- **Benefit.** Score matching without Hutchinson on the tower family.
- **How it works.** Score `s_theta(x) = grad log p_theta` is an OMBU
  field. Hyvärinen: `E[ ||s||^2 /2 + div s ]`. `div s` and
  `Tr Hess log p` use exact field / tower operators.
- **Strength.** Low-dimensional densities on the `sigma` family.
- **When to use.** After Jet-Flow / score users want DSM.
- **When not.** As "we invented exact CNF divergence" — that is
  already `integrate_cnf`. Not ImageNet diffusion SOTA. Not CCF.
- **Accuracy floor.** Hutchinson-free only on tower scores.

## 2. Where it lands

`omnibias.score` / `omnibias.score.flow`. No new package.

## 3. Prior art in omnibias

- `omnibias.score.flow.{torch,jax}.ops.cnf.cnf_dynamics`,
  `integrate_cnf` — exact field `div` for CNF
- `omnibias.score` Fokker–Planck / `hutchinson_trace_jacobian` (exists
  as an estimator with variance)
- `omnibias.fields` grad / hessian / div

**Confirmed gap.** No Hyvärinen / DSM loss on an OMBU score with exact
`Tr H`. CNF exact `div` is **not** this objective.

## 4. Mathematics

Hyvärinen score matching: minimize
`E_p [ 1/2 ||s_theta(x)||^2 + div s_theta(x) ]`. For
`s = grad sigma(w·x+b)` (simple case), `div s` is a closed-form
`sigma''` contraction (**bias collapse** order 2). No temperature
collapse.

## 5. Worked example

1-D standard logistic score of `sigmoid` location model is not the
unit Gaussian. Use `s(x) = -x` (Gaussian N(0,1) score). Then
`1/2 s^2 + s' = 1/2 x^2 - 1`. At `x=1`, value `-0.5`. An OMBU that
*represents* `-x` (linear, no `sigma`) must match. G1 is: exact
`div` of `s(x)=-x` equals `-1` to `1e-12`. G2 uses a 1-D Gaussian
dataset and an OMBU score.

## 6. Proposed API

Does not exist yet. Bit-identical torch / jax twins; default dtype.

```python
# omnibias/score/torch/score_matching.py  (and jax twin) — proposed
@dataclass(frozen=True)
class ExactSMConfig:
    kind: str = "hyvarinen"   # hyvarinen | dsm

def exact_score_matching_loss(score_fn, xs, *, config: ExactSMConfig):
    """div / Hessian trace from tower or field ops, not Hutchinson."""
```

## 7. Practical use cases

1. **1-D Gaussian** DSM vs Hutchinson (variance must be 0 for exact).
2. **Ablation** vs CNF NLL on the same 2-D mixture (no required win).
3. **Not** latent diffusion SOTA.

## 8. Acceptance gates

- **G1.** `div(-x) = -1` to `1e-12` (1-D).
- **G2 skill.** 1-D N(0,1) samples, five seeds: Hyvärinen loss of a
  trained OMBU score is below the zero-score baseline (`div 0 + 0`)
  and `|mean s(x) + x|` `< 0.2` on a probe. Hutchinson single-probe
  variance on the same `div` is recorded and is `0` for the exact
  path.
- **G3 honesty.** Artifact `cnf_div_claimed_new` is `false`.
- **G4 parity.** torch / jax bit-identical on G1.

## 9. Benchmark plan

- `benchmarks/exact_score_matching.py`
- Smoke: `docs/benchmarks/exact_score_matching_smoke.json`
- `--full`: `$OMNIBIAS_SCRATCH/inventions/score_matching/`

## 10. Honesty and scope

- Exact CNF `div` is prior art. This spec is the SM objective.
- Not stretch. Not ImageNet.
- Certificate tier: empirical.

## 11. Open questions and risks

- **OMBU score** may be a poor Gaussian score. G2 uses a weak
  absolute (`0.2`), not SOTA NLL.
- **Falsifier.** Exact path has nonzero Hutchinson-like variance
  (implementation bug).

## 12. Implementation checklist

- [x] Exact SM loss twins
- [x] Zero-variance vs Hutchinson test
- [x] `benchmarks/exact_score_matching.py` plus smoke JSON
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
