# 09-19 Jet distillation / jet-SSL

## 1. Thesis and status

A student matches a teacher's **N-jet**, not logits; a self-supervised
variant matches **1-jets across views** (equivariance as a jet
identity).

- **Status**: gated
- **Depends on**: 09-01, 09-02
- **Blocks**: none

### Operator card

- **Benefit.** Distillation / SSL that ordinary autodiff nets cannot
  cheaply state ("match `u_x` too").
- **How it works.** Loss `||J_student - J_teacher||` via `mlp_jet` /
  jet-token state. SSL: two augmentations `g1, g2` with known
  1-jet action; `J(g1·x) = g1_* J(x)`.
- **Strength.** Fitted PINN fields and 1-D profiles.
- **When to use.** First-bet with 09-02.
- **When not.** As ImageNet KD SOTA. Not CCF. Invert-and-match of
  activations is 08-03, not this.
- **Accuracy floor.** Jet order `N`; teacher error.

## 2. Where it lands

`omnibias.{torch,jax}.optim` or a small `distill` helper beside
`jet`. No new package.

## 3. Prior art in omnibias

- `omnibias.{torch,jax}.jet.mlp_jet`, `layer_jet`
- Spec 09-02 — jet-token architecture (the student/teacher body)
- Spec 08-03 — local invert-and-match (different: layer targets, not
  teacher jets)

**Confirmed gap.** No loss that matches two networks' `N`-jets.

## 4. Mathematics

`L = sum_{k=0}^N w_k || D^k u_s - D^k u_t ||^2`. Derivatives from
**bias collapse** / `compose_jet`. SSL uses the chain rule on a known
`g` (again Faà di Bruno). No temperature collapse.

## 5. Worked example

Teacher `u_t = tanh(x)`, student starts at `tanh(0.5 x)`. At `x=0`,
jets: teacher `(0, sech^2(0)=1)`, student `(0, 0.5)`. Order-1 loss
with `w=(1,1)` is `0 + 0.25 = 0.25`. After an exact GN step on the
single scale `a` in `tanh(a x)` to match the 1-jet at `0`, `a=1` and
loss is `0` to `1e-12`.

## 6. Proposed API

Does not exist yet. Bit-identical torch / jax twins; default dtype.

```python
# omnibias/torch/jet_distill.py  (and jax twin) — proposed
@dataclass(frozen=True)
class JetDistillConfig:
    jet_order: int = 1
    weights: tuple = (1.0, 1.0)
    mode: str = "distill"   # distill | ssl

def jet_distill_loss(student, teacher, xs, *, config: JetDistillConfig):
    """Match N-jets. teacher may be stopgrad."""
```

## 7. Practical use cases

1. **Compress** a fitted 1-D PINN teacher into a thinner student.
2. **SSL** on even functions: flip `x -> -x` must flip `u'` .
3. **Not** logit KD on ImageNet.

## 8. Acceptance gates

- **G1.** Worked `tanh` scale recovers `a=1` with loss `< 1e-12`.
- **G2 skill.** Student width 4 vs teacher width 16 on `sin x`:
  joint jet error `< 1e-3` (five seeds) and strictly below
  value-only distillation on the *joint* metric.
- **G3 honesty.** `imagenet_claim` false.
- **G4 parity.** torch / jax bit-identical on G1.

## 9. Benchmark plan

- `benchmarks/jet_distillation.py`
- Smoke: `docs/benchmarks/jet_distillation_smoke.json`
- `--full`: `$OMNIBIAS_SCRATCH/inventions/jet_distill/`

## 10. Honesty and scope

- Bias collapse / Faà di Bruno. Not 08-03. Not stretch.
- Certificate tier: empirical.

## 11. Open questions and risks

- **Teacher noise** in high derivatives. Cap `N=1` for G2.
- **Falsifier.** Value-only distill wins the joint metric.

## 12. Implementation checklist

- [x] Jet-distill helpers
- [x] SSL flip test
- [x] `benchmarks/jet_distillation.py` plus smoke JSON
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
