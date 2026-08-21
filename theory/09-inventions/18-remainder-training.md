# 09-18 Remainder training

## 1. Thesis and status

The **loss is the Taylor remainder after order `N`** (or a Padé
remainder), not `L` itself, so the net is forced into "this function
is well approximated by its N-jet."

- **Status**: gated
- **Depends on**: 03-10, 03-13, 09-01
- **Blocks**: none

### Operator card

- **Benefit.** A trainer that spends 03-10 as a *loss*, and may
  trigger 03-13 birth when the remainder stays large.
- **How it works.** From `mlp_jet` / TM, form `R_N(x) = f(x) -
  T_N(x)`. Minimize `||R_N||` on a probe set (and optionally a Padé
  remainder). If `||R_N||` plateaus, call pack birth (03-13).
- **Strength.** Analytic targets (exp, rational) and PINN fields that
  should be entire / meromorphic.
- **When to use.** First-bet after Pack-MoE.
- **When not.** As a rewrite of 03-10 (diagnostic) or 03-13
  (architecture search). Not CCF stretch.
- **Accuracy floor.** `R_N` of the *model*, not of the unknown PDE
  solution.

## 2. Where it lands

`omnibias.{torch,jax}.optim` helpers. No new package.

## 3. Prior art in omnibias

- Spec 03-10 — jet–Padé *locates* poles; not a training loss.
- Spec 03-13 — residual-driven pack birth/death.
- `omnibias.{torch,jax}.jet.mlp_jet`
- `omnibias.core.verified.TaylorModel` remainder interval

**Confirmed gap.** Nothing minimizes `R_N` as the objective.

## 4. Mathematics

`f(x) = T_N(x; x0) + R_N(x)`. Minimize `sum |R_N(x_i)|^2`. Jets of
`T_N` use **bias collapse**. A Padé remainder is 03-10's object used
as a scalar loss. No temperature collapse.

## 5. Worked example

`f(x)=exp(x)` about `0`, `N=2`. `T_2 = 1+x+x^2/2`. At `x=0.2`,
`exp(0.2)≈1.221403`, `T_2=1.22`, `R_2≈0.001403`. A remainder-training
step on a model that *is* `T_2` has loss `≈1.97e-6` at that point.
The implementer matches `R_2` to `1e-12` for the analytic `exp`.

## 6. Proposed API

Does not exist yet. Bit-identical torch / jax twins; default dtype.

```python
# omnibias/torch/optim_remainder.py  (and jax twin) — proposed
@dataclass(frozen=True)
class RemainderTrainConfig:
    jet_order: int = 2
    birth_threshold: float = 1e-3
    use_pade: bool = False

def remainder_loss(model, xs, x0, *, config: RemainderTrainConfig):
    """||f - T_N||. Optional 03-13 hook if loss > birth_threshold."""
```

## 7. Practical use cases

1. **Fit `exp`** with increasing `N` (remainder must drop).
2. **Trigger 03-13** when remainder plateaus on a two-scale target.
3. **Not** CCF Hilbert remainder as stretch.

## 8. Acceptance gates

- **G1 analytic.** Worked `exp` `R_2(0.2)` matches to `1e-12`.
- **G2 skill.** Train an OMBU to minimize `R_2` of its own `exp`
  target on `[-0.3,0.3]`: post-train `max|R_2| < 1e-4` (five seeds)
  and strictly below a value-only MSE ablation of the same budget
  *on the remainder metric*.
- **G3 honesty.** Does not call 03-10 or 03-13 "this spec."
  `stretch_claim` false.
- **G4 parity.** torch / jax bit-identical on G1.

## 9. Benchmark plan

- `benchmarks/remainder_training.py`
- Smoke: `docs/benchmarks/remainder_training_smoke.json`
- `--full`: `$OMNIBIAS_SCRATCH/inventions/remainder/`

## 10. Honesty and scope

- Bias collapse for jets. Not temperature collapse. Not stretch.
- Certificate tier: empirical (optional TM width as a sound bound).

## 11. Open questions and risks

- **Gaming:** a constant model has a large `R_0` but a small higher
  remainder if `T_N` fits `f`. Use a value probe plus `R_N`.
- **Falsifier.** G2: MSE ablation already has smaller `R_N`.

## 12. Implementation checklist

- [x] Remainder-loss helpers
- [x] Optional 03-13 hook
- [x] `benchmarks/remainder_training.py` plus smoke JSON
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
