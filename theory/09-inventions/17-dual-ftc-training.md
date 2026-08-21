# 09-17 Dual-FTC training

## 1. Thesis and status

Train with **two residuals at once**: a derivative residual and the
closed-form **integral** of that residual. They must agree by the
fundamental theorem.

- **Status**: concept
- **Depends on**: 09-01, 09-03
- **Blocks**: none

### Operator card

- **Benefit.** Inconsistency of `u` and `u'` is an optimization
  signal, both kernels closed form.
- **How it works.** `r_D = dI/dx - f`. `r_I = I(x) - I(a) -
  Integral_a^x f`. Loss is `||r_D||^2 + ||r_I||^2`. `I` is an FTC-Net
  cell; `dI/dx` is FTC of that cell (band), not a second network.
- **Strength.** 1-D conservation / antiderivative tasks.
- **When to use.** First-bet with 09-03.
- **When not.** As 02-04 VPINN (test functions). Not CCF.
- **Accuracy floor.** FTC identity; higher-D is 03-06.

## 2. Where it lands

`omnibias.pinn.train` or `omnibias.{torch,jax}.optim` helpers. No new
package.

## 3. Prior art in omnibias

- Spec 09-03 — FTC-Net cell.
- Spec 02-04 — weak form, not dual FTC of the same cell.
- `OperatorBlock` `integral` and `band`.

**Confirmed gap.** No trainer that penalizes FTC inconsistency of one
integral cell.

## 4. Mathematics

FTC: `d/dx (S(x+b_hi)-S(x+b_lo)) = sigma(x+b_hi)-sigma(x+b_lo)`.
If the source `f` is given, both `r_D` and `r_I` are well-defined.
**Window** knob for `I`; **bias collapse** only if a derivative head
is added as a third term (optional, labelled).

No temperature collapse.

## 5. Worked example

`I` from 09-03 at `x=0`, `delta=0.2`. `dI/dx ≈ 0.049958`. If we set
`f = dI/dx` exactly, both residuals are `0` to `1e-12`. If we set
`f=0`, `r_D ≈ 0.049958` and `r_I` over `[0,0]` is `0` (degenerate
interval). G1 uses `f=dI/dx` (zero) and `f=0` (nonzero `r_D`).

## 6. Proposed API

Does not exist yet. Bit-identical torch / jax twins; default dtype.

```python
# omnibias/pinn/torch/train/dual_ftc.py  (and jax twin) — proposed
@dataclass(frozen=True)
class DualFTCConfig:
    w_deriv: float = 1.0
    w_int: float = 1.0

def dual_ftc_loss(I_fn, f_fn, xs, *, config: DualFTCConfig):
    """Returns {loss, r_D, r_I}. I_fn must be an integral cell."""
```

## 7. Practical use cases

1. **Antiderivative of `cos`** with 09-03 (pair of first-bet).
2. **Ablation:** derivative-only vs dual (dual must not increase
   `r_I`).
3. **Not** 2-D VPINN.

## 8. Acceptance gates

- **G1 identity.** When `f = dI/dx`, `max|r_D|` and `max|r_I|` are
  `< 1e-12` on the 09-03 worked cell.
- **G2 skill.** With 09-03 on `dI/dx = cos x`, five seeds: dual loss
  reaches `max|r_D|<1e-4` and `max|r_I|<1e-4`, skill vs `I=0`
  positive, and `max|r_I|` strictly below a derivative-only ablation.
- **G3 honesty.** `claimed_vpinn` is `false`.
- **G4 parity.** torch / jax bit-identical on G1.

## 9. Benchmark plan

- `benchmarks/dual_ftc_training.py`
- Smoke: `docs/benchmarks/dual_ftc_training_smoke.json`
- `--full`: `$OMNIBIAS_SCRATCH/inventions/dual_ftc/`

## 10. Honesty and scope

- Window / FTC, not VPINN, not CCF stretch.
- Certificate tier: empirical.

## 11. Open questions and risks

- **Weighting** `w_int` vs `w_deriv` can hide one residual. Report
  both raw max-norms.
- **Falsifier.** G2: dual does not reduce `r_I`.

## 12. Implementation checklist

- [ ] Dual-FTC loss twins
- [ ] G1 identity test
- [ ] `benchmarks/dual_ftc_training.py` plus smoke JSON
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
