# 09-03 FTC-Net

## 1. Thesis and status

An architecture whose **cell is the `integral` role**
`S(z+b_hi)-S(z+b_lo)` (`S'=sigma`), so the network outputs masses and
fluxes, and derivative heads are the same pack read at bias collapse
`delta -> 0`.

- **Status**: concept
- **Depends on**: 01-01, 09-01
- **Blocks**: 09-17

### Operator card

- **Benefit.** Spends the unused sixth `OperatorBlock` role as the
  hidden cell, not only as a diagnostic.
- **How it works.** Each block evaluates the closed-form antiderivative
  window. A sibling `derivative` head on the same `(w, b_mean)` is the
  founding collapse of that pack.
- **Strength.** Conservation laws and 1-D balances where FTC is the
  identity to check.
- **When to use.** First-bet cluster with 09-17.
- **When not.** As a rewrite of 02-04 VPINN (weak form, test functions)
  or 02-06 BEM-Net (surface potentials). Not CCF.
- **Accuracy floor.** FTC residual on 1-D can hit roundoff; higher-D
  cubature is 03-06, not this cell.

## 2. Where it lands

`omnibias.{torch,jax}.blocks` / `architectures`, reusing
`OperatorBlock(op="integral")`. Fails the package test.

## 3. Prior art in omnibias

- `docs/operator-surface.md` — `integral` is `S(z+b_hi)-S(z+b_lo)`.
- `omnibias.torch.blocks.operator.OperatorBlock` — six roles including
  `integral` and `derivative`.
- Spec 02-04 — exact integrals for polynomial coefficients on boxes.
- Spec 02-06 — `sigma'` as a smoothed Green kernel on surfaces.
- Spec 03-06 — learned quadrature rules (the *model is the rule*).

**Confirmed gap.** No stacked architecture treats `integral` as the
default cell with a paired collapse head.

## 4. Mathematics

For sigmoid, `S = softplus`. One cell:

```
I(x) = S(w.x + b_hi) - S(w.x + b_lo)
```

Founding **bias collapse**: as `delta = b_hi - b_lo -> 0` with mean
`b`, `I / delta -> sigma(w.x + b)` (and higher collapses yield
`sigma^(n)`). The derivative head must be that collapse, not a second
learned pack.

No temperature collapse. Error of a finite `delta` head is the usual
Birkhoff remainder (01-01 / 01-04).

## 5. Worked example

`w=1`, `x=0`, `b_lo=-0.1`, `b_hi=0.1`, sigmoid. `I = softplus(0.1) -
softplus(-0.1)`. `softplus(0.1) ≈ 0.7443967`, `softplus(-0.1) ≈
0.6443967`, `I ≈ 0.1000000`. `I/delta = 0.5`, and `sigmoid(0) = 0.5`.
The implementer must match `|I/delta - sigmoid(0)| < 1e-12` at this
symmetric window (exactly 0.5 in float64) and `|dI/dx - (sigmoid(0.1)
- sigmoid(-0.1))| < 1e-12` by FTC.

## 6. Proposed API

Does not exist yet. Bit-identical torch / jax twins; default dtype.

```python
# omnibias/torch/architectures/ftc_net.py  (and jax twin) — proposed
@dataclass(frozen=True)
class FTCNetConfig:
    n_blocks: int = 2
    width: int = 8
    window: float = 0.2   # b_hi - b_lo; finite gap, not collapse

def ftc_block(x, w, b_lo, b_hi, *, activation: str = "sigmoid"):
    """Closed-form integral cell + collapse head sigma^(0)."""
```

JAX: no Python-side mutation inside `jax.jit`.

## 7. Practical use cases

1. **1-D conservation.** Learn flux `I`; enforce `dI/dx = source` via
   09-17.
2. **Positive masses.** Softplus windows stay non-negative.
3. **Not** 2-D Green kernels (that is 09-14 / 02-06).

## 8. Acceptance gates

- **G1 FTC.** Worked example identities hold at `1e-12`.
- **G2 skill.** On `dI/dx = cos x` with `I(0)=0`, five seeds: FTC-Net
  `max|dI/dx - cos x|` is below `1e-4` and strictly below an
  `identity`-only OMBU of the same width (median). Skill vs `I=0` is
  positive.
- **G3 not VPINN.** Artifact `claimed_weak_form` is `false`.
- **G4 parity.** torch / jax bit-identical on G1.

## 9. Benchmark plan

- `benchmarks/ftc_net.py`
- Smoke: `docs/benchmarks/ftc_net_smoke.json`
- `--full`: `$OMNIBIAS_SCRATCH/inventions/ftc_net/`
- CI smoke after implementation.

## 10. Honesty and scope

- Finite gap is the **window** knob. Collapse heads use founding
  **bias collapse**. No temperature collapse.
- Not 02-04, not 02-06, not 03-06, not CCF stretch.
- Certificate tier: empirical (09-17 may add a consistency residual).

## 11. Open questions and risks

- **Depth.** Stacking integral cells can grow like `x^L`; may need
  normalized windows.
- **Falsifier.** G2 failure means the integral cell is not a better
  flux representation than `identity`.

## 12. Implementation checklist

- [ ] `omnibias.{torch,jax}.architectures` FTC-Net
- [ ] Tests of the worked FTC identities
- [ ] `benchmarks/ftc_net.py` plus smoke JSON
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
