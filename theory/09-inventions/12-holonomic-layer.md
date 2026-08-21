# 09-12 Holonomic layer

## 1. Thesis and status

A block that **is an Ore annihilator**: representable maps are D-finite
by construction, so discovery is "read the annihilator," not SINDy on
a library soup.

- **Status**: gated
- **Depends on**: 01-10, 09-01
- **Blocks**: 09-26

### Operator card

- **Benefit.** Hypothesis class equals a linear ODE / recurrence.
- **How it works.** The layer stores an `OrePolynomial` (or
  `PRecursive`). Forward evaluates the unique D-finite series (or
  jet) annihilated by that operator, with initial jet data as
  activations.
- **Strength.** Special functions, holonomic identities, 1-D linear
  ODEs.
- **When to use.** After first-bet; pair with 09-26 export.
- **When not.** As a general PINN. Nonlinear PDEs are not D-finite
  in this sense. Not CCF.
- **Accuracy floor.** The D-finite class only.

## 2. Where it lands

`omnibias.holonomic` (already the engine). Thin torch / jax eval of
jets. No new package.

## 3. Prior art in omnibias

- `omnibias.holonomic._core.dfinite.DFinite`, `PRecursive`,
  `_fit_annihilator`
- `omnibias.holonomic._core.oreops.OrePolynomial`, `gcrd`, `lclm`
- `omnibias.symbolic` — library-free SINDy / jet discovery (different
  class)
- Spec 03-11 — Lie symmetry search (jets as a linear solve)

**Confirmed gap.** No layer whose weights *are* Ore coefficients and
whose forward is the D-finite jet.

## 4. Mathematics

An operator `L = sum_{k=0}^m p_k(x) D^k` annihilates `u` when
`L u = 0`. Given a jet of order `m-1` at a point, the higher jet is
determined. Training fits `p_k` (rational / polynomial) so a data jet
is annihilated. **Bias collapse** supplies the data jet of a `sigma`
net if the layer sits on top of one; the holonomic block itself is
not a collapse.

No temperature collapse.

## 5. Worked example

`exp` is annihilated by `D-1`. Jet at `0`: `(1, 1, 1, ...)`. A
holonomic layer with `L = D-1` and `u(0)=1` must produce
`u^{(n)}(0)=1` for `n <= 5` to `1e-12` (exact rationals: all `1`).
`sin` uses `D^2+1` with `(0,1)` at `0`; `u^{(2)}(0)=0`,
`u^{(3)}(0)=-1`.

## 6. Proposed API

Does not exist yet. Core stays pure Python. Adapters: default dtype.

```python
# omnibias/holonomic/_core/layer.py — proposed
@dataclass(frozen=True)
class HolonomicLayerSpec:
    max_order: int = 2
    max_degree: int = 1

def holonomic_jet(op: OrePolynomial, init_jet, x0: float, order: int):
    """Prolong the jet using L u = 0. Must not import torch/jax."""
```

## 7. Practical use cases

1. **Recover `D-1`** from samples of `exp` (fit annihilator).
2. **Constrain a PINN** of `u''+u=0` to the holonomic class.
3. **Not** Burgers as a D-finite claim.

## 8. Acceptance gates

- **G1 exp.** Worked `D-1` jet matches to `1e-12`.
- **G2 skill.** Fit on 16 samples of `sin` on `[0,1]`: recovered
  operator, after content-clearing, is a rational multiple of `D^2+1`
  (coefficient test) **or** max jet residual of `L u` `< 1e-8`.
- **G3 honesty.** `nonlinear_pde_claimed_dfinite` is `false`.
- **G4 purity.** Core layer imports neither torch nor jax.

## 9. Benchmark plan

- `benchmarks/holonomic_layer.py`
- Smoke: `docs/benchmarks/holonomic_layer_smoke.json`
- `--full`: `$OMNIBIAS_SCRATCH/inventions/holonomic_layer/`

## 10. Honesty and scope

- Finite / D-finite class only. Lean flags are 09-26, not asserted
  here. Not CCF. Not NS.
- Certificate tier: empirical + exact rational linear algebra already
  in holonomic.

## 11. Open questions and risks

- **Degree explosion** of `p_k`. Cap `max_degree`.
- **Falsifier.** G2 cannot recover `sin`'s annihilator.

## 12. Implementation checklist

- [x] `omnibias.holonomic` layer helpers
- [x] Exp / sin jet tests
- [x] `benchmarks/holonomic_layer.py` plus smoke JSON
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
