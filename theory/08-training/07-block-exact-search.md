# 08-07 Block / coordinate exact search

## 1. Thesis and status

Along one structured block (OMBU channel, last linear layer, or
arrangement normals), the directional loss `phi(s)` is a low-degree
combination of the Riccati tower, so the step is a **polynomial root**
— a structured special case of spec 03-12.

- **Status**: gated
- **Depends on**: 01-01, 03-12, 08-01
- **Blocks**: none

### Operator card

- **Benefit.** Exact (to jet order `N`) minimization on a coordinate or
  small block without backtracking, cheap because the direction is
  sparse.
- **How it works.** Freeze all params except one block. Build `phi(s)`
  with `mlp_jet` / a single-channel OMBU tower. Root `phi'` (companion
  or certified isolation). Apply `verify=True` from 03-12.
- **Strength.** Last-layer least squares, one OMBU bias, arrangement
  `H=2` polish.
- **When to use.** After a global GN step, as a cheap coordinate sweep;
  or inside `tab` / arrangement fit.
- **When not.** As the only trainer of a deep net; when the block is
  not low-dimensional (then use 03-12 on a dense direction).
- **Accuracy floor.** Block structure. Truncation `R_N` if `N` is too
  small for a non-polynomial `phi` (tanh nest is entire but not
  polynomial — the jet is still a model).

## 2. Where it lands

`omnibias.{torch,jax}.optim` helpers plus optional calls from
`omnibias.tab.torch.arrangement.fit_arrangement` / last-layer polish.
Not a package. Shares 03-12's line-search module once that exists.

## 3. Prior art in omnibias

- Spec 03-12 — general directional jet line search.
- `omnibias.{torch,jax}.optim.taylor_line_min` — univariate Taylor
  minimizer, not certified-radius and not block-aware.
- `omnibias.torch.unit` / OMBU — one channel, closed-form tower.
- `omnibias.tab.torch.arrangement.ArrangementClassifier` — `H` normals.

**Confirmed gap.** No coordinate-block driver that picks OMBU / last
linear / arrangement directions and reuses 03-12's contract
(`verify=True`, never-worse).

## 4. Mathematics

If only a last-layer vector `v` moves, `u(x) = v · h(x)` is linear in
`v`, and a squared residual is **quadratic** in the step `s` along a
line `v + s d`. Then `phi` is exactly degree 2, `N = 2` is exact,
`R_N = 0`.

If an OMBU bias `b` moves, `phi(s)` is a finite combination of
`sigma^(n)(z + b + s)` via the tower (bias collapse). A jet of order
`N` is the Taylor model of that path; 03-12's certified radius applies.

Arrangement normals: each coordinate of `W` enters `z = W x - t`
linearly; `phi` is again a nest of `sigma` and is handled as a 03-12
jet, not as a claim of exact polynomials.

No temperature collapse unless the arrangement caller is annealing
`beta` (05-02); this spec's step is at fixed `beta`.

## 5. Worked example

**Last-layer least squares, exact.** Hidden features
`h = (1.0, 0.5)`, target `y = 1.0`, `v = (0.0, 0.0)`, direction
`d = (1.0, 0.0)`, `phi(s) = (s * 1.0 - 1.0)^2 = (s-1)^2`.
`phi'(s) = 2(s-1) = 0` ⇒ `s* = 1`. Model value `0`. One root, exact.
The implementer asserts `s*` equals `1` to `1e-14` and
`fell_back is False`.

## 6. Proposed API

Gated. Shared algebra in `omnibias.core.block_search`; tensor twins in
`omnibias.{torch,jax}.optim_block_search`, re-exported from
`omnibias.{torch,jax}.optim`. Reuses 03-12 `jet_line_search` with
`verify=True`.

```python
from omnibias.torch.optim_block_search import (
    block_exact_search,
    last_linear_block,
)

new, result = block_exact_search(
    loss_fn, params, spec=last_linear_block(width), exact_quadratic=True
)
```

Bit-identical twins; default dtype; `mask` is boolean over flat
`params`, or a named `BlockSpec` (`last_linear` / `ombu_bias` /
`arrangement_W`).

## 7. Practical use cases

1. **1-hidden OMBU binary** — G1 vs 20 Adam steps on that block.
2. **Arrangement H=2 polish** after `fit_arrangement`.
3. **Last-layer closed-form** on a PINN decode (quadratic residual).

## 8. Acceptance gates

- **G1 block win.** On a 1-hidden OMBU binary problem (or last-layer
  least squares with frozen hidden): one coordinate / block sweep
  reaches loss `<=` 20 Adam steps on the **same block** (hidden frozen).
  Five seeds. Skill: loss beats the majority-class / zero predictor.
- **G2 exact quadratic.** Last-layer LS example in section 5:
  `s* = 1` to `1e-12`.
- **G3 never-worse.** `verify=True` default; 100% of a randomized
  block suite do not increase loss.
- **G4 parity.** torch / jax on G2.

## 9. Benchmark plan

- `benchmarks/block_exact_search.py`
- Smoke: `docs/benchmarks/block_exact_search_smoke.json`
- `--full`: `$OMNIBIAS_SCRATCH/training/block_search/`
- CI smoke after implementation.

## 10. Honesty and scope

- Special case of 03-12, not a global solver.
- Bias collapse for the tower; arrangement `beta` is caller-owned.
- Does not clear CCF stretch.

## 11. Open questions and risks

- **Coordinate cycling** can chatter. G1 is one sweep, not a claim of
  a full Gauss–Seidel optimum.
- If 03-12 G4 (step-count vs Wolfe) fails globally, this spec can
  still pass G2 (exact quadratic) and remain useful.
- **Falsifier.** G1 failure: Adam on the block is enough; keep G2 as
  a unit test and mark the OMBU claim unearned.

## 12. Implementation checklist

- [x] `block_exact_search` reusing 03-12 / `taylor_line_min` until 03-12 ships
- [x] Named blocks for OMBU / last linear / arrangement
- [x] Tests: G2 exact, G3 never-worse, G4 parity
- [x] `benchmarks/block_exact_search.py` plus smoke JSON
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
