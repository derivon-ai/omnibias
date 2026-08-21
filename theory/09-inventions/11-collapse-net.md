# 09-11 Collapse-Net

## 1. Thesis and status

Train on a lattice with umbral / irregular Birkhoff stencils (01-04)
and **infer by founding bias collapse** `delta -> 0` to `sigma^(K-1)`,
so the continuum limit is a named map, not a hope.

- **Status**: gated
- **Depends on**: 01-04, 09-01
- **Blocks**: none

### Operator card

- **Benefit.** Discrete-first training with a certified collapse at
  inference.
- **How it works.** Forward during train uses exact-`Q` stencil weights
  from `omnibias.difference`. At eval, replace the stencil with the
  collapsed `sigma^(K-1)` fastpath at the pack mean.
- **Strength.** Tasks that are naturally discrete (grids) but evaluated
  as smooth fields.
- **When to use.** After 01-04 consumers want a net, not only weights.
- **When not.** As a claim that the discrete net *is* the continuum
  PDE. Not CCF.
- **Accuracy floor.** 01-04 stencil exactness; collapse remainder is
  the Birkhoff remainder.

## 2. Where it lands

`omnibias.difference` + thin `omnibias.{torch,jax}.architectures`
wrappers. No new package.

## 3. Prior art in omnibias

- Spec 01-04 — exact-`Q` weights in `omnibias.difference` (G1–G4 earned).
- `omnibias.core.polynomials` — collapse target `sigma^(n)`.
- Spec 01-01 — multipack collapse (continuous packs, not lattice-first).

**Confirmed gap.** No architecture that trains stencil and swaps to
collapse at inference with a measured remainder.

## 4. Mathematics

A width-`K` stencil with spread `delta` reproduces `sigma^(K-1)` as
`delta -> 0` (founding **bias collapse**). Train loss uses the stencil
output on grid nodes. Inference output is the fastpath. The remainder
`R(delta)` is recorded.

No temperature collapse.

## 5. Worked example

Centered first difference of `sigmoid` at `0` with `delta=0.1`:
`(sigmoid(0.1)-sigmoid(-0.1))/0.2 ≈ 0.249791`. Collapse target
`sigmoid'(0)=0.25`. Remainder `≈ 2.09e-4`. The net's
`collapse_remainder` field must match this to relative `1e-6` when
the layer is that stencil.

## 6. Proposed API

Does not exist yet. Bit-identical torch / jax twins; default dtype.

```python
# omnibias/torch/architectures/collapse_net.py  (and jax twin) — proposed
@dataclass(frozen=True)
class CollapseNetConfig:
    order: int = 1
    delta: float = 0.1
    mode: str = "stencil"   # stencil | collapsed

def collapse_net_forward(x, params, *, config: CollapseNetConfig):
    """mode='collapsed' uses sigma^(order) fastpath."""
```

## 7. Practical use cases

1. **Grid derivative** matching 01-04 G-gates, then collapse eval.
2. **1-D PINN** trained on a coarse grid, evaluated collapsed.
3. **Not** a continuum NS claim.

## 8. Acceptance gates

- **G1 remainder.** Worked example remainder matches to relative
  `1e-6`.
- **G2 skill.** On `d/dx sin x` at 33 nodes, stencil train then
  collapsed eval: `max|error| < 1e-3` and skill vs zero positive.
  Collapsed eval must be no worse than stencil eval on a denser probe
  (the point of collapse).
- **G3 honesty.** `continuum_pde_claimed` is `false`.
- **G4 parity.** torch / jax bit-identical on G1.

## 9. Benchmark plan

- `benchmarks/collapse_net.py`
- Smoke: `docs/benchmarks/collapse_net_smoke.json`
- `--full`: `$OMNIBIAS_SCRATCH/inventions/collapse_net/`

## 10. Honesty and scope

- The only collapse is founding **bias collapse**. Not temperature
  collapse. Not a continuum theorem. Not CCF.
- Certificate tier: empirical + 01-04 stencil exactness.

## 11. Open questions and risks

- **Train/eval mismatch** can hurt if `delta` is large. G2 uses a
  named `delta`.
- **Falsifier.** Collapsed eval worse than stencil on the dense probe.

## 12. Implementation checklist

- [x] Collapse-Net wrappers on `omnibias.difference`
- [x] Remainder regression test
- [x] `benchmarks/collapse_net.py` plus smoke JSON
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
