# 09-07 Pack-MoE

## 1. Thesis and status

A mixture whose **experts are OMBU packs** (order / window / position)
and whose **router is slab mass** (`integral` or `band`), not a free
softmax MLP.

- **Status**: concept
- **Depends on**: 01-01, 04-02, 09-01, 09-03
- **Blocks**: none

### Operator card

- **Benefit.** The four knobs *are* the experts; the gate is a
  calibrated mass (04-02).
- **How it works.** Expert `e` is a pack with a named order, window,
  and scan position. Gate `g_e(x) = I_e(x) / sum_j I_j(x)` with `I_e`
  an `integral` (or `band`) cell. Output is `sum g_e expert_e(x)`.
- **Strength.** Tabular / 1-D PINN mixtures with an honest router.
- **When to use.** After 09-03; first-bet cluster.
- **When not.** As a generic MoE on ImageNet. Softmax routers are
  forbidden as the default (optional ablation only).
- **Accuracy floor.** LightGBM remains the tabular baseline (05-02).
  Temperature collapse of the router (`beta -> inf`) is named if used.

## 2. Where it lands

`omnibias.{torch,jax}.architectures`. Fails the package test.

## 3. Prior art in omnibias

- Spec 01-01 — `MultiPackUnit`.
- Spec 04-02 — slab masses as calibrated probabilities.
- Spec 05-02 — tabular arrangements; LightGBM comparison.
- `OperatorBlock` `integral` / `band`.

**Confirmed gap.** No MoE whose router *is* slab mass and whose experts
are packs with distinct knobs.

## 4. Mathematics

```
I_e(x) = S(w_e.x + b_hi,e) - S(w_e.x + b_lo,e)    # finite gap
g_e(x) = I_e(x) / (sum_j I_j(x) + eps)
y(x) = sum_e g_e(x) F_e(x)
```

`F_e` may be identity, a collapse head (`delta -> 0`), or a scan
(01-02). The router uses the **window** knob. If gates are sharpened
by a `beta` on `g_e^beta`, that is **temperature collapse** and must
be labelled; default `beta = 1`.

## 5. Worked example

Two experts, `x=0`, sigmoid, expert A window `[-0.2, 0.0]`, expert B
`[0.0, 0.2]`. `I_A = softplus(0)-softplus(-0.2)`, `I_B =
softplus(0.2)-softplus(0)`. These are equal by symmetry
(`≈ 0.09966799` each after evaluating), so `g_A = g_B = 0.5`. If
`F_A=1`, `F_B=3`, then `y=2`. The implementer matches `g` to `1e-12`.

## 6. Proposed API

Does not exist yet. Bit-identical torch / jax twins; default dtype.

```python
# omnibias/torch/architectures/pack_moe.py  (and jax twin) — proposed
@dataclass(frozen=True)
class PackMoEConfig:
    n_experts: int = 4
    router: str = "integral"   # integral | band; not softmax
    beta: float = 1.0          # >1 is temperature collapse

def pack_moe_forward(x, packs, *, config: PackMoEConfig):
    """Raises ValueError if router == 'softmax' unless allow_softmax."""
```

## 7. Practical use cases

1. **1-D piecewise profile** (two frequencies): each expert a different
   pack order (01-07).
2. **Tabular smoke** vs a single pack and vs LightGBM (must not claim
   a 05-02 G3 reversal without re-running that suite).
3. **Not** language-model MoE.

## 8. Acceptance gates

- **G1 router.** Worked example gates are `0.5` to `1e-12`.
- **G2 skill.** On a named 1-D two-tone `sin(x)+0.3 sin(4x)`, five
  seeds: Pack-MoE RMSE strictly below a single-pack OMBU of the same
  total width (median) and below `1e-2`. Skill vs zero is positive.
- **G3 honesty.** `router_is_softmax` default `false`;
  `temperature_collapse_used` recorded.
- **G4 parity.** torch / jax bit-identical on G1.

## 9. Benchmark plan

- `benchmarks/pack_moe.py`
- Smoke: `docs/benchmarks/pack_moe_smoke.json`
- `--full`: `$OMNIBIAS_SCRATCH/inventions/pack_moe/`

## 10. Honesty and scope

- Router gap is the **window** knob. Expert collapse heads use
  **bias collapse**. Optional `beta -> inf` is **temperature collapse**.
- Not a 05-02 G3 claim. Not ImageNet. Not CCF.
- Certificate tier: empirical; calibration defers to 04-02.

## 11. Open questions and risks

- **Collapsed router.** If all windows overlap, gates go uniform and
  G2 should fail.
- **Falsifier.** G2 failure: knob-experts add nothing over one pack.

## 12. Implementation checklist

- [ ] `omnibias.{torch,jax}.architectures` Pack-MoE
- [ ] Softmax-router raise test
- [ ] `benchmarks/pack_moe.py` plus smoke JSON
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
