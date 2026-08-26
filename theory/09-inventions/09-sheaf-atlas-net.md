# 09-09 Sheaf-atlas net

## 1. Thesis and status

Partition cells plus **jet-valued transition maps**, with a training
loss that includes the **cocycle residual** "chart `i -> j -> k` equals
`i -> k` to order `N`."

- **Status**: shipped (G1–G4 CI; G4 torch/jax parity on G1; not a sheaf theorem)
- **Depends on**: 01-03, 01-10, 09-01
- **Blocks**: none

### Operator card

- **Benefit.** Atlas consistency is a residual, not a hope.
- **How it works.** `omnibias.partition` weights route into cells.
  Each overlap carries a jet transition `Phi_ji`. Loss =
  task + `||Phi_kj compose Phi_ji - Phi_ki||` on jets.
- **Strength.** Piecewise metrics / piecewise PINNs with named charts.
- **When to use.** After jet-token / partition users need overlaps.
- **When not.** As a rewrite of `AtlasSpec` metrics-only blending.
  Temperature collapse of the partition (`beta -> inf`) is named if
  used.
- **Accuracy floor.** Overlap quality; empty overlaps make the cocycle
  vacuous.

## 2. Where it lands

`omnibias.geometry.atlas` + `omnibias.partition`. No new package.

## 3. Prior art in omnibias

- `omnibias.geometry.atlas.AtlasSpec`, `blended_metric`,
  `atlas_manifold` — region-wise metrics, no cocycle.
- `omnibias.partition` — `partition_weights`, temperature collapse.
- Spec 01-03 — arrangement geometry.
- `omnibias.{torch,jax}.jet.compose_jet`.

**Confirmed gap.** No cocycle residual on jet transition maps.

## 4. Mathematics

On a triple overlap, jets satisfy

```
compose_jet(J_i, tower_kj o tower_ji) = compose_jet(J_i, tower_ki)
```

to order `N` (Faà di Bruno). Partition gates may harden as
`beta -> inf` (**temperature collapse**); default training uses finite
`beta`. Chart maps themselves use **bias collapse** for `sigma^(n)`.

## 5. Worked example

Two charts on `R`: `Phi_21(x) = 2x`, `Phi_32(y) = y/2`, so
`Phi_31 = id`. Order-1 jet of `id` is `(x, 1)`. Composition of
`(2x, 2)` then `(y/2, 1/2)` is `(x, 1)`. Cocycle residual is `0` to
`1e-12`. A buggy implementation that multiplies values but not
derivatives leaves residual `1` in the 1-jet — G1 catches it.

## 6. Proposed API

Does not exist yet. Bit-identical torch / jax twins; default dtype.

```python
# omnibias/geometry/atlas/cocycle.py — proposed (core) + torch/jax
@dataclass(frozen=True)
class SheafAtlasConfig:
    jet_order: int = 1
    beta: float = 1.0

def cocycle_residual(transitions, jets, *, config: SheafAtlasConfig):
    """Max jet-norm of Phi_kj o Phi_ji - Phi_ki on named triples."""
```

## 7. Practical use cases

1. **Two-chart 1-D metric** already in `blended_metric` plus a cocycle
   head that must stay `< 1e-8` when transitions are inverses.
2. **Piecewise PINN** on two intervals with a known overlap.
3. **Not** a manifold learning SOTA claim.

## 8. Acceptance gates

- **G1 algebra.** Worked example residual `< 1e-12`.
- **G2 skill.** On a two-interval Poisson with known overlap, five
  seeds: task residual skill > 0 vs `u=0` **and** mean cocycle
  residual `< 1e-6`. If cocycle stays `> 1e-3` while task fits, G2
  fails.
- **G3 honesty.** `temperature_collapse_used` recorded; `beta -> inf`
  is not called bias collapse.
- **G4 parity.** torch / jax bit-identical on G1.

## 9. Benchmark plan

- `benchmarks/sheaf_atlas_net.py`
- Smoke: `docs/benchmarks/sheaf_atlas_net_smoke.json`
- `--full`: `$OMNIBIAS_SCRATCH/inventions/sheaf_atlas/`

## 10. Honesty and scope

- Two collapses named separately. Not CCF. Not a sheaf-cohomology
  theorem. Not P vs NP on arrangements.
- Certificate tier: empirical.

## 11. Open questions and risks

- **Empty overlap** makes G2 vacuous; the benchmark must force a
  nonempty overlap.
- **Falsifier.** G2: task fits by ignoring the cocycle.

## 12. Implementation checklist

- [x] Cocycle helpers in `omnibias.geometry.atlas`
- [x] Partition-weight tests
- [x] `benchmarks/sheaf_atlas_net.py` plus smoke JSON
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
