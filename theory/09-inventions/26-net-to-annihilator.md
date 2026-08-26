# 09-26 Net-to-annihilator export

## 1. Thesis and status

After training, **emit an Ore / jet description** plus a *finite
rational* Lean obligation — an export format, not a better loss.
Infinite analytic obligations stay out of Lean.

- **Status**: shipped (G1–G4 CI; G4 export purity; Lean flags future-earned)
- **Depends on**: 01-11, 09-01, 09-12
- **Blocks**: none

### Operator card

- **Benefit.** A trained holonomic layer becomes a checkable
  annihilator artifact.
- **How it works.** Read `OrePolynomial` / fitted annihilator from
  09-12. Emit JSON (certificate v1 style) and optional Lean that
  chains only finite rational identities (kernel or Mathlib project).
- **Strength.** `exp`, `sin`, rational D-finite toys.
- **When to use.** After 09-12 exists as code.
- **When not.** As a continuum PDE theorem. Not CCF. Flags are
  *future earned* only.
- **Accuracy floor.** Finite rational Lean only (01-11 / formal
  loop).

## 2. Where it lands

`omnibias.holonomic` + `omnibias.formal` / `omnibias.core.proof`. No
new package.

## 3. Prior art in omnibias

- Spec 09-12 — holonomic layer
- Spec 01-11 — collapse weights as rationals, Lean-checkable
- `omnibias.holonomic._core.dfinite._fit_annihilator`, `DFinite`
- `omnibias.core.proof` certificate v1 + `lean_check`
- `formal/omnibias-verified-kernel` — finite rational `ZInterval`

**Confirmed gap.** No exporter from a trained net / holonomic layer
to an Ore artifact + optional Lean obligation.

## 4. Mathematics

If `L u = 0` with rational `p_k`, a finite check is "these rational
coefficients satisfy this algebraic identity" or "this interval
residual is negative." That is the formal loop's scope. Asymptotics,
limits, and continuum statements are **not** emitted as Lean.

No bias collapse in the export itself (the jet data may have come
from collapse). No temperature collapse.

## 5. Worked example

Export `D-1` for `exp`. Artifact coefficients `[1, -1]` (or the
library's `OrePolynomial` serialization). A finite obligation "the
order is 1 and the leading coefficient is 1" is rational. Implementer:
round-trip serialize/parse is identity; `theorem_prover_verified`
stays `false` until a real `lake build`.

## 6. Proposed API

Does not exist yet. Pure-Python export; no default-dtype issue.

```python
# omnibias/holonomic/_core/export.py — proposed
@dataclass(frozen=True)
class AnnihilatorExport:
    ore: dict
    lean_path: str | None
    theorem_prover_verified: bool = False
    mathlib_verified: bool = False

def export_annihilator(layer, *, emit_lean: bool = False) -> AnnihilatorExport:
    """Must not set verified flags without a genuine lake build."""
```

## 7. Practical use cases

1. **Round-trip** `D-1` (worked example).
2. **Fit-then-export** `sin` from 09-12 G2.
3. **Not** exporting a Navier–Stokes annihilator.

## 8. Acceptance gates

- **G1 round-trip.** `D-1` serialize/parse identity.
- **G2 honesty.** Flags are `false` in the spec-only / default
  export; a later CI job may set them only after `lake build`.
- **G3 scope.** Exporter raises if asked to emit a Lean `sorry` or a
  continuum statement (string guard on forbidden claims).
- **G4 purity.** Export module imports neither torch nor jax.

## 9. Benchmark plan

- `benchmarks/net_to_annihilator.py`
- Smoke: `docs/benchmarks/net_to_annihilator_smoke.json`
- `--full`: `$OMNIBIAS_SCRATCH/inventions/annihilator_export/`

## 10. Honesty and scope

- Finite rational Lean only. `theorem_prover_verified` and
  `mathlib_verified` are never asserted here. Not NS, YM, RH, P vs
  NP, or CCF stretch.
- Certificate tier: export format; Lean tiers future-earned.

## 11. Open questions and risks

- **Fit instability** of annihilators (09-12 G2). Export the residual
  of `L u` as an interval, not a fake exact `L`.
- **Falsifier.** G3 lets a continuum sentence into Lean.

## 12. Implementation checklist

- [x] Export helpers in `omnibias.holonomic` + formal bridge
- [x] Forbidden-claim string tests
- [x] `benchmarks/net_to_annihilator.py` plus smoke JSON
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
