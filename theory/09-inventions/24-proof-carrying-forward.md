# 09-24 Proof-carrying forward

## 1. Thesis and status

A forward pass that returns `(y, certificate)` under architecture
constraints that keep Lipschitz / output boxes **non-vacuous** —
distinct from 08-09, which filters a *parameter step*.

- **Status**: shipped (G1–G4 CI; not 08-09; Lean flags unforged)
- **Depends on**: 08-09, 09-01, 09-05
- **Blocks**: none

### Operator card

- **Benefit.** Inference carries a sound box, not only a point.
- **How it works.** The net is a 09-05 TM (or a float net plus
  `taylor_output_bounds` / `lipschitz_bound`). Forward is
  `(mid(y), cert)`. Reject deploy if `width(cert)` exceeds a cap.
- **Strength.** Tiny monotone controllers and 1-D bounds.
- **When to use.** First-bet with 09-05.
- **When not.** As 08-09 (step accept). Deep residual nets (explosion).
  Not CCF. Lean flags are *future earned* only.
- **Accuracy floor.** Enclosure explosion past shallow depth.

## 2. Where it lands

`omnibias.verify` (+ 09-05 core TMs). No new package.

## 3. Prior art in omnibias

- `omnibias.verify._core.certificates.lipschitz_bound`
- `omnibias.verify._core.taylor.taylor_output_bounds`
- Spec 08-09 — accept/reject `theta'`
- Spec 09-05 — TM neuron

**Confirmed gap.** No public forward that *returns* a certificate
object as a first-class pair.

## 4. Mathematics

Soundness: `y_true in box(cert)` for all `x` in the input box.
Polynomial parts use **bias collapse** coefficients. No temperature
collapse. `theorem_prover_verified` / `mathlib_verified` may be
attached later by a genuine `lake build` of a *finite rational*
obligation (box endpoints, not a continuum claim) and are **not**
asserted by this spec.

## 5. Worked example

`y = sigmoid(x)` on `X=[0,0.2]`. True range `[0.5, sigmoid(0.2)]`.
A PCI forward must return a box containing that range with
`width < 0.1` (09-05 G2). A vacuous `[-1e6,1e6]` fails G2.

## 6. Proposed API

Does not exist yet. Default dtype on adapters; core stays pure Python.

```python
# omnibias/verify/_core/pci.py — proposed
@dataclass(frozen=True)
class PCIResult:
    y_mid: float
    box_lo: float
    box_hi: float
    vacuous: bool
    theorem_prover_verified: bool = False  # must stay False unless lake build

def proof_carrying_forward(net, input_box, spec) -> PCIResult:
    """Returns (y, cert). Forbids setting verified flags without a pass."""
```

## 7. Practical use cases

1. **1-unit sigmoid** box (worked example).
2. **Two-layer** TM nest with a width cap.
3. **Not** certified ImageNet.

## 8. Acceptance gates

- **G1 soundness.** Grid + sample of true values lie in the box
  (core verified contract).
- **G2 non-vacuous.** `width < 0.1` on the worked box;
  `vacuous is false`.
- **G3 flags.** `theorem_prover_verified` and `mathlib_verified` are
  `false` unless a real kernel / Mathlib pass is attached in a later
  implementation PR.
- **G4 distinct.** Artifact `is_08_09_step_filter` is `false`.

## 9. Benchmark plan

- `benchmarks/proof_carrying_forward.py`
- Smoke: `docs/benchmarks/proof_carrying_forward_smoke.json`
- `--full`: `$OMNIBIAS_SCRATCH/inventions/pci/`

## 10. Honesty and scope

- Sound enclosure. Lean flags never forged. Not 08-09. Not NS. Not
  stretch. Not a deep-net certificate.
- Certificate tier: sound enclosure; Lean tiers future-earned only.

## 11. Open questions and risks

- **Explosion at depth 3+** is expected; G2 is shallow on purpose.
- **Falsifier.** G1 box misses a grid point (unsound).

## 12. Implementation checklist

- [x] `omnibias.verify` PCI helper
- [x] Flag-forge tests (must not set verified without lake)
- [x] `benchmarks/proof_carrying_forward.py` plus smoke JSON
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
