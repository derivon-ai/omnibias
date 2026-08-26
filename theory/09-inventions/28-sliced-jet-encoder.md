# 09-28 Sliced-jet encoder

## 1. Thesis and status

A holistic encoder whose tokens are **jets of learned 1-D scans**
(and/or `integral` mass along `w`), optionally selected by a **named**
energy, and mixed by shipped Hopfield attention — not a patch ViT.

- **Status**: shipped (G1–G5 CI; G5 torch/jax parity on G1; not a ViT)
- **Depends on**: 01-02, 09-01, 09-13
- **Blocks**: none

03-04 applies if the loss is sliced OT. Soft max-pool / morphology
stay in 03-05.

### Operator card

- **Benefit.** A global descriptor that is tomographic (lines along
  `w`) and jet-valued, with an explicit energy if features are sparse.
- **How it works.** Learned directions `w_i`; `BiasScan` or
  `integral` scan per direction; optional 1-jet of each profile;
  optional Hopfield `E` or reconstruction + `soft_top_k`; mix with
  `attention` / `modern_hopfield_retrieve`.
- **Strength.** Small images / templates, 1-D projections.
- **When to use.** After 01-02; when "whole image" means slices, not
  patches.
- **When not.** As a patch-free ImageNet ViT. Not `R^D` equivariance.
  Not 09-02 (`compose_jet` residual stream). Not 05-01 (interface
  inverse problems).
- **Accuracy floor.** Slice truncation (finite `w_i`); jet `R_N`.

## 2. Where it lands

`omnibias.{torch,jax}.architectures` beside Scan-Net; consume
`omnibias.hopfield`. Fails the package test. No new package.

## 3. Prior art in omnibias

- Spec 01-02 — `BiasScan` / `BankSpec`; transverse convolution along
  `w`.
- Spec 01-13 — family is `scan(role)`; this file is a consumer.
- Spec 02-01 — Scan-Net stacks banks; still per-layer, on-lattice.
- `omnibias.hopfield` — `attention`, `modern_hopfield_retrieve`,
  `hopfield_energy`, closed-form log-sum-exp Jacobian / Hessian.
- Spec 09-13 — memories are germs (consume, do not rewrite).
- Spec 09-02 — tokens are jets mixed by `compose_jet`. **Different
  mix.** This encoder uses shipped vector / germ Hopfield on scan
  tokens, not a transformer residual stream.
- Spec 05-01 — "where is the interface"; not a global descriptor.
- Spec 03-04 — sliced OT / CDF (optional loss).
- Spec 03-05 — `soft_max_pool` / `soft_top_k` support.
- CmbNet — conv then `AdaptiveAvgPool2d` (the dull global readout).

**Confirmed gap.** No named encoder whose tokens **are** scan jets
(or integral masses) of an image, with a required named energy when
sparsity is claimed.

## 4. Mathematics

For an image field `f` on a box, a direction `w` and offsets `b_k`
give the scan

```
s_w(k) = role( <w, ·> + b_k ; f )
```

with `role` in `{identity, grad, derivative, band, integral}`
(01-13). A token is the profile `s_w` or its `N`-jet along the scan
axis (bias collapse for derivative heads; **window** for `integral`).

A finite set `{w_i}` is a sliced / Radon view. It is **not** the
translation group of `R^D`.

Optional selection. The energy must be **named**:

- Hopfield `E` (Ramsauer; shipped `hopfield_energy`), or
- reconstruction `‖f - decode(selected)‖` plus `soft_top_k` on scan
  responses.

`beta -> inf` / graph temperature → 0 is **temperature collapse**
and must be labelled. An unnamed "extract the most important
features" is not this spec.

Mix: shipped `attention` / retrieve on the selected tokens. That is
not 09-02's `compose_jet` mix.

## 5. Worked example

2×2 image `[[1, 0], [0, 0]]`. Directions `w = (1,0)` and `(0,1)`,
`identity` scan at pixel centers.

Horizontal profile `(1, 0)`; vertical profile `(1, 0)`. Concatenate
as two tokens of length 2. A linear decode that copies the two
profiles onto rows/columns reconstructs the image exactly
(MAE `0`). Global average-pool is `0.25` and reconstructs the
constant `0.25` (MAE `0.375`).

Implementer: this 2×2 case must report scan-encoder MAE `0` and
GAP MAE `0.375`. Skill vs the mean image is positive.

## 6. Proposed API

Does not exist yet. Bit-identical torch / jax twins; default dtype.

```python
# omnibias/{torch,jax}/architectures/sliced_jet.py  — proposed
@dataclass(frozen=True)
class SlicedJetConfig:
    n_directions: int
    role: str = "identity"   # one of the six roles
    jet_order: int = 0
    energy: str = "none"     # none | hopfield | recon_topk
    top_k: int | None = None

class SlicedJetEncoder:  # nn.Module / flax
    def encode(self, image) -> Tensor:
        """Tokens: (n_directions, scan_len, 1+jet_order)."""
    def decode(self, tokens) -> Tensor: ...
```

JAX: scans over directions with `vmap`; no Python loop over pixels.

## 7. Practical use cases

1. **2×2 worked reconstruct** (section 5).
2. **Template match** of a small blob vs GAP and vs same-width
   CNN+GAP.
3. **Hopfield retrieve** of a scan-jet dictionary (09-13 memories).
4. **Not** ImageNet classification.

## 8. Acceptance gates

- **G1 reconstruct.** Worked 2×2: encoder MAE `= 0`, GAP MAE
  `= 0.375`.
- **G2 skill.** Named small image (e.g. 8×8 two-blob): reconstruction
  MAE strictly below GAP and below a same-width CNN+GAP, five seeds,
  skill vs the mean image positive.
- **G3 energy named.** If `top_k` is set, `energy` is not `"none"`;
  a test refuses an unnamed sparse readout.
- **G4 honesty.** `imagenet_claim` false; `euclidean_RD_claim` false;
  `vit_claim` false; `stretch_claim` false.
- **G5 parity.** torch / jax bit-identical on G1.

## 9. Benchmark plan

- `benchmarks/sliced_jet_encoder.py`
- Smoke: `docs/benchmarks/sliced_jet_encoder_smoke.json`
- `--full`: `$OMNIBIAS_SCRATCH/inventions/sliced_jet_encoder/`

## 10. Honesty and scope

- Not a ViT. Tokens are scans along `w`, not patches.
- Equivariance is 01-02 / 02-01 honesty (along `w`, on-lattice).
- Not 09-02. Not 05-01. Not CCF stretch.
- Certificate tier: empirical. Optional `log(N)/beta` if the energy
  uses `soft_top_k` (reuse the graph / 03-05 bound).

## 11. Open questions and risks

- **Too few `w_i`.** Some 2-D structure is invisible to two axes;
  G2 may fail until directions learn. That is an accuracy floor.
- **Overlap with Scan-Net.** A stacked 02-01 net plus GAP is not
  this spec unless tokens are declared scan jets and the energy is
  named.
- **Falsifier.** G2: CNN+GAP wins, or a PR claims ImageNet / ViT.

## 12. Implementation checklist

- [x] `SlicedJetEncoder` torch / jax twins
- [x] 2×2 G1 test
- [x] Named-energy guard
- [x] `benchmarks/sliced_jet_encoder.py` plus smoke JSON
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
