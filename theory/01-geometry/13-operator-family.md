# 01-13 Operator family (scan of six roles)

## 1. Thesis and status

A new omnibias operator is **`scan(role)`** for one of the six
`OperatorBlock` roles — not a seventh role, not `Conv2d` on `R^D`, and
not a new invention group. This file is the generator rule and the
catalog; named architectures stay in Groups 02 / 03 / 09.

- **Status**: designed
- **Depends on**: 01-01, 01-02, 06-02, 09-01
- **Blocks**: none

### Operator card

- **Benefit.** One page answers "is this a new primitive or a scan of
  a role we already have."
- **How it works.** Role × scan table, inventable list as **pointers**,
  reject list, design test. First later spend: `BiasScan(op="integral")`,
  then spec 09-14.
- **Strength.** Stops a seventh `OperatorBlock` role and stops Group 10
  from re-specifying 09-14 / 03-05.
- **When to use.** Before proposing a convolution-like operator or a
  new `op=` string.
- **When not.** This file does not implement a scan and does not train
  a network.
- **Accuracy floor.** None of its own. Scan-Net equivariance stays
  per-layer, on-lattice. CCF Hilbert is not an operator-family gate.

## 2. Where it lands

This document (docs-only). Implementations stay in existing homes:
`omnibias.{torch,jax}.scan`, `omnibias.fields`, and the pointed 02 / 03
/ 09 specs. No new package.

## 3. Prior art in omnibias

- `docs/operator-surface.md` — six roles; `BiasScan` reuses them and
  is **not** a seventh role.
- Spec 01-02 — `BiasScan` / `BankSpec`; transverse convolution along
  `w`. Cites `cmbConv1d` / `cmbConv2d`: pixel-grid conv plus an
  operator-typed nonlinearity — a **different** object from the scan.
- `omnibias.core.scan`, `omnibias.{torch,jax}.scan`.
- Spec 02-01 — Scan-Net stacks scan banks.
- Spec 09-01 — named inventions (FTC-Net, integral-kernel, …) are
  consumers of this family, not a second catalog.

**Confirmed gap.** There is no index that says every convolution-class
operator is `scan(role)` and that `integral` / `band` scans are the
unused spend.

## 4. Mathematics

### Generator

With `z = w . x`, each bias places a parallel hyperplane. The six
roles are a choice of gap (founding **bias collapse** `delta -> 0` for
`grad` / `laplacian` / `derivative`; **window** held finite for `band`
/ `integral`). The position knob (01-02) **slides** one template:

```
scan(role)(x)[k] = role(z + b_k; template)
```

`BiasScan` is that slide. It is not a seventh `op`. Equivariance is an
interior lattice shift along `w`, not the translation group of `R^D`.

`cmbConv*` convolves a **pixel grid** and then applies an
`OperatorBlock`. That is grid convolution with an operator
nonlinearity. It is not `scan(role)`.

No temperature collapse in the generator. A softmax readout on a scan
bank (`gamma -> inf`) would be temperature collapse and must be
labelled if used (01-02).

### Role × scan

| Template role | Scan is | Status |
|---|---|---|
| `identity` | smoothed step slid along `w` | gated (01-02 / Scan-Net) |
| `grad` / `laplacian` | matched derivative filter | gated; collapse heads |
| `derivative(n)` | order-`n` matched filter | gated; 01-07 band selector |
| `band` | sliding slab (band-pass along `w`) | **spend**; role exists, named scan layer thin |
| `integral` | sliding mass `S(z+b_hi)-S(z+b_lo)` | **spend**; first later layer |

### Other first-class operators (not scans)

These are real and must not be rewritten as `scan(role)`:

| Operator | Spec / home | Kind |
|---|---|---|
| Field `grad` / `div` / `curl` / `lap` / `hess` / `jac` | `omnibias.fields` | closed-form on jet fields |
| Line Hilbert / conjugate | 01-12 | nonlocal; not a conv |
| Fractional (analytic class) | `omnibias.fractional` analytic | Gamma-ratio jet |
| Fractional (sampled) | `omnibias.fractional` GL/spectral | numerical |
| q-derivative / Hilger `delta` | `omnibias.qcalculus` / `timescale` | named `q -> 1` / `mu -> 0` |
| Holonomy band | 02-14 | abelian + transverse-constant |
| DeepONet query jets | `omnibias.pinn.operator` | neural operator, sense 3 |
| Wirtinger | `omnibias.fields` | shipped |

### Inventable (pointers only)

If a row already has a spec, implement there. Do not open a second
03 / 09 file for the same object.

| Operator | Pointer |
|---|---|
| Integral-kernel / Volterra | [09-14](../09-inventions/14-integral-kernel-operator.md) |
| FTC pair `(I, dI/dz)` | [09-03](../09-inventions/03-ftc-net.md), [09-17](../09-inventions/17-dual-ftc-training.md) |
| Order-as-frequency bank | [01-07](07-order-as-frequency-spectral-design.md), [09-04](../09-inventions/04-frame-unet.md) |
| Morphology | [03-05](../03-algorithms/05-differentiable-morphology-levelsets.md) |
| Soft max-pool / index-SE | [03-05](../03-algorithms/05-differentiable-morphology-levelsets.md) (flat dilation; `soft_top_k` support) |
| Semiring neighborhood combiner | shipped `cmbConv*` + [01-08](08-tropical-log-homotopy.md) + 03-05 |
| Sliced OT / CDF | [03-04](../03-algorithms/04-sliced-optimal-transport-cdf.md) |
| Jet-Hopfield | [09-13](../09-inventions/13-jet-hopfield.md) |
| Characteristic / transport | [09-08](../09-inventions/08-characteristic-net.md) |
| Riccati flow | [09-10](../09-inventions/10-riccati-flow-net.md) |
| Collapse stencil | [09-11](../09-inventions/11-collapse-net.md), [01-04](04-irregular-birkhoff-stencils.md) |
| BEM / single-layer potential | [02-06](../02-architectures/06-potential-theory-and-bem-net.md) |
| Mollifier / weak test | [01-05](05-mollifier-distribution-calculus.md), [02-04](../02-architectures/04-weak-form-vpinn-closed-test-functions.md) |
| Wirtinger | shipped `omnibias.fields` (also in the non-scan table) |
| Mixed space–parameter jets | [09-27](../09-inventions/27-parameter-space-jets.md) |
| Sliced-jet / energy encoder | [09-28](../09-inventions/28-sliced-jet-encoder.md) |

**First spend (later, not this pass):** a named `BiasScan(op="integral")`
layer, then 09-14.

### Rejected (no spec)

1. A seventh `OperatorBlock` role that is `conv2d`.
2. Translation equivariance on `R^D` (Scan-Net honesty forbids it).
3. "Hilbert convolution" as a CCF stretch fix (07-03 / 08-01 floor).
4. Softmax / generic MoE as an OMBU role (slab mass is 09-07 / 04-02).
5. A seventh role `maxpool` or `vit` (pool is 03-05; encoder is 09-28).
6. Patch-free ImageNet ViT (09-01 reject 4; 09-28 is not that claim).

### Design test

If the proposal is not

```
scan(identity | sigma^(n) | band | integral)  along w
```

it is a **consumer** of the tower, not a new primitive.

## 5. Worked example

**Triaging "we need a convolution operator."**

- Pixel-grid CNN with `OperatorBlock` nonlinearity: already
  `cmbConv*`. Not 01-13. Not a new role.
- Slide `sigma'` along `w`: `BiasScan(op="grad")`. Already 01-02.
- Slide the antiderivative window: `BiasScan(op="integral")`. **First
  spend.** Not a seventh role.
- 2-D Euclidean-equivariant conv: **reject 2.**

**Triaging "integral-kernel neural operator."** Already 09-14. Do not
open 01-14.

## 6. Proposed API

Does not exist as a new module. The catalog names shipped symbols.

```python
# documentation only
ROLES = ("identity", "grad", "laplacian", "derivative", "band", "integral")
GENERATOR = "BiasScan(op=role)"   # not a seventh role
FIRST_SPEND = "BiasScan(op='integral')"
REJECTED = (
    "seventh_role_conv2d",
    "euclidean_RD_equivariance",
    "hilbert_as_ccf_stretch",
    "softmax_as_ombu_role",
    "seventh_role_maxpool_or_vit",
    "patch_free_imagenet_vit",
)
```

A later `BiasScan` integral convenience wrapper lands in
`omnibias.{torch,jax}.scan`, default dtype, bit-identical twins.

## 7. Practical use cases

1. **Refusing** a seventh `op=` in a PR.
2. **Choosing the next layer:** integral scan, then 09-14.
3. **Routing** morphology / OT / BEM to their existing specs.
4. **Citation path (06-05):** the public object stays the six roles +
   `compose_jet`; this catalog does not enlarge the extract.

## 8. Acceptance gates

Document gates only.

- **G1 completeness.** Section 4 has the generator, role × scan table,
  non-scan table, inventable pointer table, and rejects.
- **G2 no duplicates.** Inventable rows point at existing specs; this
  file does not re-specify 09-14, 03-05, or 02-06.
- **G3 rejects named.** The rejected proposals have no
  implementation files.
- **G4 first spend named.** `BiasScan(op="integral")` then 09-14.
- **G5 no new package.**

## 9. Benchmark plan

None of this file's own. The first-spend implementation PR should
extend existing scan benches, not invent a new CI product:

- reuse `benchmarks` scan / Scan-Net smoke
- `--full` under `$OMNIBIAS_SCRATCH/operators/integral_scan/`

## 10. Honesty and scope

- Bias collapse vs window vs temperature collapse named in section 4.
- Not CCF stretch. Not Euclidean CNN SOTA. Not a new `OperatorBlock`
  role.
- Hilbert (01-12) stays line-only and is not a conv.
- Certificate tier: document. Numerical gates live on 01-02 / 02-01 /
  pointed specs.

## 11. Open questions and risks

- **Thin `band` / `integral` scan.** The roles exist; a missing
  convenience API is not a missing primitive. G4 records the spend
  order so a wrapper is not sold as a seventh role.
- **Overlap with 09-01.** Named nets stay in Group 09. This file is
  the *family*. Mixed `μ` jets are 09-27; sliced-jet encoder is 09-28;
  an "integral scan net" is 09-03 / 09-14, not a new role.
- **Falsifier.** A merged seventh `op="conv"` fails G3.

## 12. Implementation checklist

- [x] `theory/01-geometry/13-operator-family.md` (this file)
- [ ] Later: `BiasScan(op="integral")` convenience + tests
- [ ] Later: 09-14 implementation (not this pass)
- [x] Index row in `theory/README.md` (wired in the index pass)
- [x] Pointer in `docs/operator-surface.md` (index pass)

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
