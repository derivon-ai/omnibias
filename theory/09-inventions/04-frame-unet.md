# 09-04 Frame-UNet

## 1. Thesis and status

A U-Net whose **encoder raises pack order** (order as frequency, 01-07)
and whose **decoder is integral synthesis**, with skip connections that
are FTC identities between a collapse head and an integral cell.

- **Status**: shipped (G1–G4 CI; band skip is not a collapse head; founding bias collapse)
- **Depends on**: 01-06, 01-07, 09-01, 09-03
- **Blocks**: none

### Operator card

- **Benefit.** A wavelet-like encoder–decoder whose filters are OMBU
  packs, not learned FIR.
- **How it works.** Encoder: `identity -> grad -> laplacian ->
  derivative(n)`. Decoder: `integral` cells. Skips: `d/dz` of a decoder
  slab equals the encoder pack at that scale (FTC).
- **Strength.** 1-D multi-scale regression / denoising with a named
  band plan.
- **When to use.** After 09-03 exists as code.
- **When not.** As a claim that `sigma'` is an admissible wavelet
  (01-06 forbids this). Not Littlewood–Paley completeness.
- **Accuracy floor.** 01-06 honesty: frames are not orthonormal and not
  compactly supported.

## 2. Where it lands

`omnibias.{torch,jax}.architectures`. Fails the package test.

## 3. Prior art in omnibias

- Spec 01-06 — `FrameSpec`; `sigma'` not admissible.
- Spec 01-07 — `BandPlan`, `peak_frequency`; pack order is a band
  selector, not a complete partition of frequency.
- Spec 09-03 — integral cell.
- Spec 02-01 Scan-Net — stacked scans, not an order/integral U-Net.

**Confirmed gap.** No encoder–decoder pairs order with integral and
checks skip FTC.

## 4. Mathematics

Let `E_n` be a pack of order `n` (bias collapse to `sigma^(n)`). Let
`D_n` be an integral window at the same `w` and mean bias. The skip
residual is

```
r_skip = d/dz D_n - (S(z+b_hi_n) - S(z+b_lo_n))^{(0)} * scale
```

more honestly: `d/dz I = sigma(z+b_hi) - sigma(z+b_lo)` (band), which
is the **window** role, not collapse. Encoder order-`n` is collapse.
The spec must store **both** a band skip (FTC of integral) and an
optional collapse skip (derivative head). Mixing them is a bug.

No temperature collapse.

## 5. Worked example

Integral cell as in 09-03 (`delta=0.2` at `x=0`): `dI/dx =
sigmoid(0.1)-sigmoid(-0.1) ≈ 0.049958`. A band skip that stores that
difference must match to `1e-12`. An order-1 encoder head at the mean
is `sigmoid'(0) = 0.25`, which is **not** equal to the band skip. The
implementer must ship a test that these two numbers are different and
both are computed.

## 6. Proposed API

Does not exist yet. Bit-identical torch / jax twins; default dtype.

```python
# omnibias/torch/architectures/frame_unet.py  (and jax twin) — proposed
@dataclass(frozen=True)
class FrameUNetConfig:
    max_order: int = 2
    width: int = 8
    window: float = 0.2

def frame_unet_forward(x, *, config: FrameUNetConfig) -> tuple:
    """Returns (y, skip_residuals). skip_residuals include band-FTC
    and optional collapse heads, labelled separately."""
```

## 7. Practical use cases

1. **1-D denoising** against the 01-06 G4 smoke task (skill > 0, no
   admissibility claim).
2. **Multi-scale PINN residual** on a 1-D Helmholtz profile.
3. **Not** 2-D image U-Nets as a claimed SOTA.

## 8. Acceptance gates

- **G1 skip split.** Worked example: band skip and collapse head differ
  by more than `1e-3` and each matches its closed form to `1e-12`.
- **G2 skill.** On the 01-06 denoising smoke (or a named 1-D sine +
  noise), held-out MSE is below the zero predictor and not worse than
  a same-width Scan-Net smoke (median of five seeds). If worse on all
  seeds, G2 fails.
- **G3 honesty.** `sigma_prime_admissible` is `false`.
- **G4 parity.** torch / jax bit-identical on G1.

## 9. Benchmark plan

- `benchmarks/frame_unet.py`
- Smoke: `docs/benchmarks/frame_unet_smoke.json`
- `--full`: `$OMNIBIAS_SCRATCH/inventions/frame_unet/`

## 10. Honesty and scope

- Encoder collapse is **bias collapse**. Decoder gap is the **window**
  knob. No temperature collapse.
- 01-06: not admissible, not orthonormal, not compactly supported.
- Not CCF. Not ImageNet.
- Certificate tier: empirical.

## 11. Open questions and risks

- **Skip confusion.** The main implementation bug is treating band as
  collapse. G1 exists to catch it.
- **Falsifier.** G2 failure: the U-Net shape adds nothing over Scan-Net.

## 12. Implementation checklist

- [x] `omnibias.{torch,jax}.architectures` Frame-UNet
- [x] G1 skip-split test
- [x] `benchmarks/frame_unet.py` plus smoke JSON
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
