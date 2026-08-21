# 09-14 Integral-kernel operator

## 1. Thesis and status

A volumetric / DeepONet-style neural operator whose **kernel is an
OMBU `integral` cell** along learned directions — not a surface BEM
and not a Fourier multiplier.

- **Status**: concept
- **Depends on**: 09-01, 09-03
- **Blocks**: none

### Operator card

- **Benefit.** Operator learning with a closed-form line integral
  kernel.
- **How it works.** Trunk / branch as in existing
  `omnibias.pinn.operator`; the kernel applied to a source `f` is
  `Integral K_theta(x, y) f(y) dy` approximated by OMBU integral
  windows along learned `w`.
- **Strength.** 1-D integral operators (Volterra / Fredholm toy).
- **When to use.** After 09-03; when DeepONet users want an exact
  kernel cell.
- **When not.** As a rewrite of 02-06 BEM-Net (surface densities,
  off-surface exact Laplace/Helmholtz). Not FNO SOTA.
- **Accuracy floor.** 1-D kernels; higher-D cubature is 03-06.

## 2. Where it lands

`omnibias.pinn.operator`. No new package.

## 3. Prior art in omnibias

- Spec 02-06 — BEM-Net, surface potentials.
- `omnibias.pinn.operator` — DeepONet / FNO + conditioning.
- Spec 03-06 — certified cubature.
- Spec 09-03 — integral cell.

**Confirmed gap.** No DeepONet kernel that *is* `S(z+b_hi)-S(z+b_lo)`
along a learned direction.

## 4. Mathematics

For a 1-D source `f` on `[a,b]`,

```
(Kf)(x) = sum_m c_m [S(w_m x - w_m y + b_hi) - S(... + b_lo)]  * f-weights
```

more cleanly: the kernel in `y` is an integral-role OMBU in
`z = w(x)·(x-y)`. This is the **window** knob. Collapse `delta -> 0`
recovers a `sigma` kernel (not the default).

No temperature collapse.

## 5. Worked example

Identity-like kernel: `w=1`, window covering a bump. On `f=1` on
`[0,1]`, `x=0.5`, a single wide sigmoid-integral from `y=0` to `y=1`
is `softplus(w(x-0))-softplus(w(x-1))` with `w=1`:
`softplus(0.5)-softplus(-0.5) ≈ 1.0` wait:
`softplus(0.5)≈0.974077`, `softplus(-0.5)≈0.474077`, difference
`0.5`. The implementer records this number and treats it as a
**mass**, not as `||f||_1=1`. G1 is "matches `OperatorBlock` integral
on the same arguments to `1e-12`," not "equals 1."

## 6. Proposed API

Does not exist yet. Bit-identical torch / jax twins; default dtype.

```python
# omnibias/pinn/torch/operator/integral_kernel.py  (and jax twin) — proposed
@dataclass(frozen=True)
class IntegralKernelConfig:
    n_dirs: int = 4
    window: float = 0.5

def integral_kernel_apply(source, coords, params, *, config: IntegralKernelConfig):
    """Volumetric apply. Docstring must say 'not BEM-Net'."""
```

## 7. Practical use cases

1. **1-D Volterra** `u(x) = Integral_0^x f` vs a DeepONet with an MLP
   kernel.
2. **Antiderivative operator** (FTC) as a sanity operator.
3. **Not** 3-D Helmholtz BEM (02-06).

## 8. Acceptance gates

- **G1 cell.** Kernel evaluations match `OperatorBlock(op="integral")`
  to `1e-12` on the worked arguments.
- **G2 skill.** Antiderivative operator on `f=cos`: `max|u-sin| < 1e-3`
  on `[0, pi/2]` (five seeds, skill vs `u=0` positive) and strictly
  below a same-width MLP-kernel DeepONet smoke **or** G2 unearned if
  the MLP wins (allowed, recorded).
- **G3 honesty.** `claimed_bem_net` is `false`; `claimed_fno_sota`
  is `false`.
- **G4 parity.** torch / jax bit-identical on G1.

## 9. Benchmark plan

- `benchmarks/integral_kernel_operator.py`
- Smoke: `docs/benchmarks/integral_kernel_operator_smoke.json`
- `--full`: `$OMNIBIAS_SCRATCH/inventions/integral_kernel/`

## 10. Honesty and scope

- Window knob, not surface BEM, not CCF, not NS.
- Certificate tier: empirical.

## 11. Open questions and risks

- **G2 may lose** to an MLP kernel on a tiny budget. That is allowed.
- **Falsifier.** G1 fails if the kernel is not actually the integral
  role.

## 12. Implementation checklist

- [ ] `omnibias.pinn.operator` integral-kernel twins
- [ ] G1 vs `OperatorBlock`
- [ ] `benchmarks/integral_kernel_operator.py` plus smoke JSON
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
