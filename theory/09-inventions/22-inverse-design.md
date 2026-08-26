# 09-22 Inverse-design optimizer

## 1. Thesis and status

Find `x` such that `f_theta(x) = y` by **Newton-on-`x`** with exact
`sigma'`, optionally inside a Kantorovich ball; optionally fit `theta`
so `sigma'` stays bounded away from 0.

- **Status**: shipped (G1–G4 CI; G4 torch/jax parity on G1; not a global inverse)
- **Depends on**: 08-04, 09-01
- **Blocks**: none

### Operator card

- **Benefit.** Invert the net as the algorithm; weights are secondary.
- **How it works.** Residual `r(x) = f_theta(x) - y`. Newton
  `x <- x - (Df)^{-1} r` with `Df` from exact jets. Reject if
  `|sigma'| < eps` or if 08-04 ball is empty.
- **Strength.** Monotone 1-D maps and small inverse problems.
- **When to use.** After 08-04; control invert-to-act optional.
- **When not.** As 08-03 (that inverts *layers for targets*). Not
  CCF. Not a global inverse.
- **Accuracy floor.** Saturation of `sigma`.

## 2. Where it lands

`omnibias.{torch,jax}.optim` + optional `omnibias.control` if used as
invert-to-act. No new package.

## 3. Prior art in omnibias

- Spec 08-03 invert-and-match — invert a *layer* for a hidden target
- Spec 08-04 — unique-zero ball
- `omnibias.control` CBF-QP safety filter (consume inverse, do not
  re-specify)
- `omnibias.{torch,jax}.jet`

**Confirmed gap.** No Newton-on-input driver with exact `sigma'` and
an optional ball.

## 4. Mathematics

`Df = compose_jet` / `layer_jet` in the input direction (**bias
collapse**). For strictly monotone `sigma`, local invertibility is
`sigma' ≠ 0`. No temperature collapse.

## 5. Worked example

`f(x) = tanh(2x)`, `y = 0.5`. `artanh(0.5)/2 ≈ 0.274653`. Newton from
`x=0`: `f(0)=0`, `f'(0)=2`, step `0.25`, then converges to
`0.274653` within `1e-12` in a few steps. `|sigma'|` at the root is
`2 sech^2(0.5493) ≈ 1.5 > 0`.

## 6. Proposed API

Does not exist yet. Bit-identical torch / jax twins; default dtype.

```python
# omnibias/torch/optim_inverse.py  (and jax twin) — proposed
@dataclass(frozen=True)
class InverseDesignConfig:
    tol: float = 1e-10
    min_sigma_prime: float = 1e-4
    require_ball: bool = False

def invert_input(f, y, x0, *, config: InverseDesignConfig) -> dict:
    """Newton-on-x. Raises if |sigma'| < min_sigma_prime."""
```

JAX: `lax.while_loop` with a max step cap.

## 7. Practical use cases

1. **Worked tanh invert.**
2. **1-D PINN** "find `x` with `u(x)=u*`."
3. **Not** inverting CCF Hilbert.

## 8. Acceptance gates

- **G1.** Worked invert reaches `|f(x)-y|<1e-12`.
- **G2 skill.** 20 random `y` in `(-0.8,0.8)`: median `|f-x target|`
  `< 1e-10` and all runs finite. Saturated `|y|=0.999` must *raise*
  or reject (not silently diverge).
- **G3 honesty.** `global_inverse_claimed` false.
- **G4 parity.** torch / jax bit-identical on G1.

## 9. Benchmark plan

- `benchmarks/inverse_design.py`
- Smoke: `docs/benchmarks/inverse_design_smoke.json`
- `--full`: `$OMNIBIAS_SCRATCH/inventions/inverse_design/`

## 10. Honesty and scope

- Local Newton inverse. Not 08-03. Not stretch. Optional 08-04 ball
  is a sound enclosure, not a continuum theorem.
- Certificate tier: empirical + optional sound ball.

## 11. Open questions and risks

- **Many-to-one** `f`. Return one root and `n_roots_unknown=true`.
- **Falsifier.** G1 fails on monotone tanh.

## 12. Implementation checklist

- [x] Invert-on-x twins
- [x] Saturation raise test
- [x] `benchmarks/inverse_design.py` plus smoke JSON
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
