# 09-08 Characteristic-Net

## 1. Thesis and status

Learn a vector field `v(x)` with exact jets and **transport** the
solution along characteristics using the closed-form time `integral`
cell, so 1-D conservation laws are method-of-characteristics layers
rather than collocation hopes.

- **Status**: shipped (G1–G4 CI; G4 torch/jax parity on G1; shock flag; not 02-13)
- **Depends on**: 02-13, 09-01, 09-03
- **Blocks**: none

### Operator card

- **Benefit.** Characteristics as a layer, with exact `d v / d x`.
- **How it works.** A tower field `v_theta(x)`; foot of the
  characteristic is an integral of `v` in time (09-03 cell or field
  `integrate`); `u` is constant (or a named source ODE) along the curve.
- **Strength.** 1-D scalar conservation with a convex flux, before
  shocks.
- **When to use.** After 09-03.
- **When not.** As a rewrite of named Cole–Hopf / Miura maps (02-13).
  After characteristics cross (shocks). Not 3-D NS.
- **Accuracy floor.** Crossing time; the layer must report a shock
  flag rather than invent a unique foot.

## 2. Where it lands

`omnibias.pinn` (solver / travelling adjacent). Fails the package test.

## 3. Prior art in omnibias

- Spec 02-13 — named Cole–Hopf / Miura / Bäcklund / Darboux; exactness
  to jet order `N`.
- `omnibias.fields` — `integrate`, grad / jet field ops.
- Spec 09-03 — time-integral cell.
- Spec 02-09 — tanh-method travelling waves (algebra, not characteristics).

**Confirmed gap.** No layer that *learns* `v` and integrates
characteristics with a closed-form time integral.

## 4. Mathematics

For `u_t + c(u) u_x = 0`, characteristics satisfy `dx/dt = c(u)`,
`du/dt = 0`. With a learned `v_theta ≈ c(u)`,

```
x(t) = x0 + Integral_0^t v_theta(x(s)) ds
```

The integral is the **window** / FTC cell in time, or a field
integrator. `v_theta` jets use **bias collapse**. When two feet map
to one `x`, the spec returns `crossed=True` and does not pick a value
silently.

No temperature collapse.

## 5. Worked example

Linear advection `c=1`, `u0 = exp(-x^2)`, `t=0.2`. Foot
`x0 = x - 0.2`. `u(0.2, 0.0) = exp(-0.04) ≈ 0.960789`. A
Characteristic-Net with frozen `v=1` must match this to `1e-12`
(integral of a constant). A learned `v` on this problem must stay
within `1e-3` of `1` on `[-2,2]` after training (G2).

## 6. Proposed API

Does not exist yet. Bit-identical torch / jax twins; default dtype.

```python
# omnibias/pinn/torch/characteristic.py  (and jax twin) — proposed
@dataclass(frozen=True)
class CharacteristicConfig:
    n_time_panels: int = 8
    shock_tol: float = 1e-6

def characteristic_eval(x, t, v_fn, u0_fn, *, config: CharacteristicConfig):
    """Returns (u, crossed). crossed True if feet are non-unique."""
```

## 7. Practical use cases

1. **Linear advection** (worked example) vs a collocation PINN.
2. **Inviscid Burgers before breaking** (`t < 1` for a named sine IC).
3. **Not** post-shock unique entropy solutions as a claimed closed form.

## 8. Acceptance gates

- **G1 constant v.** Frozen `v=1` matches the Gaussian foot to `1e-12`.
- **G2 skill.** Learned `v` on linear advection: `max|u-u_exact| < 1e-3`
  on a 65-point grid at `t=0.2`, five seeds, skill vs `u=0` positive,
  and `max|u|` strictly below a same-budget collocation PINN **or**
  G2 is recorded unearned if the PINN wins (allowed).
- **G3 shock flag.** On Burgers past breaking, `crossed` is `true` on
  at least one probe; artifact `unique_after_shock_claimed` is `false`.
- **G4 parity.** torch / jax bit-identical on G1.

## 9. Benchmark plan

- `benchmarks/characteristic_net.py`
- Smoke: `docs/benchmarks/characteristic_net_smoke.json`
- `--full`: `$OMNIBIAS_SCRATCH/inventions/characteristic/`

## 10. Honesty and scope

- Time integral is the **window** / FTC knob. `v` jets are **bias
  collapse**. No temperature collapse.
- Not 02-13. Not NS. Not CCF. Not a shock-capturing theorem.
- Certificate tier: empirical.

## 11. Open questions and risks

- **Crossing** is structural. G3 must fire.
- **Falsifier.** G2 unearned on linear advection: characteristics are
  not cheaper than collocation even in the trivial case.

## 12. Implementation checklist

- [x] `omnibias.pinn` characteristic twins
- [x] Shock-flag test
- [x] `benchmarks/characteristic_net.py` plus smoke JSON
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
