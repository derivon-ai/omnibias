# 08-02 Composed-curvature joint Newton

## 1. Thesis and status

The order-2 chain rule `(f circ g)'' = f''(g) (g')^2 + f'(g) g''` is the
exact coupling between consecutive layers, so a Newton step on the **joint**
block `(W_{ell-1}, W_ell)` can leave a critical point that is a minimum of
the current-layer slice and a saddle of the pair.

- **Status**: shipped (G1–G4 CI; slice escape, not a global min)
- **Depends on**: 01-01, 01-10, 08-01
- **Blocks**: none

### Operator card

- **Benefit.** Unstick a layer-wise "minimum" that is an artifact of freezing
  the previous weights, using a closed-form composed Hessian rather than a
  finite-difference or dropped-`sigma''` Gauss–Newton term.
- **How it works.** After layer `ell-1`, keep a 2-jet of features in `k`
  directions. Form layer `ell` and a residual `r`. Build the `2 x 2` block
  Hessian of `L` on those directions. If the `W_ell` block is PD and the
  joint matrix has `lambda_min < 0`, step along that eigenvector; otherwise
  take a damped joint Newton / GN step.
- **Strength.** Early PINN / residual training, where `grad_h L` is not
  small, so the residual Hessian term `f'(g) g''` is live.
- **When to use.** A 2-hidden-layer (or last-two-layer) net whose
  layer-wise Newton has stalled; CCF / Poisson / heat residuals with a
  **local** operator.
- **When not.** ImageNet-scale full Hessians; CCF stretch (Hilbert floor);
  claiming a global min. Do not run layer-wise Newton and call it this spec.
- **Accuracy floor.** Model class and the `k`-direction subspace. Not
  float64 and not `1e-13` stretch.

## 2. Where it lands

Helpers next to `CubicNewton` in
`packages/omnibias-torch/src/omnibias/torch/optim.py` and the jax twin
`packages/omnibias-jax/src/omnibias/jax/optim.py`, or a small
`omnibias.{torch,jax}.optim_composed` pair if `optim.py` would grow past
maintainability. Same domain, same tier, same audience as the existing
curvature optimizers — **not** a package.

## 3. Prior art in omnibias

- `omnibias.{torch,jax}.jet` — `layer_jet`, `compose_jet` already evaluate
  Faà di Bruno through `sigma(W z + b)` at arbitrary order. Order 2 is
  enough for this spec.
- `omnibias.{torch,jax}.optim` — `hvp`, `CubicNewton`, `GaussNewton`,
  `JetSubspaceTensor`, `taylor_subspace_model`. Subspace second-order
  models exist; they are **not** depth-ordered two-layer blocks.
- `omnibias.core.polynomials` — `sigma'` and `sigma''` for the Riccati
  family, one `sigma` evaluation.

**Confirmed gap.** No optimizer builds the joint Hessian of consecutive
layers from the composed 2-jet, and none tests
`lambda_min(slice) >= 0` against `lambda_min(joint) < 0` as an escape
predicate.

## 4. Mathematics

Let `g` be the map through layer `ell-1` and `f` the current layer plus
the scalar loss (or half-mean-square residual). The founding **bias
collapse** supplies exact `sigma'` and `sigma''`. Then

```
(f circ g)'' = f''(g) (g')^2 + f'(g) g''.
```

In weights, with `h = g(x; W_{ell-1})` and `z = W_ell h + b`,

```
Hess_W L = J_W^T (Hess_h L) J_W  +  (grad_h L) · Hess_W h.
```

The first term is Gauss–Newton (PSD if `L` is convex in `h`). The second
is the residual term: it contains `sigma''` and the incoming `g''`. Near
a good fit `grad_h L -> 0` and GN is enough. Early in residual training
the second term is the coupling that can make the **joint** block
indefinite while the `W_ell` slice is PD.

A critical point of the slice `L(W_ell | W_{neq ell} frozen)` is not a
critical point of `L(W_{ell-1}, W_ell)`. Escape means a step with
nonzero component in `W_{ell-1}` along a negative eigenvector of the
joint subspace Hessian.

**Truncation.** The subspace is `k` directions (`k << P`), not the full
`P x P` tensor. Directions may be the last GN step, a random Krylov
vector, and the incoming feature gradient. Error is the usual Rayleigh
restriction: a negative mode orthogonal to the subspace is invisible.

No temperature collapse appears.

## 5. Worked example

**A two-layer scalar nest where the slice minimum is a joint saddle.**

Let `g(w) = tanh(w)` at input `x = 1`, and `f(h, v) = (v h - 1)^2` with
`v` the second-layer weight. At `(w, v) = (0, 0)`:

```
h = tanh(0) = 0
g' = sech^2(0) = 1
g'' = -2 tanh(0) sech^2(0) = 0
L = 1
dL/dv = 2(v h - 1) h = 0
dL/dw = 2(v h - 1) v sech^2(w) = 0
```

Both partials vanish. The `v`-slice Hessian (freeze `w = 0`) is
`d^2 L / dv^2 = 2 h^2 = 0` (degenerate). Shift to
`(w, v) = (0.2, 0.1)` by hand for a numeric check (four decimals):

```
h = tanh(0.2) ≈ 0.1974
L ≈ (0.01974 - 1)^2 ≈ 0.9609
```

A one-variable Newton step in `v` only moves toward `v* = 1/h ≈ 5.07`,
a slice minimizer (`L -> 0` if `w` is frozen — here it does go to 0).
The interesting case is a residual that **cannot** be zeroed by `v`
alone (e.g. replace the target by a PDE residual that still depends on
`w` after the optimal `v`). Then `Hess_v L > 0` at the slice critical
point while the mixed `d^2 L / dw dv` term makes
`det(Hess_joint) < 0`. The implementer's unit test must construct that
residual explicitly (1-D Poisson with a two-layer MLP is the intended
gate, not this scalar sketch).

Expected diagnostic at a successful escape:

```
lambda_min_slice >= 0
lambda_min_joint < 0
residual_after < residual_before
```

## 6. Proposed API

Does not exist yet. Torch and jax twins, bit-identical, default dtype,
coefficients from `omnibias.core.polynomials`.

```python
# omnibias/torch/optim.py  (and jax twin) — proposed
@dataclass(frozen=True)
class ComposedCurvatureConfig:
    n_directions: int = 4          # k << P; API must reject "all params"
    include_residual_hess: bool = True  # keep f'(g) g''; False => GN
    escape_tol: float = 0.0        # lambda_min(joint) < -escape_tol
    damping: float = 1e-6

@dataclass(frozen=True)
class ComposedCurvatureReport:
    lambda_min_slice: float
    lambda_min_joint: float
    escaped: bool
    step_norm: float

def composed_block_hessian(
    residual_fn, params_prev, params_curr, directions, *,
    config: ComposedCurvatureConfig,
) -> tuple[Any, Any, Any]:
    """Return (H_slice, H_joint, cross) in the k-direction subspace."""

def composed_curvature_step(
    residual_fn, params_prev, params_curr, *,
    config: ComposedCurvatureConfig,
) -> tuple[Any, Any, ComposedCurvatureReport]:
    """Joint damped Newton or escape step. Does not exist yet."""
```

JAX: residual and jets must be traceable (`jax.jit` of the step is not
required for G1; G3 parity is on eager float64).

## 7. Practical use cases

1. **Two-hidden-layer PINN** whose last-layer least squares has stalled
   while the first layer sits in a slice min. Joint step vs layer-wise
   Newton is the gate.
2. **OMBU / arrangement polish** on the last two packs, where `sigma''`
   is closed form and the residual Hessian is cheap.
3. **Warm-start into CubicNewton.** After one escape, hand the joint
   direction to spec 03-12 for the length.
4. **Not** a replacement for Adam on a 50-layer classifier: `k` will
   miss the relevant mode and the win is undefined.

## 8. Acceptance gates

Baselines: Adam, and Newton / GN on `W_ell` only (same `k`, same budget).

- **G1 identity.** On a scalar nest with known `tanh` derivatives, the
  composed Hessian matches high-precision finite differences to
  `<= 1e-10` relative for the three blocks (slice, cross, joint).
- **G2 escape.** On a 2-hidden-layer 1-D Poisson PINN (or an equivalent
  constructed residual), over five seeds: there exists a point where
  layer-wise Newton reports `lambda_min_slice >= 0` and has stalled
  (`|grad_slice|` below a named tol), and the joint step reports
  `lambda_min_joint < 0` and strictly decreases residual. If no such
  point appears in the suite, G2 **fails** and the escape story is
  unearned (ordinary Newton may still ship).
- **G3 parity.** torch and jax bit-identical on G1's scalar nest.
- **G4 no flood.** Passing `n_directions >= n_params` without an explicit
  override raises `ValueError`.

Skill > 0 versus the zero predictor on the Poisson residual is required
for G2 (the field must beat `u = 0`).

## 9. Benchmark plan

- `benchmarks/composed_curvature.py`: G1 scalar nest, G2 2-layer Poisson,
  G3 parity.
- Smoke JSON: `docs/benchmarks/composed_curvature_smoke.json` (tiny net,
  one seed).
- `--full` under `$OMNIBIAS_SCRATCH/training/composed_curvature/`.
- CI: smoke only, next to other theory-program falsifiers, after
  implementation.

## 10. Honesty and scope

- Jets and `sigma''` come from the founding **bias collapse**
  (`delta -> 0`). No temperature collapse.
- **Not** a global minimum of a deep nest. Escape is from a *slice*
  critical point to a lower residual, locally.
- **Not** "we do not use the chain rule." This spec *is* the order-2
  chain rule.
- **Not** CCF stretch and not Navier–Stokes regularity. Hilbert ×
  dictionary remains `~1e-1`. A local Poisson residual is the gate.
- Certificate tier: empirical (G2). No Lean flag.

## 11. Open questions and risks

- **The residual term may be negligible.** If G2 never sees
  `lambda_min_joint < 0` at a PD slice, drop `include_residual_hess`
  and this spec collapses to subspace GN. Record that.
- **Subspace miss.** The negative mode may lie outside the `k`
  directions. Raising `k` is allowed; materializing `P x P` is not.
- **Two layers only.** Extending to a sliding window of three layers is
  a later spec; the gate is two.
- **Falsifier.** G2 failure deletes the "escape local minima of the
  current layer" claim from the ledger row; the identity G1 may still
  pass.

## 12. Implementation checklist

- [x] `composed_block_hessian` / `composed_curvature_step` in
      `omnibias.{torch,jax}.optim` (or `optim_composed`)
- [x] Reuse `layer_jet` order 2; no new jet arithmetic
- [x] Raise on full-parameter Jacobian
- [x] Tests: G1 FD match, G2 escape or honest fail, G3 parity
- [x] `benchmarks/composed_curvature.py` plus smoke JSON
- [x] Docs page only after a gate passes
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
