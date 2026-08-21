# 08-08 Implicit / DEQ Newton

## 1. Thesis and status

An equilibrium layer `u = sigma(W u + x)` is differentiated by the
implicit-function theorem with exact `sigma'`, so one linear solve
replaces unrolled backprop through a fixed-point iteration.

- **Status**: gated
- **Depends on**: 01-01, 08-01
- **Blocks**: none

### Operator card

- **Benefit.** Constant memory in implicit depth; exact IFT Jacobian
  (closed-form `I - diag(sigma') W`) instead of truncated unrolling.
- **How it works.** Solve `u = sigma(W u + x)` to a named residual
  (Anderson / Newton). Then
  `du/dtheta = (I - diag(sigma'(z)) W)^{-1} (explicit partials)`.
  Train the implicit residual or a downstream loss with that VJP.
- **Strength.** 1-D implicit PINNs and small DEQs where the spectral
  radius of `diag(sigma') W` is `< 1` after a contraction constraint.
- **When to use.** When unrolled depth is the memory bottleneck and a
  fixed point is a natural model (implicit layers, some PINN decoders).
- **When not.** When the fixed point does not exist or is unstable;
  CCF stretch; as a global-min claim.
- **Accuracy floor.** Fixed-point residual and the condition of
  `I - sigma' W`. Solver tol must be below the IFT FD gate.

## 2. Where it lands

New submodule `omnibias.{torch,jax}.implicit` next to `jet`. Fails the
package test: same domain and audience as existing layers. No
`omnibias-deq` distribution.

## 3. Prior art in omnibias

- `omnibias.{torch,jax}.jet` — explicit compositions, not fixed points.
- `omnibias.core.polynomials` — `sigma'`.
- `omnibias.core.verified.kantorovich` — can later seal the fixed point
  (optional; 08-04 is the policy spec).
- Spec 01-09 / 02-12 — equality locus / implicit ansatz for **solutions
  of equations**, not a DEQ trainer. Do not merge.

**Confirmed gap.** No `u = sigma(W u + x)` solver with IFT gradients
and torch/jax twins.

## 4. Mathematics

Let `z = W u + x` and `F(u) = u - sigma(z)`. A fixed point satisfies
`F(u*) = 0`. If `I - diag(sigma'(z*)) W` is invertible,

```
d u* / d W = (I - diag(sigma'(z*)) W)^{-1}
             (diag(sigma'(z*)) (d(W u* + x)/d W)_explicit)
```

`sigma'` is the founding **bias collapse** tower at order 1. Error:
solver residual `||F(u)||` plus linear-solve residual. No temperature
collapse.

**Contraction (recommended, not the only mode).** If
`||diag(sigma') W|| < 1` on the iterate, Banach iteration converges;
the API must report `spectral_radius_bound` (a cheap infinity-norm
bound is enough for the gate).

JAX tracing: the solver loop is `lax.while_loop` or a fixed `lax.scan`
of Newton steps; Python `for` with data-dependent break is not
jit-safe.

## 5. Worked example

**Scalar.** `W = 0.2`, `x = 0.5`, `sigma = tanh`.
Iterate `u <- tanh(0.2 u + 0.5)` from `u = 0`:

```
u0 = 0
u1 = tanh(0.5) ≈ 0.4621
u2 = tanh(0.2*0.4621 + 0.5) ≈ tanh(0.5924) ≈ 0.5316
u3 ≈ tanh(0.6063) ≈ 0.5413
u4 ≈ tanh(0.6083) ≈ 0.5426
```

Fixed point near `0.5430` (implementer refines).
`sigma' = 1 - u*^2 ≈ 0.705`.
`I - sigma' W ≈ 1 - 0.141 = 0.859`.
IFT `du/dW` is a scalar the FD check (G1) must match to `1e-8`.

## 6. Proposed API

Does not exist yet.

```python
# omnibias/torch/implicit.py  (and jax twin) — proposed
@dataclass(frozen=True)
class DEQConfig:
    solver: str = "newton"     # newton | anderson | iterate
    max_iter: int = 50
    tol: float = 1e-10
    require_contraction: bool = False

@dataclass(frozen=True)
class DEQResult:
    u: Any
    residual: float
    n_iter: int
    spectral_radius_bound: float

def deq_solve(W, x, spec, *, config: DEQConfig) -> DEQResult:
    """Solve u = sigma(W u + x). Does not exist yet."""

def deq_vjp(W, x, spec, g, *, config: DEQConfig) -> Any:
    """VJP of u* wrt W through IFT. Does not exist yet."""
```

Default dtype; jax marks `deq_solve` as the scan/while primitive.

## 7. Practical use cases

1. **Tiny DEQ gradient check** — G1.
2. **1-D implicit PINN** — G2 named residual.
3. **Memory-constant implicit depth** vs unrolled ResNet of equal
   effective depth (report memory; no required win for the first gate).

## 8. Acceptance gates

- **G1 IFT vs FD.** Tiny DEQ (width `<= 4`): `deq_vjp` matches
  central FD to `<= 1e-8` relative, and `||F(u*)|| <= 1e-10`.
- **G2 implicit PINN.** 1-D Poisson (or a scalar implicit residual)
  trained with IFT-GN reaches a named absolute residual that `u=0`
  fails (skill > 0), five seeds.
- **G3 parity.** torch / jax bit-identical on G1.
- **G4 tracing note.** Docs / docstring state which loop is
  `lax.scan` / `while_loop` vs Python.

## 9. Benchmark plan

- `benchmarks/implicit_deq.py`
- Smoke: `docs/benchmarks/implicit_deq_smoke.json`
- `--full`: `$OMNIBIAS_SCRATCH/training/implicit_deq/`
- CI smoke after implementation.

## 10. Honesty and scope

- IFT is the chain rule at a fixed point, not an absence of the chain
  rule.
- Not a global min; not CCF stretch; not continuum existence of a PDE
  (08-04 may later seal the finite map).
- Bias collapse for `sigma'` only.

## 11. Open questions and risks

- **Divergence.** If `require_contraction=True` and the bound is
  `>= 1`, the solver must raise, not silently unroll.
- **Anderson vs Newton.** Gate uses Newton; Anderson is extra.
- **Falsifier.** G1 FD mismatch: do not ship. G2 failure: module can
  remain as a solver without a PINN claim.

## 12. Implementation checklist

- [x] `omnibias.{torch,jax}.implicit`
- [x] IFT VJP with exact `sigma'`
- [x] Tests: G1 FD, G3 parity, contraction raise
- [x] `benchmarks/implicit_deq.py` plus smoke JSON
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
