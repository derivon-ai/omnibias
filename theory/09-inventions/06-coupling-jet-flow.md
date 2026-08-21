# 09-06 Coupling Jet-Flow

## 1. Thesis and status

A finite-depth coupling flow whose Jacobian determinant is
`sum log sigma'(z_i)` in closed form and whose inverse is Newton with
exact `sigma'`.

- **Status**: gated
- **Depends on**: 09-01
- **Blocks**: none

### Operator card

- **Benefit.** Exact log-det and an exact-derivative inverse, without
  integrating a CNF ODE.
- **How it works.** Affine couplings plus elementwise `sigma`.
  `log|det J| = sum_i log sigma'(z_i)` from the Riccati / tower
  fastpath. Invert each `sigma` by damped Newton using `sigma'`.
- **Strength.** Small density models and 1-D / 2-D toy flows.
- **When to use.** First-bet after MAML / TM cluster.
- **When not.** As a rewrite of shipped `integrate_cnf`. Not ImageNet
  generative SOTA. Not CCF.
- **Accuracy floor.** Saturation (`sigma' -> 0`) breaks invertibility.

## 2. Where it lands

`omnibias.score.flow` beside `cnf_dynamics` / `integrate_cnf`. Same
package, new submodule. Fails a new-package test.

## 3. Prior art in omnibias

- `omnibias.score.flow.{torch,jax}.ops.cnf.cnf_dynamics`,
  `integrate_cnf` — continuous flow; exact field `div`.
- `omnibias.{torch,jax}.jet` — `sigma` tower for `sigma'`.
- Spec 08-08 — DEQ `u = sigma(W u + x)` with IFT, not a coupling flow.

**Confirmed gap.** No finite coupling stack with closed-form
`sum log sigma'` and Newton inverse.

## 4. Mathematics

One coupling: split `x = (x_a, x_b)`, `y_a = x_a`,
`y_b = sigma(s(x_a) * x_b + t(x_a))` (or a monotone `sigma` of an
affine). Then `log|det J| = sum log|s| + sum log sigma'(...)`.
`sigma'` is the founding **bias collapse** order-1 fastpath.

Inverse: solve `sigma(z) = y` by Newton `z <- z - (sigma(z)-y)/sigma'(z)`
with damping when `|sigma'|` is small.

No temperature collapse.

## 5. Worked example

1-D map `y = tanh(2 x)` at `x = 0.3`. `y ≈ 0.5370496`.
`log|det| = log(2 sech^2(0.6))`. `sech^2(0.6) ≈ 0.711378`,
`2*that ≈ 1.422756`, `log ≈ 0.352656`. Newton inverse from `z=0`
recovers `x` to `1e-12` in a handful of steps. The implementer must
match both numbers.

## 6. Proposed API

Does not exist yet. Bit-identical torch / jax twins; default dtype.

```python
# omnibias/score/flow/torch/jet_flow.py  (and jax twin) — proposed
@dataclass(frozen=True)
class JetFlowConfig:
    n_couplings: int = 2
    hidden: int = 8
    activation: str = "tanh"

def jet_flow_forward(x, params, *, config: JetFlowConfig) -> tuple:
    """Returns (y, log_det). log_det from closed-form sigma'."""

def jet_flow_inverse(y, params, *, config: JetFlowConfig, tol: float = 1e-10):
    """Newton invert with exact sigma'. Raises if |sigma'| < eps."""
```

JAX: inverse is a `lax.while_loop` with a fixed max step count.

## 7. Practical use cases

1. **2-D Gaussian mixture** NLL vs a 2-coupling RealNVP-style baseline
   that uses autodiff log-det (same architecture, noisier det).
2. **Round-trip** `inverse(forward(x)) = x` to `1e-10`.
3. **Not** latent ImageNet.

## 8. Acceptance gates

- **G1 det.** Worked example `log|det|` matches closed form to `1e-12`.
- **G2 invert.** Round-trip max error `< 1e-10` on a 64-point 1-D grid
  in `(-0.8, 0.8)` (away from saturation).
- **G3 skill.** On a named 2-D two-Gaussian mixture, five seeds: mean
  NLL strictly below an isotropic Gaussian baseline (skill > 0) and
  not worse than `integrate_cnf` at equal wall on the smoke budget.
  If worse than the Gaussian on all seeds, G3 fails.
- **G4 parity.** torch / jax bit-identical on G1.

## 9. Benchmark plan

- `benchmarks/coupling_jet_flow.py`
- Smoke: `docs/benchmarks/coupling_jet_flow_smoke.json`
- `--full`: `$OMNIBIAS_SCRATCH/inventions/jet_flow/`

## 10. Honesty and scope

- Bias collapse supplies `sigma'`. No temperature collapse.
- Distinct from CNF (finite couplings, not an ODE).
- Not CCF. Not a generative SOTA claim.
- Certificate tier: empirical.

## 11. Open questions and risks

- **Saturation.** `tanh` at `|z|>3` makes Newton stall; G2 excludes
  that region on purpose.
- **Falsifier.** G3 failure: exact det does not beat a Gaussian.

## 12. Implementation checklist

- [x] `omnibias.score.flow` jet-flow twins
- [x] Round-trip and log-det tests
- [x] `benchmarks/coupling_jet_flow.py` plus smoke JSON
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
