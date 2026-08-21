# 08-03 Depth-causal local jet

## 1. Thesis and status

A layer may update from the **forward** jet of its incoming representation
before the final loss is known, so training can be causal in depth: each
layer does a local Gauss–Newton step on a named local residual and pushes
a compressed jet to the next layer.

- **Status**: gated
- **Depends on**: 01-01, 01-10, 08-01
- **Blocks**: none

### Operator card

- **Benefit.** Uses closed-form `layer_jet` as a learning signal, not only
  as an analysis tool; later layers see an already-corrected representation.
- **How it works.** State is `(h, J)` with `J` equal to `k` directional
  jets or `d h / d x`. Local residual: a linear readout, a PDE projection,
  invert-and-match (`sigma^{-1}` of a target), or predictive-coding
  `eps = h - sigma(W h_prev)`. Damped GN on that residual; then
  `layer_jet` onward.
- **Strength.** PINNs and other systems that have a residual at every
  site; warm-start before end-to-end GN.
- **When to use.** After a cold start, before `CubicNewton` / spec 03-12;
  or as a depth-march that is then refined end-to-end.
- **When not.** As a claimed replacement for backprop on large supervised
  nets; with a full `d h / d theta` tensor; as a CCF stretch strategy.
- **Accuracy floor.** Greedy bias: early layers fit a proxy. End-to-end
  residual after a local-only run may sit above jointly trained residual.

## 2. Where it lands

A small submodule `omnibias.{torch,jax}.train_local` beside `jet` and
`optim`, **or** functions in `omnibias.{torch,jax}.optim` if the surface
stays under a few hundred lines. Fails the package test: same domain as
existing optimizers.

## 3. Prior art in omnibias

- `omnibias.{torch,jax}.jet.layer_jet` — the forward  N-jet through one
  `sigma(W z + b)`.
- `omnibias.{torch,jax}.optim.GaussNewton` / `gauss_newton_direction` —
  the local step, once a residual is defined.
- `omnibias.pinn.train` — causal in **time**, not in depth. 08-05 is the
  PDE-residual sibling of this spec; this file's local objective may be a
  *proxy* (readout or prediction error).
- Spec 03-12 — line search along a *global* direction; complementary.

**Confirmed gap.** Nothing consumes `layer_jet` to write `dW` before the
next layer, and nothing forbids a full parameter Jacobian in an API.

## 4. Mathematics

At layer `ell`, with incoming jet `J_{ell-1}` (shape `(N+1, k, width)`
or a PINN spatial jet),

```
u = W h + b
h_new = sigma(u)
J_new = compose_jet(affine_jet(J_{ell-1}; W, b), sigma_tower)
```

`sigma_tower` is the founding **bias collapse** (`delta -> 0`) evaluated
from `omnibias.core.polynomials`. The local residual `r_ell(h_new)` must
be **named in the call** (no implicit "match the label if this were the
last layer" unless a readout is passed).

Gauss–Newton: `dW` solves `min || r_ell + (dr/dW) dW ||` with damping.
Then optionally apply `dW`, recompute `(h_new, J_new)`, and continue.

**Variants (same module, same gates unless noted).**

1. **Readout / projection.** `r_ell = A_ell h_new - y` or a linear
   projection of a PDE residual (08-05 owns the full PDE-at-every-layer
   rule).
2. **Invert-and-match.** For strictly monotone `sigma` (sigmoid, tanh),
   `z* = sigma^{-1}(target)` and `r = u - z*`.
3. **Predictive coding.** `r = h_new - stopgrad(sigma(W h_prev))` with
   the usual local error; updates use exact `sigma'`.

No temperature collapse. Cost is `O(L k)` plus one GN solve per layer,
not `O(P width)`.

## 5. Worked example

**One hidden layer, local readout, numbers.**

Input `x = 1.0`, hidden `h = tanh(w x)` with `w = 0.5`,
`h = tanh(0.5) ≈ 0.4621`. Local readout `yhat = v h`, `v = 0.2`,
target `y = 1.0`, `r = 0.0924 - 1.0 = -0.9076`.

`dr/dv = h ≈ 0.4621`, so the GN step in `v` (no damping) is
`dv = 0.9076 / 0.4621 ≈ 1.964`, new `v ≈ 2.164`, new
`yhat ≈ 1.000` on this point. The hidden weight `w` is then updated
from `dr/dw = v * sech^2(0.5) * x` using the **new** `v` (causal:
the readout step happened first). That is the "flow": the next
parameter sees a corrected downstream map.

A two-point check the implementer must reproduce: after the local `v`
step, `|r|` on this sample is `< 1e-12` before `w` moves; after `w`
moves, `|r|` stays below `1e-6` if `v` is immediately re-fit (one
alternating sweep).

## 6. Proposed API

Shipped as `omnibias.{torch,jax}.train_local`. Bit-identical torch / jax
twins; default dtype.

```python
# omnibias/torch/train_local.py  (and jax twin)
@dataclass(frozen=True)
class LocalJetConfig:
    n_directions: int = 4
    jet_order: int = 1
    damping: float = 1e-4
    variant: str = "readout"   # readout | invert | predcode
    allow_full: bool = False
    refit_last: bool = True

class LocalJetForbidden(ValueError):
    """Raised when the caller requests a full d h / d theta."""

def local_jet_step(
    layers, x, *, config: LocalJetConfig, target=None,
    local_residual_fn=None, allow_full=None,
) -> tuple[list, LocalJetReport]:
    """One reverse-depth local GN sweep, then a forward layer_jet.
    Raises LocalJetForbidden if n_directions >= n_params without
    allow_full."""
```

JAX: `local_residual_fn` must be a pure function of activations; no
Python-side mutation inside `jax.jit`. The driver itself is not
`jax.jit`-wrapped.

## 7. Practical use cases

1. **Warm-start a 1-D Poisson PINN.** A few local-jet sweeps, then 50
   end-to-end GN steps. The gate is vs cold GN (same 50 steps).
2. **Invert-and-match** on a tanh last hidden layer when a target
   embedding is known (autoencoder-style). Exact `artanh` plus `sigma'`.
3. **Predictive coding** as a debug mode: local errors should go to
   machine epsilon on a linear network (sanity).
4. **Not** ImageNet. Local proxies lose to backprop when the only true
   residual is at the logit.

## 8. Acceptance gates

- **G1 forbid flood.** `n_directions >= n_params` raises
  `LocalJetForbidden` unless `allow_full=True` (test-only flag, off).
- **G2 warm-start.** On a named 1-D Poisson (or heat) PINN, five seeds:
  local-jet warm-start + 50 `GaussNewton` steps reaches a residual
  strictly below cold `GaussNewton` (50 steps) **or** below a named
  absolute threshold that both must beat vs `u = 0` (skill > 0). If
  warm-start is worse on all seeds, G2 fails.
- **G3 greedy honesty.** A local-only arm (no end-to-end GN) is reported
  and **must not** be written as "optimal." The artifact field
  `greedy_only_claimed_optimal` is `false`.
- **G4 parity.** torch / jax bit-identical on a 2-layer toy.

## 9. Benchmark plan

- `benchmarks/depth_causal_local_jet.py`
- Smoke: `docs/benchmarks/depth_causal_local_jet_smoke.json`
- `--full`: `$OMNIBIAS_SCRATCH/training/local_jet/`
- CI smoke after implementation.

## 10. Honesty and scope

- Bias collapse supplies the jet. No temperature collapse.
- **Not** "we skip the chain rule." `compose_jet` is the chain rule.
- **Not** a global min. Local GN is greedy.
- **Not** CCF stretch. Hilbert remains the campaign floor (spec 07-03,
  spec 08-01).
- Certificate tier: empirical.

## 11. Open questions and risks

- **Proxy mismatch.** A readout residual can paint features that hurt
  the PDE residual. G2 uses a PINN residual as the *score*, even if the
  local proxy differed.
- **Invert-and-match** fails for non-monotone activations (GELU). Variant
  must raise, not silently use a local linearization as an inverse.
- **Falsifier.** G2 failure means this is not a useful warm start; keep
  the module as a research hook or delete the PINN claim from the ledger.

## 12. Implementation checklist

- [x] `omnibias.{torch,jax}.train_local` (or `optim` helpers)
- [x] `LocalJetForbidden` on full Jacobians
- [x] Tests for invert / predcode variants
- [x] `benchmarks/depth_causal_local_jet.py` plus smoke JSON
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
