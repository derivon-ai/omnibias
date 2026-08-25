# 08-05 Depth-causal residual learner

## 1. Thesis and status

Every hidden layer emits a spatial jet of the field `u` and a cheap PDE
residual, so training can **march in network depth** the way
`omnibias.pinn.train` already marches in physical time.

- **Status**: gated
- **Depends on**: 01-01, 01-10, 08-01, 08-03
- **Blocks**: none

### Operator card

- **Benefit.** Features are shaped by the actual residual as soon as a
  layer can express a jet of `u(x)`, not only at the last readout.
- **How it works.** After layer `ell`, interpret a linear map of `h_ell`
  as a field (or as coefficients of a known basis), evaluate
  `mlp_jet` / spatial jets, form `r_ell = PDE(u_ell)`, and take a damped
  GN step on that residual. Then proceed to layer `ell+1` with the
  updated activations.
- **Strength.** 1-D Poisson / causal heat, where a residual is defined
  at every collocation point and jets are cheap.
- **When to use.** PINNs with a **local** differential operator. Pair
  with 08-03's machinery (`k` directions, no full `d h / d theta`).
- **When not.** CCF stretch (Hilbert is nonlocal and floors `~1e-1`);
  as a substitute for last-layer training when intermediate `u_ell` is
  not a valid field (must be declared).
- **Accuracy floor.** Intermediate-field validity: a bad readout of
  `h_ell -> u` poisons features. Hilbert / nonlocal operators are out
  of scope for the gate.

## 2. Where it lands

`omnibias.pinn.train` — a sibling of `march.py` / `_core/causality.py`,
e.g. `omnibias.pinn.train._core.depth_residual` plus torch/jax drivers.
Not a package: same domain and audience as causal marching.

## 3. Prior art in omnibias

- `omnibias.pinn.train._core.causality` — `causality_index`,
  `unlocked_fraction` (Wang–Perdikaris). Time axis.
- `omnibias.pinn.train.torch.march` — time-slab collocation.
- `omnibias.{torch,jax}.jet.mlp_jet` — spatial / directional jets of
  `u`.
- `omnibias.fields` — `grad`, `laplacian` on `FieldState`.
- Spec 08-03 — local GN on a **proxy**. This spec's local objective
  **is** the PDE residual (or a named projection of it).

**Confirmed gap.** Causal marching is in physical time. Nothing marches
the residual in **layer index**.

## 4. Mathematics

Let `u_ell(x) = Decode_ell(h_ell(x))` be a declared decoding (linear
readout to a scalar field, or the identity if `h` is already scalar).
The residual at depth `ell` is the same differential operator `N` used
at the last layer:

```
r_ell(x) = N[u_ell](x)  -  f(x)
```

on the same collocation set. Exact derivatives in `N` come from the
founding **bias collapse** (`sigma^(n)` / `mlp_jet`), not from nested
AD. A GN step updates only `W_ell` (and optionally `Decode_ell`) using
`k` directions.

**Relation to time marching.** Wang–Perdikaris weights unlock later
*time* bins when earlier residuals are small. Here, later *layers* may
be unlocked when `||r_ell||` drops below a named fraction of
`||r_{ell-1}||` (optional; the gate does not require unlocking).

No temperature collapse. No continuum NS claim:
`navier_stokes_proof_claim = false` on artifacts.

## 5. Worked example

**1-D Poisson `-u'' = pi^2 sin(pi x)` on `[0, 1]`, `u(0)=u(1)=0`.**
True `u = sin(pi x)`. A 2-hidden-layer tanh MLP, width 8. After the
first hidden layer, `Decode_1` is a linear map `R^8 -> R`. Then
`u_1''` is a 2-jet of that scalar field. `r_1` is larger than `r_2`
(less expressive). One GN sweep on `W_1` against `r_1`, then on
`W_2` against `r_2`, must produce a final `||r_2||` that beats `u=0`
(`||f||` skill > 0) and is at most the last-layer-only control
(same architecture, same total GN steps, five seeds) — that is G1.

Hand check for the manufactured solution: if `Decode` and the net
implement `sin(pi x)` exactly, `r = 0`. The gate uses a random
initialization, not this oracle.

## 6. Proposed API

Does not exist yet.

```python
# omnibias/pinn/train/_core/depth_residual.py  — proposed
@dataclass(frozen=True)
class DepthResidualConfig:
    n_directions: int = 4
    jet_order: int = 2
    damping: float = 1e-4
    unlock_ratio: float | None = None  # optional Wang-style unlock

def depth_residual_sweep(
    layers, decode, pde_residual, collocation, *,
    config: DepthResidualConfig,
) -> dict:
    """One causal-in-depth sweep. Does not exist yet."""
```

Torch / jax drivers must be bit-identical; default dtype; `pde_residual`
must use `mlp_jet` / field ops, not a hardcoded `float32` FD stencil
unless labelled `NUMERICAL`.

## 7. Practical use cases

1. **1-D Poisson / heat** — the gate.
2. **Causal heat with both axes.** Time marching (existing) plus depth
   marching (this spec) on the same residual; report both.
3. **Not CCF.** Nonlocal Hilbert is the recorded floor (07-03).

## 8. Acceptance gates

- **G1 last-layer comparison.** 1-D Poisson or causal heat, five seeds:
  depth-causal residual `<=` the same architecture trained only at the
  last layer (matched GN step budget). Absolute residual below a named
  threshold that the zero field fails (skill > 0). Reuse PINN four-gap
  style thresholds where they already exist for that PDE; do not invent
  a weaker baseline.
- **G2 honesty fields.** Artifact records
  `navier_stokes_proof_claim: false`, `stretch_1e-13_cleared: false`,
  `hilbert_not_in_scope: true` when the PDE is local.
- **G3 parity.** torch / jax on a tiny Poisson.

## 9. Benchmark plan

- `benchmarks/depth_causal_residual.py`
- Smoke: `docs/benchmarks/depth_causal_residual_smoke.json`
- `--full`: `$OMNIBIAS_SCRATCH/training/depth_residual/`
- CI smoke after implementation.

## 10. Honesty and scope

- Bias collapse for jets. Not temperature collapse.
- **Does not clear CCF stretch.** Hilbert remains `~1e-1` until 07-03
  moves the operator.
- **Not** a global minimum of a deep nest. Depth-causal residuals still
  use the chain rule (`layer_jet` / Faà di Bruno).
- **Not** global regularity. Finite collocation residual only.
- Distinct from 08-03 (proxy residual).

## 11. Open questions and risks

- **Invalid intermediate fields.** If `Decode_ell` cannot represent
  boundary conditions, `r_ell` is the wrong teacher. Hard-BC wrappers
  (`omnibias` hard-condition layers) should be used when the last layer
  uses them.
- **Double counting.** Depth + time causal weights can starve later
  layers. Unlock schedules need a gate of their own if enabled.
- **Falsifier.** G1 failure: last-layer-only is better; the ledger row
  becomes "unearned as a PINN trainer."

## 12. Implementation checklist

- [x] `omnibias.pinn.train._core.depth_residual` plus torch/jax drivers
- [x] Reuse `mlp_jet` / field Laplacian; no FD unless labelled
- [x] Tests: G1 Poisson, G2 honesty keys, G3 parity
- [x] `benchmarks/depth_causal_residual.py` plus smoke JSON
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
