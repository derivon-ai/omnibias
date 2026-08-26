# 09-02 Jet-token transformer

## 1. Thesis and status

A residual stream whose tokens are **N-jets** (value plus directional
derivatives), mixed by `compose_jet` rather than a softmax of dots, so
the network predicts the contact of a function, not a point.

- **Status**: shipped (G1–G4 CI; G4 torch/jax parity on the worked mix; model jet, not ImageNet; founding bias collapse)
- **Depends on**: 01-10, 09-01
- **Blocks**: 09-19

### Operator card

- **Benefit.** Hidden state is a germ; attention preserves jet algebra.
- **How it works.** Each token is a jet of order `N` in `k` directions.
  A mix is Faà di Bruno composition (`compose_jet` / `compose_jet_mv`),
  not `softmax(QK^T)V` on vectors.
- **Strength.** Fields, CDFs, and 1-D profiles where the 1-jet is the
  object of interest.
- **When to use.** After 09-01 first-bet cluster; pair with 09-19.
- **When not.** ImageNet classification; CCF stretch; as a rewrite of
  Jet-KAN (edge-wise univariate bases) or hopfield vector attention.
- **Accuracy floor.** Jet truncation `R_N`. Mixing is exact for the
  *model* jet, not the target.

## 2. Where it lands

`omnibias.{torch,jax}.architectures` beside gated Scan-Net / Jet-KAN.
Fails the package test: same domain and audience as existing
architectures.

## 3. Prior art in omnibias

- `omnibias.{torch,jax}.jet.compose_jet`, `layer_jet`, `mlp_jet`.
- `omnibias.{torch,jax}.jet_mv.compose_jet_mv`, `layer_jet_mv`.
- Spec 02-03 Jet-KAN — univariate edge bases; exactness of the model
  jet, not a jet-valued residual stream.
- `omnibias.hopfield` — `attention`, `modern_hopfield_retrieve`,
  `logsumexp_hessian` on **vectors**.

**Confirmed gap.** Nothing stores an `N`-jet as the residual stream or
mixes tokens with `compose_jet`.

## 4. Mathematics

A token at site `i` is `J_i = (u_i, D_v1 u_i, ..., D_v1^{N} u_i)` or
the multivariate analogue. A mixing map `Phi` on values lifts by Faà
di Bruno:

```
J_out = compose_jet(J_in, sigma_tower)
```

`sigma_tower` comes from the founding **bias collapse** (`delta -> 0`)
via `omnibias.core.polynomials`. Softmax attention on values is a
*different* mix: it does not transport derivatives. This spec forbids
calling value-softmax a jet mix.

No temperature collapse unless a hopfield `beta` head is an optional
side path, in which case `beta -> inf` is named and is not the jet mix.

Error: omitted order-`N+1` remainder `R_N` of each composition.

## 5. Worked example

Two tokens, order-1 jets, one direction. Token A: `(u, u') = (1.0, 0.5)`.
Token B: `(0.0, 1.0)`. Affine mix `z = 0.5 A + 0.5 B` then `tanh`.

Value: `z = 0.5`, `tanh(0.5) ≈ 0.462117`. Derivative: `sech^2(0.5) * 0.75
≈ 0.786448 * 0.75 ≈ 0.589836`. The implementer must match
`compose_jet` / `layer_jet` on this pair to `1e-12`.

A vector-attention baseline that mixes only values and finite-differences
the derivative must *not* be reported as the jet mix.

## 6. Proposed API

Does not exist yet. Bit-identical torch / jax twins; default dtype.

```python
# omnibias/torch/architectures/jet_token.py  (and jax twin) — proposed
@dataclass(frozen=True)
class JetTokenConfig:
    jet_order: int = 1
    n_directions: int = 2
    n_layers: int = 2
    width: int = 8

def jet_token_forward(tokens, *, config: JetTokenConfig) -> tuple:
    """tokens: (batch, seq, jet_order+1, n_directions, width).
    Mix by compose_jet. Raise if n_directions * width >= n_params
    without allow_full (08-01 reject 2)."""
```

JAX: pure function; `jax.jit` over a fixed `jet_order`.

## 7. Practical use cases

1. **1-D field regression.** Predict `u` and `u_x` together; vs a vector
   MLP that is differentiated after the fact.
2. **Teacher jet matching** (09-19) on a fitted PINN field.
3. **Not** token classification on language.

## 8. Acceptance gates

- **G1 algebra.** The worked example matches `compose_jet` to `1e-12`.
- **G2 skill.** On a named 1-D harmonic (`u = sin x` on `[-pi, pi]`),
  held-out `max(|u-hat u|, |u'-hat u'|)` is strictly below a value-only
  transformer of the same width (five seeds, median) **and** below `1e-3`.
  Skill vs the zero predictor is positive. If the jet-token loses on
  the joint jet error, G2 fails.
- **G3 honesty.** Artifact field `imagenet_claim` is `false`.
- **G4 parity.** torch / jax bit-identical on the worked example.

## 9. Benchmark plan

- `benchmarks/jet_token_transformer.py`
- Smoke: `docs/benchmarks/jet_token_transformer_smoke.json`
- `--full`: `$OMNIBIAS_SCRATCH/inventions/jet_token/`
- CI smoke after implementation.

## 10. Honesty and scope

- Bias collapse supplies `sigma^(n)`. No temperature collapse in the
  default mix.
- Exactness is of the **model** jet (same as 02-03), not the target.
- Not CCF stretch. Not ImageNet. Not Jet-KAN.
- Certificate tier: empirical.

## 11. Open questions and risks

- **Cost.** `O(N^2)` Faà di Bruno per mix may lose iso-wall to a
  value net plus one JVP.
- **Falsifier.** G2 failure means jet tokens are not worth the mix;
  keep as a research hook or drop the "predict contact" claim.

## 12. Implementation checklist

- [x] `omnibias.{torch,jax}.architectures` jet-token module
- [x] Tests vs `compose_jet` on the worked example
- [x] `benchmarks/jet_token_transformer.py` plus smoke JSON
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
