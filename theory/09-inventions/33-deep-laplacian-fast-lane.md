# 09-33 Deep-network Laplacian fast lane

## 1. Thesis and status

Deep MLP ansätze need `Delta^k u` without materialising every mixed partial
to order `2k`. This spec ships a three-tier dispatch
(`omnibias.core.contraction.select_mode`) on the existing torch/jax twins:
Tier A is an exact forward-Laplacian recursion for `k = 1` at any `D`; Tier B
is an exact support-grouped local multivariate jet for `k >= 2` when the
design fits a budget; Tier C is an unbiased sphere estimator with a reported
standard error and a sound `hoeffding_enclosure` interval when the budget is
exceeded.

- **Status**: shipped (G1–G5 CI; no new package; founding bias collapse, not
  temperature collapse)
- **Depends on**: 01-01, 01-10, 06-02
- **Blocks**: none

### Operator card

- **Benefit.** Exact and ceiling-free for the Laplacian; exact-or-enclosed at
  any dimension for `Delta^k`, without `comb(D + 2k, D)` multi-index blowup.
- **How it works.** Tier A carries one Jacobian and one scalar Laplacian per
  hidden unit through the layer chain. Tier B groups the multinomial terms of
  `(sum_i d_i^2)^k` by support set and reads rows off a local multivariate jet
  of dimension `|S| <= k`. Tier C estimates the same contraction on the sphere
  when the support design exceeds `DEFAULT_SUPPORT_BUDGET`.
- **Strength.** Removes `MAX_MULTI_INDICES` for Laplacian / poly-Laplacian on
  linear-chain MLPs; measured 13× CPU win vs nested AD at `D = 256`; succeeds
  at `D = 5000` where `mlp_jet_mv` refuses.
- **When to use.** High-dimensional PINN residuals needing `grad u`, `Delta u`,
  or `Delta^k u` on deep `JetMLP` / `FourierFeatureMLP` / DeepONet trunk /
  Mscale band mixtures.
- **When not.** General mixed partials beyond a Laplacian contraction;
  convolutions, normalisation, RNNs, skip-DAGs; `AttentionJetMLP` softmax
  readouts; claiming O(1) cost at arbitrary order or bit-identity across
  backends for the full recursion.
- **Accuracy floor.** Tier A and Tier B are exact in float64. Tier C is exact
  only in expectation; the shipped report carries stderr and a sound interval.

## 2. Where it lands

- `omnibias.core.contraction` — pure-Python combinatorics, mode selection,
  support-term tables, normalizer.
- `omnibias.core.verified.sampled` — `ConcentrationReport`,
  `hoeffding_enclosure` (Tier C).
- `omnibias.{torch,jax}.laplacian` — `deep_field_value_grad_laplacian`,
  `deep_field_laplacian`, `deep_field_polylaplacian`,
  `deep_field_polylaplacian_with_report`, `restrict_first_layer`.
- `omnibias.pinn.{torch,jax}.fields.jet_mlp` — field rewire plus
  `_fast_lane_layer_groups` / `_fast_lane_contract` hooks.
- `omnibias.pinn.operator.{torch,jax}.deeponet` — DeepONet trunk fast lane.
- `benchmarks/deep_laplacian_scaling.py` — G1–G5 gates.

No new package: the domain is the existing torch/jax laplacian and PINN field
surface.

## 3. Prior art in omnibias

- `omnibias.{torch,jax}.jet_mv.mlp_jet_mv` — full multivariate jet oracle;
  refuses past `omnibias.core.multi_index.MAX_MULTI_INDICES`.
- `omnibias.{torch,jax}.laplacian.neural_field_*` — one-hidden-layer closed
  form; does not compose through depth.
- `omnibias.core.multi_index` — combinatorial ceiling and configurable budget.
- `omnibias.core.cubature` (03-06) — neural quadrature; **distinct** from Tier
  B support-grouped jets (vocabulary collision resolved by hard rename).
- `omnibias.pinn.operator.deeponet` — branch contraction linear in trunk basis;
  fast lane runs on trunk layers then `einsum` with `coeffs`.

**Confirmed gap before 09-33.** No deep-network Laplacian path avoided the
multi-index ceiling; PINN fields fell back to `mlp_jet_mv` or per-partial
`_partial` loops.

## 4. Mathematics

For a deep chain `f = W_L sigma(... sigma(W_1 x + b_1) ...) + b_L`, the
Laplacian satisfies a forward recursion carrying `J = nabla f` and
`L = Delta f` per hidden unit. At each layer with pre-activation `u = W a + b`
and `a' = sigma(u)`:

```
J' = sigma'(u) * (W @ J)
L' = sigma''(u) * sum_h (J_u,h)^2 + sigma'(u) * (W @ L)
```

Every `sigma'`, `sigma''` comes from the founding `delta -> 0` bias collapse
(the closed-form polynomial fastpath), **not** temperature collapse (`beta -> inf`).

For `k >= 2`, expand `Delta^k = (sum_i d_i^2)^k` via multinomial coefficients,
group terms by the support set `S` of participating partial indices, restrict
the first affine layer to coordinates in `S`, and read the diagonal
`(2,2,...,2)` row off a local `|S|`-dimensional jet. This is exact — no sphere,
no quadrature error. When the number of support evaluations exceeds
`DEFAULT_SUPPORT_BUDGET`, Tier C uses the spherical identity

```
Delta^k f(x) = C(D,k) * E_{v ~ S^{D-1}}[(v . nabla)^{2k} f(x)]
```

estimated by sampled directions; exact only in expectation.

## 5. Worked example

`D = 4`, depth 3, hidden 6, `tanh`, `float64`, seed 0. Tier A
`deep_field_laplacian(x, layers)` agrees with the `mlp_jet_mv` order-2 oracle
to `1e-9` absolute. At `D = 5000`, depth 2: Tier A succeeds;
`mlp_jet_mv(x, layers, 2)` raises `ValueError` (12.5M multi-indices needed,
budget 200000).

## 6. Proposed API

**Shipped** (this section records the live surface):

```python
from omnibias.core.contraction import select_mode, support_jet_count
from omnibias.torch.laplacian import (
    deep_field_laplacian,
    deep_field_polylaplacian,
    deep_field_polylaplacian_with_report,
    restrict_first_layer,
)
```

Torch and JAX twins are numerically tight at float64 (`~1e-15` absolute on
tested configs) but **not** bit-identical: intermediate `tensordot` / `sum`
steps pick up 1–2 ULP between backends. Default dtype is the framework default.

## 7. Practical use cases

1. **High-D Poisson PINN.** `Delta u = f` with `D` in the thousands: Tier A
   replaces a nested Hessian trace that costs `O(D^2)` and blows memory.
2. **Biharmonic / poly-harmonic.** `Delta^2 u` or `Delta^k u` on a deep MLP:
   Tier B exact to moderate `D`, Tier C beyond with a sound interval.
3. **DeepONet trunk.** Branch-contracted operator Laplacian via trunk fast lane
   plus `einsum` with per-sample `coeffs`; bias drops from derivatives.
4. **Mscale spectral bias.** Band mixture `sum_j f_j(alpha_j x)` via
   `_band_layer_specs()` without multi-index ceiling.
5. **Mixed spatial + time PINNs.** `restrict_first_layer` plus batched Tier A
   on spatial axes only.

## 8. Acceptance gates

Implemented in `benchmarks/deep_laplacian_scaling.py`:

| Gate | Claim |
|---|---|
| G1 | Tier A matches `mlp_jet_mv` Laplacian oracle at small `D` |
| G2 | Tier A succeeds at `D = 5000`; `mlp_jet_mv` raises |
| G3 | Tier A beats nested `torch.func.hessian` trace (cost parity) |
| G4 | Torch vs JAX tight float64 tolerance; **not** bit-exact |
| G5 | Tier C sample mean within stderr of Tier B exact value |

## 9. Benchmark plan

- Smoke: `docs/benchmarks/deep_laplacian_scaling_smoke.json` via
  `uv run python benchmarks/deep_laplacian_scaling.py` (CI CPU job).
- Full: `$OMNIBIAS_SCRATCH/citation/deep_laplacian/deep_laplacian_scaling.json`
  via `--full` on the GPU cluster (`-app benchmark`).

## 10. Honesty and scope

**Permitted:** exact and ceiling-free for the Laplacian; exact-or-enclosed at
any dimension for `Delta^k` (Tier B exact when budget allows; Tier C exact in
expectation with reported stderr and `hoeffding_enclosure`).

**Forbidden:**

- "O(1) at arbitrary order" — Tier A is `O(B*H*D)` per layer; Tier B grows
  with support count; Tier C grows with `n_directions`.
- Unqualified "no ceiling" — dense Hessian via `hessian_full` still hits the
  multi-index budget near `D ~ 631`; `AttentionJetMLP` keeps the ceiling.
- Bit-identity across torch/JAX for the full deep recursion — measured 1–2 ULP
  gap from differing reduction order.
- Calling Tier B a "sphere cubature" — it is support-grouped local jets; no
  quadrature error.

Founding bias collapse supplies `sigma'` / `sigma''`; this is not temperature
collapse. No `theorem_prover_verified` or `mathlib_verified` tier is claimed.

## 11. Open questions and risks

- Dense Jacobian memory `O(B*D*H)` is inherent to Tier A; `chunk_size` may be
  needed if measurements show it binding on GPU.
- ReLU and piecewise activations: order `>= 2` is almost-everywhere zero but
  was a silent footgun; kernels now refuse unless explicitly opted in.
- Float64 representability limits usable jet order despite `MAX_ORDER = 512`.

## 12. Implementation checklist

- [x] `omnibias.core.contraction`
- [x] `omnibias.{torch,jax}.laplacian` deep-field kernels
- [x] PINN field rewire and DeepONet / Mscale hooks
- [x] `benchmarks/deep_laplacian_scaling.py` G1–G5
- [x] `packages/omnibias-{core,jax,torch}/tests/test_deep_laplacian.py`
- [x] `tests/test_deep_laplacian_parity.py`
- [x] `docs/api/deep_laplacian.md` and mkdocs nav
- [x] CI smoke job in `.github/workflows/ci.yml`
- [x] Index row in `theory/README.md`

## 13. Parent problem

High-dimensional physics-informed networks need iterated Laplacians without
the `comb(D + N, N)` multi-index wall of full multivariate jets. This spec
attacks that wall for linear-chain MLP ansätze and their DeepONet / Mscale
extensions; it does not claim a continuum PDE theorem or global regularity.

---

## Repo invariants this spec must respect

- Pure core: `omnibias.core.contraction` has no torch/jax imports.
- Numerical twins: torch and jax agree at tight float64 tolerance; polynomial
  coefficients from `omnibias.core.polynomials` only.
- Default dtype: framework default, never hardcoded `float32`.
- Terminology: founding bias collapse, not temperature collapse.
- Tier B renamed away from `cubature` to avoid collision with 03-06.
