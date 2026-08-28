# 05-04 TabPOU: axis-aligned POU boosting with neural leaves

## 1. Thesis and status

A from-scratch tabular model that keeps the tree inductive bias (axis-aligned
oblivious splits, Newton boosting, Hessian weights) and the neural inductive
bias (band/integral numerical embeddings, optional TabM-style residual heads),
wired through omnibias's exact gate curvature and a sound soft-to-hard rounding
gap.

- **Status**: gated (`benchmarks/tabular_pou.py --full` has run end to end;
  every measured number is in section 8 and
  [`docs/benchmarks/tabular_pou.json`](../../docs/benchmarks/tabular_pou.json).
  **G0 / G0b / G5 / G6 passed.** **G1 / G2 / G3 failed** their predeclared
  bars (axis always won G1 but the worst-seed margin was `0.022`, need
  `>= 0.05`; G2 skill vs mean predictor failed on
  `mlp_heteroscedastic_20d`; G3 not-worse-both `3/6`, need `>= 5/6`).
  **G4 leftover-recorded**: the RealMLP/TabM harness is regression-only, so
  only 3 of 6 rows are comparable (need `>= 4/6`). The code is retained,
  correct, and tested; boost-only TabPOU is not claimed as a GBDT win.
  G3 stays the named **v0** arm for 05-05 and is not retuned.)
- **Depends on**: 05-02 (`omnibias.tab`), 05-03 (`BandFeatureEmbedder`,
  `score_grad_hess`), 01-03 (`omnibias.partition` `split_kind="axis"`),
  01-13 (`band` / `integral` roles)
- **Blocks**: none

This spec targets **league 1** of the 2024–2026 tabular board (train from
scratch: GBDT, RealMLP, TabM). Tabular foundation models (TabPFN-3, EXAONE
Tabular, TabFM) sit in a different league via synthetic-prior in-context
learning and are a recorded Phase-2 leftover, not a claim here.

## 2. Where it lands

A new submodule of the existing `omnibias-tab` package. No new distribution.

- `omnibias.tab._core.config` -- `SoftTreeConfig.split_kind` in
  `{axis, oblique, sparse}`, default `oblique` so the 05-02 flagship is
  unchanged.
- `omnibias.tab._core.leaves` -- numpy `closed_form_leaves` (weighted ridge
  on the membership matrix).
- `omnibias.tab.pou` -- `TabPOUConfig`, `TabPOU`, `TabPreprocessor`,
  `fit_tabpou`, `LinearBatchEnsemble` / `TabMResidual`.
- `omnibias.tab.jax.pou` -- forward twin (no trainer).
- `omnibias.tab.torch.boosting` -- `leaf_solver="adam"|"closed_form"` on
  `fit_boosted`; default stays `"adam"` (G0 adam bit-identity passed;
  flagship `fit_boosted` is not silently replaced).

The 06-03 earn-independence rule fails: same domain, same audience, same
dependency tier as `omnibias-tab`. Folding a submodule out later is cheap.

## 3. Prior art in omnibias

Already shipped, reused:

- `omnibias.tab._core.forward.forward_np` / `leaf_memberships` -- oblivious
  product POU. Confirmed: this *is* CatBoost's decision table.
- `omnibias.tab._core.loss.score_grad_hess` -- closed-form Newton target
  `-g/h` and Hessian weight `h`.
- `omnibias.tab.torch.boosting._fit_weak_learner` -- Adam-MSE on a fresh
  `SoftTreeEnsemble`. This is the miss: leaves enter **linearly** through
  memberships, so the Newton-optimal leaves are a ridge solve, not 40–60
  Adam steps.
- `omnibias.tab.torch.embed.BandFeatureEmbedder` -- differentiable PLE with
  a closed-form integral twin. Built for 05-03, never the model front-end.
- `omnibias.partition.PartitionConfig.split_kind` -- `"axis"` already
  exists. Confirmed gap: `SoftTreeConfig` has no `split_kind`; every tab
  forest is dense oblique `W`.
- `omnibias.tab.certify.certify_tab_gap` -- sound `|F_soft - F_hard|`.
- `omnibias.partition.RegionModels.combine` -- per-region blend. Confirmed
  gap: unused as a residual head on a boosted forest.
- `omnibias.torch.optim.TrustRegionNewtonCG` / `CubicGaussNewton` -- residual
  trainer.

Confirmed absences (do not re-derive): FT-Transformer attention (TabM,
ICLR 2025, showed it does not pay), STE-only hard splits (GRANDE; omnibias
already has exact `sigma'`), a TabM BatchEnsemble layer inside tab.

External priors this composition is answering: Grinsztajn et al. 2022
(axis / non-rotational / irregular targets), Gorishniy et al. 2022 (numerical
embeddings), Holzmüller et al. NeurIPS 2024 (RealMLP bag of tricks),
Gorishniy et al. ICLR 2025 (TabM parameter-efficient ensembling), Marton et
al. ICLR 2024 (GRANDE: axis-aligned gradient trees).

## 4. Mathematics

**Limit in play:** temperature collapse. The split gate
`g = sigmoid(beta (x_f - t))` hardens as `beta -> inf` to `1[x_f > t]`.
The founding bias collapse (`delta -> 0` to `sigma^(K-1)`) does **not**
appear.

**Axis gate.** For `split_kind="axis"`, each `(tree, depth)` slot stores a
one-hot direction `W[m,j] = e_{f(m,j)}`, so
`z = x_f - t`. The forward is unchanged: `einsum` with a one-hot `W` *is*
the axis split. The POU membership of leaf `l` is still the soft-AND
`P_l = prod_j (g_j if bit_j(l) else 1 - g_j)`. Rows of `P` are nonnegative
and sum to 1.

**Closed-form Newton leaves.** Frozen memberships `P in R^{n x T x L}`,
Newton residual `r = -g/h in R^{n x k}`, Hessian weights `h in R^{n x k}`.
Per tree `m` and output `o` the unique minimizer of
`sum_n h_{n o} (r_{n o} - P_{n m} . ell)^2 + leaf_l2 ||ell||^2` is the
ridge

```
(A + leaf_l2 I) ell = b,
A = P_m^T diag(h_{:o}) P_m,   b = P_m^T (h_{:o} * r_{:o}).
```

`L = 2**depth` (depth 3 => 8 x 8). Because each tree's memberships sum to
1, a constant sits in the leaf span; weak-learner `b0` is identically 0.

**Greedy oblivious grow.** For depth `j = 0..D-1`, scan a column subsample
and `n_quantiles` empirical quantiles of `x_f`, score each candidate by the
closed-form leaf loss at the *current* depth `j+1`, keep the minimizer,
freeze that gate. This is CatBoost's decision-table grow with an exact leaf
solve instead of an approximate histogram.

**Band front-end.** `BandFeatureEmbedder(role="band")` is the telescoping
POU `sigma(beta (x - t_{j-1})) - sigma(beta (x - t_j))` with open tails.
Concatenating the scaled scalar (RealMLP PBLD) yields tokens
`(E(x), s * x_scaled)`. Axis splits on a bin coordinate are histogram
splits.

**TabM residual.** Shared backbone `W`, per-member rank-1 scales
`(r_k, s_k)` (BatchEnsemble). Prediction is the mean of `k` heads trained
on `y - F_boost`.

**Error.** Soft vs hard: the existing `rounding_gap` bound is
`O(sum_j sigmoid(-beta |z_j|))` by TV subadditivity of the leaf product.
No founding-collapse remainder.

## 5. Worked example

One axis stump, two points, regression, `beta -> inf` (hard), `leaf_l2=0`.

```
X = [[0.0], [2.0]], r = [[0.0], [2.0]], h = [[1.0], [1.0]]
f=0, t=1.0  =>  hard memberships P = [[1, 0], [0, 1]]
leaves = [0.0, 2.0]     (left mean, right mean)
```

Soft at `beta=8`: `g(0)=sigmoid(-8)~0.0003`, `g(2)=sigmoid(8)~0.9997`.
The ridge recovers leaves within `1e-3` of `[0, 2]`. Numpy check: call
`closed_form_leaves` on the soft `P`; the weighted MSE is strictly below
the constant-leaf baseline `mean(r)=1`.

## 6. Proposed API

This API is the contract the implementation must match. Dtype: tab's
existing float64 certify convention (not a hardcoded float32). JAX twin is
forward-only; bit-identical to numpy at `~1e-9` with `jax_enable_x64`.

```python
from omnibias.tab import SoftTreeConfig
SoftTreeConfig(..., split_kind="oblique")  # default; "axis" | "sparse"

from omnibias.tab._core.leaves import closed_form_leaves
# closed_form_leaves(P, residual, weight, leaf_l2) -> (T, L, k)

from omnibias.tab.pou import TabPOUConfig, TabPOU, fit_tabpou
cfg = TabPOUConfig(n_features=d, task="regression", n_outputs=1)
model, result = fit_tabpou(X, y, cfg, val=(X_val, y_val))
F = model.predict(X_te)

from omnibias.tab.torch.boosting import fit_boosted
fit_boosted(..., leaf_solver="adam")       # default, flagship-safe
fit_boosted(..., leaf_solver="closed_form")
```

`TabPOUConfig` defaults: `depth=3`, `n_stages=60`, `trees_per_stage=1`,
`learning_rate=0.3`, `beta_final=8.0`, `split_kind="axis"`,
`use_embed=True`, `concat_raw=True`, `use_residual=False`,
`residual_k=8`.

JAX: `omnibias.tab.jax.pou.pou_forward_arrays` (band tokens + tree).

## 7. Practical use cases

1. **Axis-AND / threshold rules.** Planted `1[x_a > t_a] AND 1[x_b < t_b]`
   with distractor columns. Oblique `w.x` mixes noise; axis TabPOU recovers
   the two coordinates. Win over the current `SoftTreeEnsemble`.
2. **Public binary tables vs LightGBM.** Same split, named LGBM
   (`n_estimators=200, lr=0.05, num_leaves=31`). Newton leaves + 60 stages
   close the 05-03 budget hole.
3. **Public regression vs CatBoost / RealMLP / TabM.** Band tokens give the
   MLP embedding win; residual TabM covers a smooth leftover.
4. **Certified deploy.** Train soft, ship hard, quote `certify_tab_gap`.
   Unique vs GRANDE / NODE / TabM.
5. **Uninformative features.** Column subsample + axis pick never splits a
   junk column. Dense oblique always can.

## 8. Acceptance gates

K=5 seeds, worst-seed aggregation, same train/test split as the named
baseline on that row. Follow `benchmarks/_gates.py`: finite reference,
skill vs mean predictor, then the named baseline.

- **G0 (identity / wiring, CI smoke).** (a) Frozen-gate closed-form leaves
  vs long Adam-on-leaves: Newton train-loss ratio `loss_cf / loss_adam <= 1.01`
  on a tiny synthetic. (b) `fit_boosted(..., leaf_solver="adam")` remains
  bit-identical to today's `fit_boosted`. (c) `split_kind="axis"` init: each
  `W[m,j]` is a one-hot.
- **G0b (budget control, `--full`, report only).** Current `fit_boosted` at
  flagship `TabConfig` (`n_stages=60, depth=2, inner_steps=40`) vs the 05-03
  under-budget (`n_stages=10`). Not a TabPOU pass/fail.
- **G1 (axis prior).** `make_axis_rule` plus two noisy features. Axis
  TabPOU vs oblique `SoftTreeEnsemble` at matched tree-count; axis
  worst-seed accuracy margin `>= 0.05`.
- **G2 (embedding).** Band+raw vs raw-scaled, same booster. Both beat the
  mean predictor (skill > 0). Band not-worse than raw on
  `mlp_heteroscedastic_20d` and one public regression set (worst-seed RMSE).
- **G3 (beat GBDT).** Boost-only TabPOU vs `fit_predict_lightgbm` **and**
  `fit_predict_catboost` on 6 datasets (3 class from
  `ARRANGEMENT_PUBLIC_SUITE`: `breast_cancer`, `banknote`, `ionosphere`;
  3 reg from `NOISE_PUBLIC_SUITE`: `kin8nm`, `wine_quality`,
  `energy_efficiency`): not-worse on `>= 5/6`, strict win on `>= 3/6`,
  worst-seed. Not-worse = mean metric within the baseline's across-seed std.
- **G4 (match tabular DL).** Same suite, residual-on TabPOU vs
  `fit_predict_realmlp` / `fit_predict_tabm` at the 05-03 epoch caps.
  Not-worse on `>= 4/6` vs each. Win/loss/tie table, no aggregate-only
  claim. `--full` only (no pytabkit in CI).
- **G5 (certificate still sound).** `certify_tab_gap(...).is_sound` on a
  trained axis model; depth-1 additive path still certifiable. Failure is
  spec-killing, not a leftover.
- **G6 (no flagship regression).** The 4-dataset LightGBM table in
  `docs/benchmarks.md` is **not** silently replaced. TabPOU is reported
  beside `fit_boosted`. Optional: TabPOU not-worse 4/4 on that smoke suite
  at flagship budget.

Leftover-record, not in CI `all_passed`: wall-clock vs LightGBM; TabPFN-3 /
EXAONE; `k`-sweep of the residual; PLR vs band.

Smoke (default): G0 + tiny G1 wiring. Only G0 or G5 fails the process.
`--full` is the scientific experiment.

### Measured results (`--full`, 5 seeds, worst-seed-of-5)

Run 2026-08-28, `docs/benchmarks/tabular_pou.json` (`generated_utc =
2026-08-28T15:26:46Z`, `elapsed_seconds = 12892`, flagship budget
`n_stages=60`, `depth=3`, `n_quantiles=16`). `gates.all_passed = false`.
`smoke_process_passed = true`. Hardware class in the JSON is commodity
x86-64 CPU / float64. **Not retuned** after this run (section 10).

**G0 passed.** Closed-form vs Adam Newton-loss ratio
`loss_cf / loss_adam = 1.000000000007797` (`<= 1.01`).
`fit_boosted(..., leaf_solver="adam")` bit-identity holds. Axis init is
one-hot.

**G0b (report only) passed as a report.** Current oblique `fit_boosted`
under-budget (`n_stages=10`) RMSE mean `0.180` vs flagship
(`n_stages=60`, `depth=2`, `inner_steps=40`) RMSE mean `0.135`. The 05-03
budget gap is real; this is not a TabPOU pass/fail.

**G1 failed.** Axis TabPOU beat matched-count oblique `SoftTreeEnsemble`
on all 5 seeds, but the worst-seed accuracy margin is `0.022`
(need `>= 0.05`). Axis accuracies `0.986`–`0.994`; oblique
`0.952`–`0.970`. The axis prior is directionally right; the bar is not
met against a 60-stage oblique ensemble.

**G2 failed.** `skill_ok = false` on `mlp_heteroscedastic_20d` (worst-seed
RMSE raw `81.10`, band `81.64`). Band is not-worse than raw on diabetes
(worst-seed RMSE `58.23` vs `61.07`) and slightly worse on the synthetic.
Band+raw is not licensed as a default win.

**G3 failed** (boost-only, no residual). Not-worse vs **both** named
LightGBM and CatBoost on `3/6` (need `>= 5/6`). Strict wins `3/6` (meets
the win count, not the not-worse count). Classification is competitive;
regression trails CatBoost.

| Dataset | Task | TabPOU | LightGBM | CatBoost | not-worse both | strict win |
|---|---|---|---|---|---|---|
| `breast_cancer` | acc | `0.964 ± 0.014` | `0.962 ± 0.009` | `0.966 ± 0.016` | yes | vs LGBM |
| `banknote` | acc | `1.000 ± 0` | `0.992 ± 0.006` | `0.997 ± 0.003` | yes | vs both |
| `ionosphere` | acc | `0.932 ± 0.028` | `0.936 ± 0.026` | `0.932 ± 0.024` | yes | neither |
| `kin8nm` | RMSE | `0.150 ± 0.003` | `0.138 ± 0.004` | `0.116 ± 0.003` | no | no |
| `wine_quality` | RMSE | `0.697 ± 0.014` | `0.685 ± 0.014` | `0.679 ± 0.012` | LGBM only | no |
| `energy_efficiency` | RMSE | `0.472 ± 0.019` | `0.619 ± 0.081` | `0.366 ± 0.027` | LGBM only | vs LGBM |

**G4 leftover-recorded.** Residual-on TabPOU vs named RealMLP / TabM.
`n_comparable = 3` because `fit_predict_realmlp` / `fit_predict_tabm` are
regression-only (05-03 harness); the bar needs not-worse on `>= 4/6`.
On the three regression rows: not-worse RealMLP `3/3`, not-worse TabM
`2/3`. Residual RMSE on `kin8nm` is `0.103` (boost-only G3 was `0.150`).
`energy_efficiency` residual RMSE `0.470` vs RealMLP `4.40` / TabM `2.08`
at the 05-03 epoch caps. Not a 6-row DL win; not retuned into G3.

**G5 passed.** `certify_tab_gap` `is_sound` on depth-1 (`max_gap 2.98`
contains measured `2.52`) and depth-2 (`max_gap 9.42` contains measured
`5.45`). Spec-killing certificate gate is green.

**G6 passed.** Flagship 4-dataset LightGBM table in `docs/benchmarks.md`
is **not** replaced (`flagship_table_replaced = false`). Optional TabPOU
vs LightGBM on that smoke suite is not-worse `4/4`
(`breast_cancer` `0.951`/`0.951`, `wine` `1.0`/`1.0`, `digits`
`0.989`/`0.976`, diabetes RMSE `58.3`/`63.6`).

## 9. Benchmark plan

- Script: `benchmarks/tabular_pou.py` (`--full`, `--workers`).
- Smoke JSON (committed): `docs/benchmarks/tabular_pou_smoke.json`.
- Full JSON: `$OMNIBIAS_SCRATCH/tabular_pou_full.json` (default
  `artifacts/`). A summary may be copied into `docs/benchmarks/` after the
  run.
- CI: `tab` job step, `JAX_PLATFORMS=cpu python benchmarks/tabular_pou.py`.
- No pytabkit in CI. RealMLP / TabM stay `--full`.

## 10. Honesty and scope

- TabPOU is a **from-scratch** model. It does not claim TabPFN-3, EXAONE
  Tabular, or TabFM Elo.
- Temperature collapse (`beta -> inf`) is the gate limit. Founding bias
  collapse is not this spec.
- Soft-tree training is a surrogate. The licensed deployable object is the
  **hard** tree plus a sound rounding-gap certificate, or the soft model
  if the gap is unused.
- "Best of both worlds" is earned per named suite, never as a slogan. If
  G3 fails after the constructive route (axis + closed-form leaves +
  flagship budget + band tokens), the spec stays `gated` and records the
  loss (05-03 precedent).
- No P vs NP, no AutoGluon replacement, no ImageNet, no new package.
- Certificate tier: empirical gates plus the existing sound
  `RoundingGapCertificate`. `theorem_prover_verified` stays false unless a
  genuine `lake build` is run (it is not required here).
- 05-03's recorded G1–G5 verdicts are historical and are not rewritten.
- G3 stays **boost-only**. After `tabular_pou.py --full` the JSON is the
  named **v0** arm for theory 05-05; this spec is not retuned from that
  run. The joint tree+net claim lives in 05-05, not here.

## 11. Open questions and risks

- Greedy oblivious grow can miss a good *pair* of features that only score
  jointly at full depth. Depth-2 XOR is the diagnostic; if G1-style axis-AND
  passes and XOR fails, add a pairwise warm-start (05-02 already has one).
- Band tokens inflate `d` (`n_bins+1` per feature). Greedy over 360
  coordinates may be slower than LightGBM's histogram. Colsample and
  grouping by original feature are the mitigations; wall-clock is leftover.
- Closed-form leaves assume frozen gates. If the residual TabM is the only
  thing that wins G4, the "tree+net" thesis survives but the boost-only G3
  may not.
- One-hot of already-ordinal OpenML columns is a heuristic. True
  categorical IDs would be better; 05-04 does not rewrite 05-03 loaders.
- `leaf_solver="closed_form"` on *oblique* forests is a weaker tree
  (random dense `W`, exact leaves). Do not advertise it as TabPOU.

## 12. Implementation checklist

- [x] `theory/05-applications/04-tabpou-architecture.md` and index row
- [x] `SoftTreeConfig.split_kind` + axis `init_params` + `closed_form_leaves`
- [x] G0 tests (leaves vs Adam, adam bit-identity, axis one-hot)
- [x] `omnibias.tab.pou` (`TabPOUConfig`, `TabPreprocessor`, `TabPOU`,
      `fit_tabpou`, BatchEnsemble residual)
- [x] `fit_boosted(..., leaf_solver=)` default adam
- [x] JAX `pou_forward_arrays` + numpy/torch/jax parity
- [x] `benchmarks/tabular_pou.py` G0–G6, smoke CI
- [x] `docs/api/tabpou.md`, mkdocs nav, `tab.md` pointer, CHANGELOG
- [x] regenerate `__all__` on touched `__init__.py`
- [x] `python -m pytest packages/omnibias-tab/tests -q` (222 passed, 7 skipped)
- [x] G3 frozen as boost-only v0 for 05-05 (no relicense)
- [x] `benchmarks/tabular_pou.py --full` locked; status `gated` from G3 miss
