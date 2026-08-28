# 05-05 TabPOU joint worlds: tree+net with unique tower levers

## 1. Thesis and status

A joint TabPOU matches the **tree world** on tree-shaped tasks and the **net
world** on net-shaped tasks, using only levers this tower uniquely has (exact
`sigma'` on gates, closed-form band/integral tokens, POU
`RegionModels.combine`, sound `certify_tab_gap`).

- **Status**: shipped (G-tree / G-net / G-hybrid / G5 earned on
  `docs/benchmarks/tabular_pou_joint.json`; frozen combo is v0 + sequential
  TabM residual after H1 accept / H6 reject; H0/H3/H4/H6 rejected; H2/H5
  inconclusive; H7 report-only; 05-04 G3 stays report-only v0)
- **Depends on**: 05-04 (`omnibias.tab.pou` v0 boost-only), 05-02 (pairwise
  warm-start, `make_axis_rule` / `make_oblique_xor`), 01-03 (`RegionModels`),
  08-07 (`block_exact_search`), 03-13 / binary (`binarize01`)
- **Blocks**: none

05-04 G3 stays **boost-only** and is not retuned after `--full`. That JSON
(`docs/benchmarks/tabular_pou.json`, 2026-08-28; G3 not-worse-both `3/6`)
is the named **v0** arm, not a ship bit for this spec. 05-04 status is
**gated**.

## 2. Where it lands

A submodule of the existing `omnibias-tab` package: `omnibias.tab.pou.joint`
(`TabPOUJointConfig`, `fit_tabpou_joint`). No new distribution. Same domain,
same audience, same dependency tier as `omnibias.tab.pou` (06-03 earn-
independence fails).

## 3. Prior art in omnibias

Already shipped, reused:

- `omnibias.tab.pou.train.fit_tabpou` -- axis greedy + closed-form Newton
  leaves + optional sequential TabM residual. This is **v0**. Confirmed gap:
  G3 turns `use_residual=False`; embedder is frozen
  (`learnable_beta=False`, `learnable_thresholds=False`); gates freeze after
  the quantile scan; axis greedy scans every band-token column.
- `omnibias.tab.torch.embed.BandFeatureEmbedder` -- already supports
  `learnable_beta=True` / `learnable_thresholds=True`. v0 never trains them.
- `omnibias.torch.optim_block_search.block_exact_search` (08-07) --
  never-worse line search on a named block. Confirmed gap: unused on TabPOU
  thresholds `t`.
- `omnibias.binary.torch.ops.quantize.binarize01` -- hard `{0,1}` forward,
  Eulerian `sigmoid'` backward. Confirmed gap: TabPOU eval stays soft
  `sigmoid(beta z)` at `beta=8`.
- `omnibias.tab.torch.train.fit_joint` -- `head(encoder(X))` with
  TrustRegionNewtonCG. Confirmed gap: residual is 40 sequential Newton steps
  on `y - F_boost`, trees frozen.
- `omnibias.tab.torch.arrangement._sparse_warmstarts` -- feature-pair +
  threshold grid. Confirmed gap: TabPOU grow is depth-by-depth greedy only.
- `omnibias.tab.certify.certify_tab_gap` -- sound `|F_soft - F_hard|`.

Confirmed absences (do not re-derive): CatBoost ordered target statistics,
LightGBM leaf-wise growth, TabPFN-3 ICL. Those are leftover-not-this-spec
(GBDT / foundation engineering, not the unique combo).

## 4. Mathematics

**Limit in play:** temperature collapse. `g = sigmoid(beta (x_f - t))` hardens
as `beta -> inf`. Founding bias collapse (`delta -> 0`) does **not** appear.

**Threshold polish.** Frozen one-hot `W` and leaves; `t` is a vector in
`R^{T D}`. One 08-07 block search along `-grad_t L` with `verify=True`
never-worse. The loss is the same Newton weighted MSE the grower used.

**Hard eval.** `binarize01(z)` is `1[z > 0]` in the forward and
`d/dz sigmoid(beta z)` in the backward. Deploy scores may use
`hard_forward_np` (`G = 1[z > 0]`). The licensed gap is still
`certify_tab_gap`.

**Grouped axis pick.** Band tokens for source feature `f` occupy columns
`f * (n_bins+1) : (f+1)*(n_bins+1)` plus optional raw column
`d_src * (n_bins+1) + f`. A grouped split samples **source features**, then
scans only that group's columns, so the greedy does not treat 17 collinear
bins as 17 unrelated coordinates.

**Pairwise warm-start (depth 2).** Score candidate pairs `(f0, t0), (f1, t1)`
by the closed-form leaf loss at **full** depth, keep the minimizer. This is
the 05-02 pair grid with Newton leaves.

**Joint residual.** After boosting, the graph is
`F = forest(tokens(embedder(x))) + TabMResidual(tokens, POU)`. `fit_joint`
moves embedder (if unfrozen), `t`, and residual parameters; `W` stays one-hot.

## 5. Worked example

One axis stump, two points, `leaf_l2=0`, `beta -> inf`:

```
X = [[0.0], [2.0]], r = [[0.0], [2.0]], h = [[1.0], [1.0]]
greedy: f=0, t=1.0, leaves = [0.0, 2.0]
```

A 08-07 polish of `t` on this hard problem is a no-op (the split is already
the unique minimizer). Soft `beta=8`: `g(0)~0.0003`, `g(2)~0.9997`; polish
must not increase Newton loss (never-worse). Numpy check: call
`polish_thresholds` then `newton_leaf_loss`; the ratio vs pre-polish is
`<= 1 + 1e-12`.

## 6. Proposed API

Dtype: tab float64 certify convention. JAX twin remains forward-only
(`pou_forward_arrays`).

```python
from omnibias.tab.pou import TabPOUConfig, fit_tabpou
from omnibias.tab.pou import TabPOUJointConfig, fit_tabpou_joint

base = TabPOUConfig(n_features=d, task="regression", use_embed=False, n_stages=8)
cfg = TabPOUJointConfig(
    base=base,
    polish_thresholds=True,
    train_embed=False,
    grouped_splits=False,
    joint_residual=False,
    binarize_eval=False,
    pairwise_warmstart=False,
)
model, result = fit_tabpou_joint(X, y, cfg)
```

v0 `fit_tabpou` is unchanged when joint flags are off.

## 7. Practical use cases

1. **Tree-shaped public tables** (`breast_cancer`, `banknote`, `ionosphere`).
   Axis Newton boosting should match LightGBM/CatBoost. The joint residual
   must not destroy this (G-hybrid).
2. **Net-shaped regression** (`kin8nm`, `energy_efficiency`,
   `mlp_heteroscedastic_20d`). Sequential or joint TabM-on-POU should close
   the MLP gap that boost-only cannot.
3. **Planted axis-AND.** 05-04 G1; joint must not regress vs v0.
4. **Planted XOR / pair.** Greedy depth-by-depth vs pairwise warm-start (H5).
5. **Certified deploy.** Train soft (optional `binarize01` backward), ship
   hard, quote `certify_tab_gap`.

## 8. Acceptance gates

Family split is **predeclared** (not after looking at test):

- Tree-shaped: `make_axis_rule`, `breast_cancer`, `banknote`, `ionosphere`.
- Net-shaped: `mlp_heteroscedastic_20d`, `kin8nm`, `energy_efficiency`.
- Mixed probe (report-only): `wine_quality`.

K=5 on the lock; Phase A uses 3 seeds and two public probes
(`breast_cancer`, `kin8nm`) plus synthetics. Val-only decisions. No
6-dataset fishing. Follow `benchmarks/_gates.py`: finite scores, skill vs
mean predictor, then the named baseline. Not-worse = mean metric within the
baseline's across-seed std (higher-is-better primary).

**Phase 0 (v0 freeze).** `benchmarks/tabular_pou.py --full --skip-g4`.
05-04 G3 is report-only. Do not retune v0.

**Hypothesis battery (Phase A), each `accepted` / `rejected` /
`inconclusive`:**

- **H0 Newton leaves.** Same axis greedy; `leaf_solver=closed_form` vs
  `adam`. ACCEPT if CF not-worse on both probes and wall-clock `<=` adam;
  REJECT if within noise on the metric.
- **H1 residual-on.** v0 boost-only vs `use_residual=True` (sequential TabM).
  ACCEPT if kin8nm RMSE drops `>= 5%` relative **and** breast_cancer accuracy
  drop `<= 1` pt; REJECT if kin8nm within noise.
- **H2 threshold polish.** After greedy freeze, polish only `t` with
  `block_exact_search`. ACCEPT if `make_axis_rule` worst-seed margin vs v0
  `>= 0.02` **or** (not-worse on both probes and strictly better on one);
  REJECT if polish is within 1% relative.
- **H3 trained vs frozen embed.** Arms: `none` / frozen band / trained band /
  trained `integral`. ACCEPT trained-band if not-worse than raw on both
  `mlp_heteroscedastic_20d` and `kin8nm` (worst-seed RMSE). REJECT if frozen
  or none wins -- then the lock config must **not** 17x `d`.
- **H4 d-inflation.** Frozen 16-bin concat-raw vs grouped source-feature
  pick vs `colsample=0.5`. ACCEPT grouping or colsample if kin8nm RMSE or
  greedy train-loss improves vs naive 17x scan; else leftover wall-clock.
- **H5 pair-miss.** `make_oblique_xor` plus `make_axis_rule` with distractors;
  greedy vs pairwise warm-start. ACCEPT only if greedy fails
  (`margin < 0.05` vs majority) **and** warm-start recovers. If greedy already
  wins, REJECT this as the public-suite explanation.
- **H6 sequential vs joint.** `fit_joint` on embedder + `t` + residual vs
  sequential 40 residual steps. ACCEPT if G-hybrid holds on the two probes
  **and** kin8nm beats sequential; REJECT if joint wrecks breast_cancer.
- **H7 capacity (report-only).** `n_stages=60` vs `200`, `colsample=1` vs
  `0.5`. Do **not** promote 200-stage boost-only as 05-05 shipped.

**Constructive pick (one pass):** from Phase A **val** curves, freeze exactly
one `TabPOUJointConfig`. No second public-suite retune (05-02 leftover #49).

**Lock bars (Phase B, K=5):**

- **G-tree:** joint not-worse vs LightGBM **and** CatBoost on `>= 3/3` public
  tree-shaped rows; worst-seed `make_axis_rule` accuracy still `>=` v0 G1.
- **G-net:** joint not-worse vs RealMLP **and** TabM on `>= 2/3` net-shaped
  rows (`kin8nm`, `energy_efficiency`, `mlp_heteroscedastic_20d`).
  Classification RealMLP/TabM stay out of this bar.
- **G-hybrid:** on every public lock row, joint is not-worse than v0
  boost-only. Failure **rejects** "best of both worlds".
- **G5:** `certify_tab_gap(...).is_sound` on the joint axis forest.
  Failure is spec-killing.
- **Not in `all_passed`:** wall-clock; TabPFN-3; ordered TS; leaf-wise;
  05-04 G3 (report-only).

Smoke (default): H0 wiring + tiny H1 + G5. Only G5 (and H0 identity) fail
the process. `--full` is Phase A. `--lock` is Phase B.

**Phase A (val-only, 3 seeds; `docs/benchmarks/tabular_pou_joint_full.json`):**

- H0 **rejected** -- closed-form leaves not-worse on the probes but not
  faster than adam (`seconds_cf` 8.8 vs 7.1).
- H1 **accepted** -- kin8nm RMSE ratio `0.67` (>= 5% drop) and breast_cancer
  accuracy did not drop (mean drop `-0.6` pt). This is the frozen lever.
- H2 **inconclusive** -- axis-AND worst-seed margin `-0.005`; polish did not
  clear `0.02` or the 1% reject band.
- H3 **rejected** -- trained-band lost the synth worst-seed RMSE to raw;
  lock config must **not** 17x `d`.
- H4 **rejected** -- grouped RMSE identical to naive concat-raw.
- H5 **inconclusive** -- greedy XOR margin `-0.015` (fails) but pairwise
  `0.020` did not recover `>= 0.05`. Pair-miss is not the public-suite
  explanation.
- H6 **rejected** -- joint residual beat sequential on the net probe but
  wrecked the tree probe. Sequential residual (H1) is kept.
- H7 report-only -- `n_stages=200` did not beat 60 on axis-AND (`0.99` vs
  `0.995`). Not promoted.

**Frozen combo:** `use_residual=True` (sequential TabM-on-POU),
`n_stages=60`, no embed, no `t`-polish, no pairwise, no `fit_joint`.

**Phase B (K=5; `docs/benchmarks/tabular_pou_joint.json`):** G-tree `3/3`
not-worse vs LightGBM and CatBoost, G1 margin `0` vs v0; G-net `2/3`
not-worse vs RealMLP and TabM (`energy_efficiency` +
`mlp_heteroscedastic_20d`; **kin8nm lost** to both DL baselines); G-hybrid
`6/6` not-worse than v0 boost-only; G5 `is_sound`. kin8nm vs RealMLP/TabM
is a row loss inside a passing family bar, not a leftover-recorded G-net
fail.

## 9. Benchmark plan

- Script: `benchmarks/tabular_pou_joint.py` (`--full`, `--lock`, `--workers`).
- Smoke JSON (committed): `docs/benchmarks/tabular_pou_joint_smoke.json`.
- Phase A JSON: `$OMNIBIAS_SCRATCH/tabular_pou_joint_full.json` (copy:
  `docs/benchmarks/tabular_pou_joint_full.json`).
- Phase B JSON: `$OMNIBIAS_SCRATCH/tabular_pou_joint_lock.json` (copy:
  `docs/benchmarks/tabular_pou_joint.json`).
- v0 JSON: `$OMNIBIAS_SCRATCH/tabular_pou_full.json` (05-04 harness;
  `docs/benchmarks/tabular_pou.json`).
- CI: `tab` job step, `JAX_PLATFORMS=cpu python benchmarks/tabular_pou_joint.py`.
- No pytabkit in CI. RealMLP / TabM stay `--lock` / `--full` only.

## 10. Honesty and scope

- Joint TabPOU is a **from-scratch** model. It does not claim TabPFN-3,
  EXAONE Tabular, or TabFM Elo.
- Temperature collapse (`beta -> inf`) is the gate limit. Founding bias
  collapse is not this spec.
- 05-04 G3 (boost-only vs both GBDTs; `--full` locked at not-worse-both
  `3/6`) is **report-only**. This spec does not silently replace it.
- "Best of both worlds" is G-tree **and** G-net, with G-hybrid as the
  anti-slogan: the mix must not destroy the tree half.
- If G-tree fails after H2+H5: leftover-record; do not add ordered TS.
  (Did not fire: G-tree `3/3`.)
- If G-net fails after H1+H3+H6: leftover-record; certified v0 booster
  remains 05-04. (Did not fire: G-net `2/3`. kin8nm vs RealMLP/TabM is a
  row loss, not a family leftover.)
- If G-hybrid fails: **reject** the combo. (Did not fire: `6/6` vs v0.)
- No P vs NP, no AutoGluon, no ImageNet, no new package.
- `theorem_prover_verified` stays false unless a genuine `lake build`
  (not required). `mathlib_verified` stays false.

## 11. Open questions and risks

- Joint training of `t` + residual can erase axis one-hots if `W` is left
  unfrozen. The implementation freezes `W`.
- Trained band tokens may still inflate `d`; H3/H4 must win before the lock
  uses them.
- `binarize01` eval can move accuracy relative to soft `beta=8`; G5 must
  still sandwich the hard tree.
- kin8nm vs RealMLP/TabM remained a row loss after H1 (G-net still
  shipped on `2/3`). That is allowed; it is not a family leftover.
- Phase A 3-seed probes can disagree with Phase B 5-seed lock. The lock is
  the licensed number; Phase A is not relicensed after seeing test.

## 12. Implementation checklist

- [x] `theory/05-applications/05-tabpou-joint-worlds.md` and index row
- [x] Freeze 05-04 G3 as report-only v0 (honesty paragraph)
- [x] `omnibias.tab.pou.joint` (`TabPOUJointConfig`, `fit_tabpou_joint`,
      `polish_thresholds`, grouped splits, pairwise grow)
- [x] `make_embedder(..., freeze=)` and `fit_tabpou(..., feature_groups=)`
- [x] `benchmarks/tabular_pou_joint.py` H0–H7 + G-tree/G-net/G-hybrid/G5
- [x] Tests: polish never-worse; joint loss drop; G5; grouped columns
- [x] `docs/api/tabpou.md` pointer, mkdocs nav, CHANGELOG
- [x] CI smoke step on the `tab` job
- [x] regenerate `__all__` on touched `__init__.py`
