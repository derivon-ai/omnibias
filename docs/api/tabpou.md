# TabPOU (05-04)

A from-scratch tabular model: **axis-aligned** oblivious partition-of-unity
trees, **closed-form Newton leaves**, optional **band/integral** numerical
embeddings, and an optional TabM-style residual. The split gate
`sigmoid(beta (x[f] - t))` hardens as `beta -> inf` -- **temperature
collapse** (feasibility sense), not the founding `delta -> 0` bias collapse.

Status is **gated**. `benchmarks/tabular_pou.py --full` is locked
([`tabular_pou.json`](../benchmarks/tabular_pou.json)): G0 / G0b / G5 / G6
passed; G1 / G2 / G3 failed their predeclared bars (G3 not-worse-both
`3/6`, need `>= 5/6`); G4 leftover-recorded (`n_comparable=3` because the
RealMLP/TabM harness is regression-only). Boost-only TabPOU is not a GBDT
win. G3 stays the named v0 arm for 05-05 and is not retuned. This is
league-1 (train from scratch: GBDT / RealMLP / TabM). It does not claim
TabPFN-3 / EXAONE / TabFM Elo. Soft-tree training is a surrogate; the
licensed deployable object is the **hard** tree plus a sound
`certify_tab_gap`, or the soft model if the gap is unused. Theory spec:
[05-04](https://github.com/derivon-ai/omnibias/blob/main/theory/05-applications/04-tabpou-architecture.md).
Theory 05-05 (`fit_tabpou_joint`) is **shipped** on the same submodule
(G-tree `3/3`, G-net `2/3`, G-hybrid `6/6`, G5 sound). 05-04 G3 stays the
named boost-only **v0** arm and is not retuned. See
[05-05](https://github.com/derivon-ai/omnibias/blob/main/theory/05-applications/05-tabpou-joint-worlds.md).

`fit_boosted`'s default `leaf_solver="adam"` is unchanged, so the 05-02
flagship LightGBM table is not silently replaced.

## Closed-form Newton leaves

Leaves enter linearly through the POU memberships `P`. Given a Newton
residual `r = -g/h` and Hessian weights `h`, each tree is an `L x L`
weighted ridge (`L = 2**depth`):

```python
import numpy as np
from omnibias.tab import SoftTreeConfig, init_params
from omnibias.tab._core.forward import leaf_memberships
from omnibias.tab._core.leaves import closed_form_leaves, newton_leaf_loss

rng = np.random.default_rng(0)
cfg = SoftTreeConfig(n_features=3, n_trees=1, depth=1, task="regression", n_outputs=1, seed=0)
params = init_params(cfg, rng)
X = rng.standard_normal((20, 3))
r = rng.standard_normal((20, 1))
h = np.ones((20, 1))
P = leaf_memberships(params, X, cfg.beta_final)
leaves = closed_form_leaves(P, r, h, leaf_l2=1e-4)
assert leaves.shape == (1, 2, 1)
assert newton_leaf_loss(P, r, h, leaves) < newton_leaf_loss(P, r, h, np.zeros_like(leaves))
```

## Axis `split_kind`

`SoftTreeConfig.split_kind` is `"oblique"` by default (05-02 flagship).
`"axis"` stores a one-hot feature selector per gate:

```python
import numpy as np
from omnibias.tab import SoftTreeConfig, init_params

cfg = SoftTreeConfig(n_features=5, n_trees=2, depth=2, split_kind="axis", task="binary", seed=3)
W = init_params(cfg, 3).W
for m in range(2):
    for j in range(2):
        assert int(np.count_nonzero(np.abs(W[m, j]) > 1e-15)) == 1
```

## `fit_tabpou`

Preprocessor (robust scale + low-card one-hot) is fit on train only. Band
thresholds are frozen quantile edges. Each stage grows one axis-aligned
oblivious tree with closed-form leaves.

```python
import numpy as np
from omnibias.tab.pou import TabPOUConfig, fit_tabpou

rng = np.random.default_rng(1)
X = rng.standard_normal((40, 3))
y = np.sin(X[:, 0]) + 0.3 * X[:, 1]
cfg = TabPOUConfig(
    n_features=3, task="regression", depth=1, n_stages=4, n_quantiles=6,
    n_bins=4, use_embed=False, onehot_max_card=1, seed=1, patience=None,
)
model, result = fit_tabpou(X, y, cfg)
pred = model.predict(X)
assert pred.shape == (40,)
assert result.history[-1] <= result.history[0]
```

Turn `use_embed=True` for the band+raw front-end (RealMLP's PBLD trick in
tower language). Turn `use_residual=True` for the k-head BatchEnsemble
residual on the same POU weights (`RegionModels.combine`). JAX is
forward-only: `omnibias.tab.jax.pou.pou_forward_arrays`.

## Joint trainer (05-05)

`fit_tabpou_joint` keeps the v0 grower, then optionally polishes thresholds
`t` with `block_exact_search` (`verify=True`, never-worse), unfreezes
`BandFeatureEmbedder`, and jointly trains a POU-gated residual. Gate
hardening is still temperature collapse (`beta -> inf`). `W` stays one-hot.

```python
import numpy as np
from omnibias.tab.pou import TabPOUConfig, TabPOUJointConfig, fit_tabpou_joint

rng = np.random.default_rng(2)
X = rng.standard_normal((32, 3))
y = np.sin(X[:, 0]) + 0.2 * X[:, 1]
base = TabPOUConfig(
    n_features=3, task="regression", depth=1, n_stages=3, n_quantiles=5,
    n_bins=3, use_embed=False, onehot_max_card=1, seed=2, patience=None,
)
model, result = fit_tabpou_joint(
    X, y, TabPOUJointConfig(base=base, polish_thresholds=True, polish_sweeps=2),
)
assert np.isfinite(model.predict(X)).all()
assert result.polished
assert result.polish_loss_ratio <= 1.0 + 1e-8
```

`benchmarks/tabular_pou_joint.py` is the 05-05 harness. Status is
**shipped**: G-tree `3/3`, G-net `2/3`, G-hybrid `6/6`, G5 sound
([`tabular_pou_joint.json`](../benchmarks/tabular_pou_joint.json)).
The frozen combo is v0 boosting plus sequential TabM residual (H1);
threshold polish / trained embed / joint residual / pairwise warm-start
did not survive Phase A. 05-04 G3 remains report-only v0.

## Benchmark harness

`benchmarks/tabular_pou.py` follows the repo `--full` convention. Smoke
(default) is a wiring gate (G0 + G5). `--full` is the 5-seed scientific
run. G0b reports the current `fit_boosted` at flagship `TabConfig` vs the
05-03 under-budget; it is not a TabPOU pass/fail. G6 does not replace the
flagship LightGBM table. RealMLP / TabM stay `--full`-only (no pytabkit in
CI).

::: omnibias.tab.pou.config
    options:
      show_root_heading: false
      heading_level: 3

::: omnibias.tab.pou.preprocess
    options:
      show_root_heading: false
      heading_level: 3

::: omnibias.tab._core.leaves
    options:
      show_root_heading: false
      heading_level: 3
