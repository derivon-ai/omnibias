# omnibias-tab

**Status: Alpha (0.1.0a1).**

Differentiable, **exactly second-order-trained**, and **certified** soft decision-tree
ensembles for tabular data, on the omnibias stack. A decision-tree split is a hard
threshold `1[w.x > t]`; omnibias makes it a **soft oblique gate**
`g(x) = sigmoid(beta * (w.x - t))` and anneals `beta -> inf` to recover a genuine hard
split. Learning an *optimal* hard tree is NP-hard, so -- exactly like the discrete
consumers -- this package answers a **yes-if** rather than claiming the impossible:

> **Yes** you get a tree model you can train end to end with (second-order) gradients
> **and** certify, **if** you accept soft splits annealed toward hard ones (with a
> *certified* soft->hard rounding gap) in place of a combinatorially-exact greedy tree.

## Why this is not "just another GBM"

Even at accuracy parity with LightGBM, a `tab` model is a different object:

1. **End-to-end differentiable trees** -- `forward` is a tensor-in / tensor-out
   layer on `(..., d)` features. Compose with any torch encoder
   (`.to(device, dtype)` then the user's optimizer); JAX kernels
   (`forward_arrays`, `arrangement_forward`, `boosted_forward`) compose the
   same way (`as_head` / `docs/examples/tab_as_layer.py` and
   `tab_as_layer_jax.py`). GBM trees are not differentiable.
2. **Exact second-order training of the *whole* model** (splits included) via
   `omnibias.torch.optim` (`CubicNewton` / `TrustRegionNewtonCG` / `KFAC` /
   `NaturalGradient`) -- not just a per-leaf Newton step. A stagewise **Newton-boosting**
   driver mirrors the GBM recipe with the closed-form loss curvature.
3. **Soundness certificates** (`omnibias.tab.certify`): output bounds, Lipschitz,
   per-feature **monotonicity** (as a *certified* constraint, not a soft prior), an
   optional sealed scalar global-min, and a certified **train-soft / deploy-hard**
   rounding gap as `beta -> inf`.
4. **Bit-identical torch / jax** forward (float64 parity `~1e-9`).

## The two "collapse" axes (honesty)

The gate's `sigmoid(beta * (w.x - t))`, `beta -> inf` is the **feasibility / temperature**
sense of "collapse" -- a soft indicator hardening to a 0/1 step -- **not** the *founding
bias collapse* (the multi-bias `delta -> 0` limit of an `OMBU` to the closed-form
derivative `sigma^(K-1)`; see `docs/theory.md`). The derivative tower is still used: exact
gate derivatives feed the second-order trainer, and the `beta -> inf` limit gets a
certified rounding gap.

## What's in the box

<!-- docs-test: slow -->
```python
import numpy as np
from omnibias.tab import SoftTreeConfig
from omnibias.tab.torch import SoftTreeEnsemble, fit_second_order
from omnibias.tab.certify import certify_tab

rng = np.random.default_rng(0)
X = rng.standard_normal((256, 8)); y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(float)

cfg = SoftTreeConfig(n_features=8, n_trees=16, depth=1, task="binary")  # additive
model = SoftTreeEnsemble(cfg)
fit_second_order(model, X, y, optimizer="trust_region", steps=50)       # exact 2nd order

box = np.stack([X.min(0), X.max(0)])          # feature hyper-box
cert = certify_tab(model.to_params(), box, monotone_features={0: +1})
print(cert.output_bounds, cert.lipschitz, cert.monotone_ok, cert.rounding_gap)
```

- **`SoftTreeConfig`** -- ensemble shape: `n_trees`, `depth` (`1` = additive
  sum-of-sigmoids; `>=2` = multiplicative oblivious soft trees), `task`
  (`"binary"` / `"multiclass"` / `"regression"`), and the `beta` anneal schedule.
- **`SoftTreeEnsemble` (torch) / `forward` (jax)** -- bit-identical soft-tree forward;
  plugin `forward` on `(..., d)` after `.to(device, dtype)`.
- **`ArrangementClassifier` / `ArrangementBoosted` (torch)** -- hyperplane
  arrangement layer and autograd-preserving boosted ensemble; JAX
  `arrangement_forward` / `boosted_forward`.
- **`fit_second_order` / `fit_first_order` / `fit_boosted` / `fit_joint`** -- the
  exact-curvature joint trainer, an Adam baseline, the stagewise Newton-boosting
  (GBM-mirror) driver, and a plugin `encoder ∘ head` trainer. Optional `encoder=`
  on `fit_first_order` / `fit_second_order` / `fit_arrangement` (G3 `fit_*`
  signatures stay tabular). Stagewise boosted trainers reject `encoder`.
- **`as_head(z, kind)`** -- returns a `TabHead` wrapping SoftTree / Arrangement /
  Boosted on `z.device` / `z.dtype`. Keras 3 twins live in `omnibias.tab.keras`
  (`[keras]` extra), including `ArrangementBoosted`; `learnable_beta` on Boosted
  is member-`beta` (ensemble `learning_rate` / `base` stay frozen). Optional
  Equinox wrappers: `omnibias.tab.jax.equinox_head` (`[equinox]` extra;
  `ArrangementHead` / `SoftTreeHead` / `BoostedHead`). Equinox tests **fail**
  when `CI` is set and the extra is missing; local runs still `importorskip`.
- **`certify_tab(params, feature_box, ...)`** / **`certify_composed`** -- a
  `TabCertificate` (`output_bounds`, `lipschitz`, `monotone_ok`, `rounding_gap`,
  `.is_sound`); composed IBP of a supported Linear/activation encoder then the
  tab head on the latent box; SoftTree / Arrangement encoders use interval
  output bounds (`tab+tab` / `ibp+tab+tab`); otherwise `sampled_latent` (not a
  sound enclosure of `E(box)`).
- **`extract_tree_jet`** -- depth-1 `mlp_jet_mv` (multivariate) / `mlp_jet`
  (directional); depth `>= 2` Leibniz product of sigmoid jets of the **soft**
  surrogate (`delta -> 0`); hardening remains `beta -> inf`.

## Honest scope

- Certificates are **genuine, sound** enclosures (outward-rounded intervals / Taylor
  models), verified in the test suite against a dense grid **and** a random sample. They
  are **not** exact-optimality claims and the rounding gap is never asserted zero.
- The additive (`depth=1`) model certifies via `omnibias-verify` ingestion (tighter /
  sealed); the multiplicative (`depth>=2`) model certifies via a bespoke interval /
  Taylor-model forward. Without the `verify` extra the additive path degrades to the
  (still sound) interval bound.
- Beating LightGBM is a *benchmarked* claim, earned per dataset (see
  `docs/benchmarks.md`), never asserted. Where a dataset stays behind after refinement it
  is reported and the claim scoped -- soundness tests are never loosened to win.
- Extension-tier typing (authored strict-clean; not on the shared strict CI gate).

## Tests

```bash
pip install -e "packages/omnibias-tab[torch,jax,verify,test]"
python -m pytest packages/omnibias-tab/tests -q
```

## License

Dual-licensed: AGPL-3.0-or-later OR a commercial licence from Derivon
(`LicenseRef-omnibias-Commercial`). See [`LICENSE`](LICENSE),
[`../../LICENSING.md`](../../LICENSING.md), and
[`../../COMMERCIAL-LICENSE.md`](../../COMMERCIAL-LICENSE.md). Contact
info@derivon.ai for commercial terms.
