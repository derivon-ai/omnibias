---
name: omnibias-tab
description: Train certified soft decision-tree ensembles: oblique sigmoid gates, exact second-order joint training, Newton boosting, and sealed output / Lipschitz / monotonicity certificates. Use when inventing a new tabular head or a soft-to-hard rounding gap.
---

# omnibias-tab

Differentiable, exactly second-order-trained, and certified soft decision-tree ensembles
for tabular data: oblique soft-split gates annealed by temperature collapse (the beta ->
inf penalty) from soft toward hard trees, trained end to end with omnibias
exact-curvature optimizers plus a Newton-boosting driver, carrying sound certificates
(output bounds, Lipschitz, per-feature monotonicity, and a certified soft->hard rounding
gap) with bit-identical torch/jax forwards, and benchmarked against gradient boosting
(LightGBM).

## Why nested AD fails

Hard trees have no gradient; GBDT Hessians are diagonal guesses. Nested AD through a
deep product of sigmoids is unstable and has no IBP certificate. LightGBM cannot expose
a sealed min.

## What only this tower unlocks

Differentiable, exactly second-order-trained, and certified soft decision-tree ensembles
for tabular data: oblique soft-split gates annealed by temperature collapse (the beta ->
inf penalty) from soft toward hard trees, trained end to end with omnibias
exact-curvature optimizers plus a Newton-boosting driver, carrying sound certificates
(output bounds, Lipschitz, per-feature monotonicity, and a certified soft->hard rounding
gap) with bit-identical torch/jax forwards, and benchmarked against gradient boosting
(LightGBM).

Coefficients live in `omnibias.core.polynomials` and are imported, never forked.
When torch and jax twins exist they stay bit-identical by construction. Tracked
files stay vendor-neutral.

## Use

Temperature collapse (`beta -> inf`) anneals soft splits toward hard trees.

| You want | Import |
| --- | --- |
| Soft trees / arrangement boost | `omnibias.tab` |
| Axis POU booster / joint TabPOU (05-04 / 05-05) | `omnibias.tab.pou` (`fit_tabpou`, `fit_tabpou_joint`) |
| Certify composed encoder+head | `omnibias.tab.certify_composed` |
| Keras tab layers | `omnibias.tab.keras` |

## Extend

- Source: [`packages/omnibias-tab`](../../../packages/omnibias-tab).
- Namespace: `omnibias.tab`. Inspect `__init__.py` and `docs/packages.md` before changing a public seam.
- Tests: `python -m pytest packages/omnibias-tab/tests -q`.
- Compose with `omnibias-partition`, `omnibias-curvature`, `omnibias-verify`, `omnibias-struct` by **new names**.
- New tensors follow the framework default dtype. Add a regression test for every behavioral change. Regenerate sorted `__all__` when a public symbol moves.
- Heavy compute follows the workspace rule; artifacts go to `$OMNIBIAS_SCRATCH`.

## Next invention

A depth-2 arrangement whose Newton-boost path beats LightGBM on a named public table and
whose sealed min matches `certified_minimize`.


## Further references

- Capability matrix: [`docs/operator-surface.md`](../../../docs/operator-surface.md)
- Package index: [`docs/packages.md`](../../../docs/packages.md)
- Repository map: [`AGENTS.md`](../../../AGENTS.md)
