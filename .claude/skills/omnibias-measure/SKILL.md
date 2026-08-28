---
name: omnibias-measure
description: Integrate against autograd-native Measures: pushforward, product, importance reweighting, layer-cake, and simple-function primitives with trainable torch/jax layers. Use when inventing a new measure layer or a sliced transport identity on activation mixtures.
---

# omnibias-measure

Autograd-native measure-theoretic integration for omnibias: a Measure abstraction
(discrete measure = nodes + weights, with pushforward / product / importance
reweighting) and the measure integral int f dmu, plus the layer-cake /
distribution-function formula, self-normalized importance-sampling expectations, and
simple-function (from-below) approximation -- each with bit-identical torch + jax twins
and trainable nn.Module / functional layers.

## Why nested AD fails

Monte-Carlo expectations through nested AD are noisy and have no layer-cake identity.
Generic integration libraries do not expose a Measure object whose pushforward rides the
closed-form tower.

## What only this tower unlocks

Autograd-native measure-theoretic integration for omnibias: a Measure abstraction
(discrete measure = nodes + weights, with pushforward / product / importance
reweighting) and the measure integral int f dmu, plus the layer-cake /
distribution-function formula, self-normalized importance-sampling expectations, and
simple-function (from-below) approximation -- each with bit-identical torch + jax twins
and trainable nn.Module / functional layers.

Coefficients live in `omnibias.core.polynomials` and are imported, never forked.
When torch and jax twins exist they stay bit-identical by construction. Tracked
files stay vendor-neutral.

## Use

This is sense (3) of "integral" on the operator surface: `int f dmu`,
numerical. Sense (1) is OperatorBlock `integral`; sense (2) is field
quadrature.

| You want | Import |
| --- | --- |
| Measure / pushforward / product | `omnibias.measure` |
| Sliced W1 of activation mixtures | `omnibias.measure.transport` |

## Extend

- Source: [`packages/omnibias-measure`](../../../packages/omnibias-measure).
- Namespace: `omnibias.measure`. Inspect `__init__.py` and `docs/packages.md` before changing a public seam.
- Tests: `python -m pytest packages/omnibias-measure/tests -q`.
- Compose with `omnibias-fields`, `omnibias-score`, `omnibias-verify` by **new names**.
- New tensors follow the framework default dtype. Add a regression test for every behavioral change. Regenerate sorted `__all__` when a public symbol moves.
- Heavy compute follows the workspace rule; artifacts go to `$OMNIBIAS_SCRATCH`.

## Next invention

A trainable product measure whose layer-cake identity is exact on the activation mixture
and whose torch/jax twins match at float64.


## Further references

- Capability matrix: [`docs/operator-surface.md`](../../../docs/operator-surface.md)
- Package index: [`docs/packages.md`](../../../docs/packages.md)
- Repository map: [`AGENTS.md`](../../../AGENTS.md)
