---
name: omnibias-spiking
description: Train LIF / IF spiking neurons with closed-form surrogate gradients from the activation dictionary. Use when inventing a new surrogate, a certified spike Lipschitz, or a jet through a reset.
---

# omnibias-spiking

Spiking-neuron LIF/IF primitives with exact omnibias closed-form surrogate gradients
(torch + jax).

## Why nested AD fails

Spikes are steps; nested AD is zero. Hand-chosen surrogate shapes (SuperSpike, arctan)
are first-order guesses with no n-th derivative and no shared polynomial source.

## What only this tower unlocks

Spiking-neuron LIF/IF primitives with exact omnibias closed-form surrogate gradients
(torch + jax).

Coefficients live in `omnibias.core.polynomials` and are imported, never forked.
When torch and jax twins exist they stay bit-identical by construction. Tracked
files stay vendor-neutral.

## Use

Surrogates are the Riccati tower, not a clipped identity.

| You want | Import |
| --- | --- |
| LIF / IF + surrogate | `omnibias.spiking` |

## Extend

- Source: [`packages/omnibias-spiking`](../../../packages/omnibias-spiking).
- Namespace: `omnibias.spiking`. Inspect `__init__.py` and `docs/packages.md` before changing a public seam.
- Tests: `python -m pytest packages/omnibias-spiking/tests -q`.
- Compose with `omnibias-torch`, `omnibias-jax`, `omnibias-binary` by **new names**.
- New tensors follow the framework default dtype. Add a regression test for every behavioral change. Regenerate sorted `__all__` when a public symbol moves.
- Heavy compute follows the workspace rule; artifacts go to `$OMNIBIAS_SCRATCH`.

## Next invention

A LIF whose closed-form surrogate Hessian trains with CubicNewton and whose spike-count
Lipschitz is a sealed Interval.


## Further references

- Capability matrix: [`docs/operator-surface.md`](../../../docs/operator-surface.md)
- Package index: [`docs/packages.md`](../../../docs/packages.md)
- Repository map: [`AGENTS.md`](../../../AGENTS.md)
