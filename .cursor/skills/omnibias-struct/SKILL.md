---
name: omnibias-struct
description: Differentiate certified dynamic programming: soft Viterbi, CTC, CKY, Eisner, matrix-tree, with logsumexp_beta relaxation differentiated by the closed-form tower and a log(N)/beta gap certificate. Use when inventing a new semiring layer or a sealed decode.
---

# omnibias-struct

Certified differentiable dynamic programming: soft Viterbi / shortest-path / CTC layers
whose logsumexp_beta relaxation (the beta->inf temperature axis) is differentiated
exactly by the closed-form softplus/sigmoid derivative tower (the delta->0 axis) via
omnibias.{torch,jax}.jet, with a closed-form logsumexp_beta >= max gap certificate
validated against brute-force hard DP; bit-identical torch + jax twins.

## Why nested AD fails

Hard DP has no gradient. Nested AD through a logsumexp chart is cubic and still has no
gap vs brute-force hard DP. Generic structured-prediction layers do not expose exact
jets of the chart.

## What only this tower unlocks

Certified differentiable dynamic programming: soft Viterbi / shortest-path / CTC layers
whose logsumexp_beta relaxation (the beta->inf temperature axis) is differentiated
exactly by the closed-form softplus/sigmoid derivative tower (the delta->0 axis) via
omnibias.{torch,jax}.jet, with a closed-form logsumexp_beta >= max gap certificate
validated against brute-force hard DP; bit-identical torch + jax twins.

Coefficients live in `omnibias.core.polynomials` and are imported, never forked.
When torch and jax twins exist they stay bit-identical by construction. Tracked
files stay vendor-neutral.

## Use

Temperature collapse (`beta -> inf`) is the DP feasibility axis; the
derivative tower is founding `delta -> 0`.

| You want | Import |
| --- | --- |
| Soft Viterbi / CTC / CKY / Eisner | `omnibias.struct` |
| Tropical / path-follow | `omnibias.struct._core.tropical` |
| Certified decode | `omnibias.struct.decode` |

## Extend

- Source: [`packages/omnibias-struct`](../../../packages/omnibias-struct).
- Namespace: `omnibias.struct`. Inspect `__init__.py` and `docs/packages.md` before changing a public seam.
- Tests: `python -m pytest packages/omnibias-struct/tests -q`.
- Compose with `omnibias-tab`, `omnibias-partition`, `omnibias-verify` by **new names**.
- New tensors follow the framework default dtype. Add a regression test for every behavioral change. Regenerate sorted `__all__` when a public symbol moves.
- Heavy compute follows the workspace rule; artifacts go to `$OMNIBIAS_SCRATCH`.

## Next invention

A CKY chart whose `cky_lse_jet` matches nested AD at order 3 and whose log(N)/beta gap
contains the brute-force hard-decode residual.


## Further references

- Capability matrix: [`docs/operator-surface.md`](../../../docs/operator-surface.md)
- Package index: [`docs/packages.md`](../../../docs/packages.md)
- Repository map: [`AGENTS.md`](../../../AGENTS.md)
