---
name: omnibias-curvature
description: Maintain parameter Hessian formulas, Fisher factors and matrix-free curvature approximations.
---

# Maintaining omnibias-curvature

Owned implementation: [source](../../../packages/omnibias-curvature/src/omnibias/curvature);
public surface: [API](../../../docs/api/curvature.md).
Load [shared numerical contracts](../../../docs/development/numerical-contracts.md)
when changing mathematics or tensor behavior.

`one_layer.py` owns supported one-layer parameter formulas. `operators.py`,
`information/` and `torch/` provide operator forms and training integrations.
Keep parameter packing order explicit: a correct matrix with the wrong parameter
ordering gives incorrect updates while retaining symmetry.

Identify the mathematical object before changing an implementation: exact loss
Hessian, residual Gauss–Newton, Fisher information, diagonal approximation and
Kronecker factors are not interchangeable. Verify reduction conventions over
examples and outputs, damping placement, and whether a solve applies an inverse
or a preconditioner. For matrix-free paths, compare operator-vector products
against a small explicit matrix without adding dense materialization to the
production implementation.

Sharpness and regularization changes must preserve the distinction between an
exact construction and a stochastic estimate. Test deterministic seeds and
estimator metadata where randomness is exposed. `test_one_layer.py` checks formula
indexing; matrix-free tests exercise operators and solves; Torch integration tests
cover differentiation through the training objective. Optimizer subclasses owned
by the Torch distribution remain there; coordinate interface changes rather than
copying their implementation into this package.

From the omnibias repository, start with:

```bash
uv run pytest packages/omnibias-curvature/tests/test_one_layer.py packages/omnibias-curvature/tests/test_matrixfree_operators.py packages/omnibias-curvature/tests/test_torch_sharpness_integration.py -q
```
