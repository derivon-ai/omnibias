---
name: omnibias-qcalculus
description: Maintain exact q-number algebra, Jackson operators and q-deformed series.
---

# Maintaining omnibias-qcalculus

Owned implementation: [source](../../../packages/omnibias-qcalculus/src/omnibias/qcalculus);
public surface: [API](../../../docs/api/qcalculus.md).
Load [shared numerical contracts](../../../docs/development/numerical-contracts.md)
when changing mathematics or tensor behavior.

`_core/` owns q-numbers, products, series and Jackson definitions; `umbral.py`
owns q-deformed sequence operations. Tensor implementations live in `torch/` and
`jax/`. Ordinary finite-difference and recurrence definitions remain in the
`difference` distribution, which this package consumes.

A q-operator's behavior depends on its base point and deformation parameter.
Handle the `q → 1` limit through the documented limiting expression, including
zero coordinates and removable singularities. Do not replace exact integer or
rational special cases with a floating approximation. For infinite series,
convergence domains and certified tail conditions are part of the return value's
meaning; an exhausted budget is not convergence.

Check the same operator in exact algebra, numerical evaluation and its classical
limit. Hybrid tests cover compositions with the ordinary register. Tensor changes
need gradients in both inputs and differentiable order or deformation parameters
where the API promises them. The package tests cover algebraic boundary cases;
`test_qderiv.py` and `test_hybrid.py` exercise operator behavior. Keep unrelated
quantum-physics application residuals in consumers.

From the omnibias repository, start with:

```bash
uv run pytest packages/omnibias-qcalculus/tests/test_qnumbers.py packages/omnibias-qcalculus/tests/test_qderiv.py packages/omnibias-qcalculus/tests/test_hybrid.py -q
```
