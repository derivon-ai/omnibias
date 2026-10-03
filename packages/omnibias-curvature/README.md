# omnibias-curvature

## See how the loss bends.

**Parameter Hessians, Fisher constructions, matrix-free operators and structured curvature approximations.**

[API reference](https://omnibias.ai/api/curvature/) · [Source](https://github.com/derivon-ai/omnibias/tree/main/packages/omnibias-curvature/src/omnibias/curvature) · [Tests](https://github.com/derivon-ai/omnibias/tree/main/packages/omnibias-curvature/tests) · [Talk to Derivon](mailto:info@derivon.ai)

## Why this package exists

A gradient gives a local direction; curvature describes how that direction changes with parameters. For supported model structures, analytic parameter derivatives avoid rebuilding a generic parameter-Hessian computation. Other paths trade exactness or density for structured factors and operator products.

## What you can build

- One-layer parameter gradients and Hessians.
- MSE Hessian, Gauss–Newton/Fisher and KFAC constructions.
- Damped solves, matrix-free curvature and sharpness-oriented utilities.

Use curvature when a training method, uncertainty calculation or neural-VMC optimizer needs more than first-order information. Start from the objective and parameterization that the formula supports. Dense Hessians are useful small-model references; operator and factorized forms address different memory budgets.

## Install the source edition

This README describes the current source tree. Published artifacts can lag this
branch, and this migration does not overwrite existing package versions. Build
the coordinated local wheelhouse using the [release guide](https://github.com/derivon-ai/omnibias/blob/main/RELEASE.md),
then install only this package and its selected dependencies:

```bash
python -m pip install --constraint artifacts/wheel-validation/constraints.txt "omnibias-curvature"
```

Run that command from the repository containing the generated wheelhouse.
The constraint file selects the built distributions rather than silently mixing
an older published dependency with the current source. Supported Python versions,
optional features and runtime dependencies are declared in [pyproject.toml](https://github.com/derivon-ai/omnibias/blob/main/packages/omnibias-curvature/pyproject.toml).

## A working example

```python
import jax.numpy as jnp
from omnibias.curvature import one_layer_param_hessian

x = jnp.array([0.2, -0.1])
W = jnp.array([[0.4, 0.2], [-0.3, 0.5]])
H = one_layer_param_hessian(x, W, jnp.zeros(2), jnp.ones(2), jnp.array(0.), "tanh")
assert H.shape == (9, 9)  # scalar bias + readout + hidden bias + weights
assert bool(jnp.allclose(H, H.T))
```

## Choose the right contract

An exact Hessian, a Gauss–Newton matrix, a Fisher matrix and KFAC are not synonyms. Damping changes the solve and its interpretation. One-layer formulas do not automatically cover arbitrary deep modules; validate both the mathematical structure and parameter layout before using a second-order step.

## Explore and validate

The [API guide](https://github.com/derivon-ai/omnibias/blob/main/docs/api/curvature.md) contains the generated module/export
inventory. Use it to find the focused implementation rather than guessing a
symbol from another package. The [capability map](https://github.com/derivon-ai/omnibias/blob/main/docs/capabilities.md)
connects the primitives to larger scientific workflows.

From the main repository, run the package’s regression suite:

```bash
uv run pytest packages/omnibias-curvature/tests -q
```

Examples above are executable smoke checks, not a substitute for application
validation. For a production integration, measure the intended objective,
precision, parameter gradients, memory and wall time on representative inputs.

## License

AGPL-3.0-or-later **or commercial**. See [LICENSE](https://github.com/derivon-ai/omnibias/blob/main/packages/omnibias-curvature/LICENSE) and the [licensing policy](https://github.com/derivon-ai/omnibias/blob/main/LICENSING.md).
Comply with the AGPL terms or obtain a signed commercial grant; commercial use alone does not require payment. [Commercial terms](https://github.com/derivon-ai/omnibias/blob/main/packages/omnibias-curvature/COMMERCIAL-LICENSE.md) describe the alternative. Previously distributed Apache editions, where applicable, retain their original grants.
