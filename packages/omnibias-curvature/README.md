# omnibias-curvature

**Let curvature shape the optimization step.** A loss landscape, its local quadratic model and a damped update.

![Let curvature shape the optimization step.](https://raw.githubusercontent.com/derivon-ai/omnibias/62200950132cb8fc53cc627f4614b5380058be82/packages/omnibias-curvature/docs/visuals/story.gif)

[Static poster](https://raw.githubusercontent.com/derivon-ai/omnibias/62200950132cb8fc53cc627f4614b5380058be82/packages/omnibias-curvature/docs/visuals/poster.png) · [Narrow-screen animation](https://raw.githubusercontent.com/derivon-ai/omnibias/62200950132cb8fc53cc627f4614b5380058be82/packages/omnibias-curvature/docs/visuals/story-mobile.gif) · [How this visual is computed](https://raw.githubusercontent.com/derivon-ai/omnibias/62200950132cb8fc53cc627f4614b5380058be82/packages/omnibias-curvature/docs/visuals/scene.py)

Network parameters and a stated curvature quantity enter; a Hessian, Gauss–Newton object or structured factor leaves. Optimizer builders use these operators to choose a parameter-space step without confusing an approximation with an exact loss Hessian.

The animation uses computed outputs to explain this package. Frame transitions
are illustrative unless a training step is explicitly identified; it is not a
performance comparison.


[API reference](https://github.com/derivon-ai/omnibias/blob/main/docs/api/curvature.md) · [Source](https://github.com/derivon-ai/omnibias/tree/main/packages/omnibias-curvature/src/omnibias/curvature) · [Tests](https://github.com/derivon-ai/omnibias/tree/main/packages/omnibias-curvature/tests) · [Talk to Derivon](mailto:info@derivon.ai)

## The mathematical connection

Bias-collapse activation derivatives provide analytic ingredients for parameter curvature. A signed Hessian describes local behavior and can be indefinite. Temperature collapse may appear in a consuming soft-decision model, but is not the definition of Hessian, Fisher or KFAC. Damping and step acceptance remain optimizer responsibilities.

## Run this README

The examples use `omnibias-curvature` on Python >=3.10. Their installed-wheel
profile selects runtime features, not an editable workspace. Install the prepared
prerelease from PyPI:

```bash
python -m pip install --pre "omnibias-curvature==0.1.0a2"
```

For local development before publication, build and test the coordinated wheelhouse
using the [release guide](https://github.com/derivon-ai/omnibias/blob/main/RELEASE.md).
The package's [wheel profile](https://github.com/derivon-ai/omnibias/blob/main/packages/omnibias-curvature/wheel-tests.toml)
executes the examples below outside the source checkout.

Existing published consumers may need historical primitive versions; see the
[compatibility policy](https://github.com/derivon-ai/omnibias/blob/main/RELEASE.md#published-consumer-compatibility).

## Why this package exists

A gradient gives a local direction; curvature describes how that direction changes with parameters. For supported model structures, analytic parameter derivatives avoid rebuilding a generic parameter-Hessian computation. Other paths trade exactness or density for structured factors and operator products.

## What you can build

- One-layer parameter gradients and Hessians.
- MSE Hessian, Gauss–Newton/Fisher and KFAC constructions.
- Damped solves, matrix-free curvature and sharpness-oriented utilities.

Use curvature when a training method, uncertainty calculation or neural-VMC optimizer needs more than first-order information. Start from the objective and parameterization that the formula supports. Dense Hessians are useful small-model references; operator and factorized forms address different memory budgets.

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

## License

AGPL-3.0-or-later **or commercial**. See [LICENSE](https://github.com/derivon-ai/omnibias/blob/main/packages/omnibias-curvature/LICENSE) and the [licensing policy](https://github.com/derivon-ai/omnibias/blob/main/LICENSING.md).
Comply with the AGPL terms or obtain a signed commercial grant; commercial use alone does not require payment. [Commercial terms](https://github.com/derivon-ai/omnibias/blob/main/packages/omnibias-curvature/COMMERCIAL-LICENSE.md) describe the alternative. Previously distributed Apache editions, where applicable, retain their original grants.
