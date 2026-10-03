# omnibias-jax

## Derivative towers that compose with JAX.

**Functional activation derivatives and neural jets built to work with compilation, batching and parameter differentiation.**

[API reference](https://omnibias.ai/api/jax/) · [Source](https://github.com/derivon-ai/omnibias/tree/main/packages/omnibias-jax/src/omnibias/jax) · [Tests](https://github.com/derivon-ai/omnibias/tree/main/packages/omnibias-jax/tests) · [Talk to Derivon](mailto:info@derivon.ai)

![Derivative towers that compose with JAX.](https://raw.githubusercontent.com/derivon-ai/omnibias/62c9646964ea58b00cc3de5b80bcc2134b45f511/docs/img/explain/bias-collapse.svg)

<details>
<summary>Watch the mechanism change continuously</summary>

![Animated mathematical explanation](https://raw.githubusercontent.com/derivon-ai/omnibias/62c9646964ea58b00cc3de5b80bcc2134b45f511/docs/img/explain/bias-collapse.gif)

The animation illustrates the mechanism; it is not a speed benchmark.
</details>

## Why this package exists

High-order spatial derivatives and parameter gradients are different jobs. JAX omnibias carries the former through an explicit derivative representation, while jax.grad handles learning. This separation makes the computation inspectable and lets you select a contraction rather than materialize every entry of a high-order derivative tensor.

## What you can build

- Functional directional and multivariate jets with the same coefficient convention as Torch.
- Direct one-layer, repeated and deep Laplacian paths for supported structures.
- Runtime parameter flow through jit, vmap and gradient transformations.

Build a scalar residual loss whose weights are ordinary function arguments. Batch the operator over points, compile the intended workload, and differentiate with respect to parameters. For small workloads a native JAX transform may win; the benchmark guide includes those cases.

## Install the source edition

This README describes the current source tree. Published artifacts can lag this
branch, and this migration does not overwrite existing package versions. Build
the coordinated local wheelhouse using the [release guide](https://github.com/derivon-ai/omnibias/blob/main/RELEASE.md),
then install only this package and its selected dependencies:

```bash
python -m pip install --constraint artifacts/wheel-validation/constraints.txt "omnibias-jax"
```

Run that command from the repository containing the generated wheelhouse.
The constraint file selects the built distributions rather than silently mixing
an older published dependency with the current source. Supported Python versions,
optional features and runtime dependencies are declared in [pyproject.toml](https://github.com/derivon-ai/omnibias/blob/main/packages/omnibias-jax/pyproject.toml).

## A working example

```python
import jax
import jax.numpy as jnp
from omnibias.jax.jet import mlp_jet, jet_to_tower

def loss(w):
    layers = [(w, jnp.array([0.1, 0.2]), "tanh"),
              (jnp.array([[0.6, -0.4]]), None, None)]
    tower = jet_to_tower(mlp_jet(jnp.array([0.3]), jnp.ones(1), layers, order=4))
    return jnp.mean((tower[4] + tower[0]) ** 2)

value, gradient = jax.jit(jax.value_and_grad(loss))(jnp.array([[0.5], [-0.3]]))
assert gradient.shape == (2, 1)
assert bool(jnp.all(jnp.isfinite(gradient)))
```

## Choose the right contract

Do not close over benchmark weights or inputs: constant folding can remove the work you intend to time. Synchronize results, separate compilation from execution, and enable x64 before constructing float64 arrays. Full mixed jets enumerate coefficients; the directional and direct-contraction paths have different scaling.

## Evidence, not a universal speed claim

The [performance guide](https://github.com/derivon-ai/omnibias/blob/main/docs/performance.md) separates activation derivatives,
specialized contractions and general deep-network jets. Its artifacts record
workloads, precision, compilation and independent accuracy checks, including
baseline wins. The reported speedups do not automatically transfer to an entire
training loop, another activation, a different device or this package’s every API.

## Explore and validate

The [API guide](https://github.com/derivon-ai/omnibias/blob/main/docs/api/jax.md) contains the generated module/export
inventory. Use it to find the focused implementation rather than guessing a
symbol from another package. The [capability map](https://github.com/derivon-ai/omnibias/blob/main/docs/capabilities.md)
connects the primitives to larger scientific workflows.

From the main repository, run the package’s regression suite:

```bash
uv run pytest packages/omnibias-jax/tests -q
```

Examples above are executable smoke checks, not a substitute for application
validation. For a production integration, measure the intended objective,
precision, parameter gradients, memory and wall time on representative inputs.

## License

Apache-2.0. See [LICENSE](https://github.com/derivon-ai/omnibias/blob/main/packages/omnibias-jax/LICENSE) and the [licensing policy](https://github.com/derivon-ai/omnibias/blob/main/LICENSING.md).
