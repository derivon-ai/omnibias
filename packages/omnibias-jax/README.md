# omnibias-jax

**Trace once. Differentiate a batch.** Functional directional jets compose with jit and vmap.

![Trace once. Differentiate a batch.](https://raw.githubusercontent.com/derivon-ai/omnibias/62200950132cb8fc53cc627f4614b5380058be82/packages/omnibias-jax/docs/visuals/story.gif)

[Static poster](https://raw.githubusercontent.com/derivon-ai/omnibias/62200950132cb8fc53cc627f4614b5380058be82/packages/omnibias-jax/docs/visuals/poster.png) · [Narrow-screen animation](https://raw.githubusercontent.com/derivon-ai/omnibias/62200950132cb8fc53cc627f4614b5380058be82/packages/omnibias-jax/docs/visuals/story-mobile.gif) · [How this visual is computed](https://raw.githubusercontent.com/derivon-ai/omnibias/62200950132cb8fc53cc627f4614b5380058be82/packages/omnibias-jax/docs/visuals/scene.py)

A functionally represented network, coordinates and directions enter; traced derivative operators leave. Batch with vmap, compile with jit and keep coordinates and weights as runtime arguments. The traced program can be reused across data of the same abstract shape.

The animation uses computed outputs to explain this package. Frame transitions
are illustrative unless a training step is explicitly identified; it is not a
performance comparison.


[API reference](https://github.com/derivon-ai/omnibias/blob/main/docs/api/jax.md) · [Source](https://github.com/derivon-ai/omnibias/tree/main/packages/omnibias-jax/src/omnibias/jax) · [Tests](https://github.com/derivon-ai/omnibias/tree/main/packages/omnibias-jax/tests) · [Talk to Derivon](mailto:info@derivon.ai)

## The mathematical connection

The bias-collapse limit is evaluated by shared polynomial kernels and forward Taylor composition. This bypasses nested high-order spatial reverse-mode graphs on supported paths. JAX also supplies tensor operations to temperature-based consumers, but directional jets are not an annealing procedure. Compilation, steady-state evaluation and numerical error are separate measurements.

## Run this README

The examples use `omnibias-jax` on Python >=3.10. Their installed-wheel
profile selects runtime features, not an editable workspace. Install the prepared
prerelease from PyPI:

```bash
python -m pip install --pre "omnibias-jax==0.5.0rc1"
```

For local development before publication, build and test the coordinated wheelhouse
using the [release guide](https://github.com/derivon-ai/omnibias/blob/main/RELEASE.md).
The package's [wheel profile](https://github.com/derivon-ai/omnibias/blob/main/packages/omnibias-jax/wheel-tests.toml)
executes the examples below outside the source checkout.

Existing published consumers may need historical primitive versions; see the
[compatibility policy](https://github.com/derivon-ai/omnibias/blob/main/RELEASE.md#published-consumer-compatibility).

## Why this package exists

High-order spatial derivatives and parameter gradients are different jobs. JAX omnibias carries the former through an explicit derivative representation, while jax.grad handles learning. This separation makes the computation inspectable and lets you select a contraction rather than materialize every entry of a high-order derivative tensor.

## What you can build

- Functional directional and multivariate jets with the same coefficient convention as Torch.
- Direct one-layer, repeated and deep Laplacian paths for supported structures.
- Runtime parameter flow through jit, vmap and gradient transformations.

Build a scalar residual loss whose weights are ordinary function arguments. Batch the operator over points, compile the intended workload, and differentiate with respect to parameters. For small workloads a native JAX transform may win; the benchmark guide includes those cases.

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

## Explore and validate

The [API guide](https://github.com/derivon-ai/omnibias/blob/main/docs/api/jax.md) contains the generated module/export
inventory. Use it to find the focused implementation rather than guessing a
symbol from another package. The [capability map](https://github.com/derivon-ai/omnibias/blob/main/docs/capabilities.md)
connects the primitives to larger scientific workflows.

From the main repository, run the package’s regression suite:

```bash
uv run pytest packages/omnibias-jax/tests -q
```

## License

Apache-2.0. See [LICENSE](https://github.com/derivon-ai/omnibias/blob/main/packages/omnibias-jax/LICENSE) and the [licensing policy](https://github.com/derivon-ai/omnibias/blob/main/LICENSING.md).
