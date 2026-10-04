# omnibias-fields

**One field. Several operators.** Coordinate-aware state shares derivative work across views.

![One field. Several operators.](https://raw.githubusercontent.com/derivon-ai/omnibias/62200950132cb8fc53cc627f4614b5380058be82/packages/omnibias-fields/docs/visuals/story.gif)

[Static poster](https://raw.githubusercontent.com/derivon-ai/omnibias/62200950132cb8fc53cc627f4614b5380058be82/packages/omnibias-fields/docs/visuals/poster.png) · [Narrow-screen animation](https://raw.githubusercontent.com/derivon-ai/omnibias/62200950132cb8fc53cc627f4614b5380058be82/packages/omnibias-fields/docs/visuals/story-mobile.gif) · [How this visual is computed](https://raw.githubusercontent.com/derivon-ai/omnibias/62200950132cb8fc53cc627f4614b5380058be82/packages/omnibias-fields/docs/visuals/scene.py)

Coordinates, named components and derivative providers enter a FieldState. Gradient, divergence, Hessian and Laplacian views leave through one dispatch and caching contract. This is infrastructure for PDE and geometry libraries, not a PDE solver.

The animation uses computed outputs to explain this package. Frame transitions
are illustrative unless a training step is explicitly identified; it is not a
performance comparison.


[API reference](https://github.com/derivon-ai/omnibias/blob/main/docs/api/fields.md) · [Source](https://github.com/derivon-ai/omnibias/tree/main/packages/omnibias-fields/src/omnibias/fields) · [Tests](https://github.com/derivon-ai/omnibias/tree/main/packages/omnibias-fields/tests) · [Talk to Derivon](mailto:info@derivon.ai)

## The mathematical connection

Bias-collapse backends can provide the cached derivative tower; field operators contract and compose those derivatives without requiring concrete consumer classes. Other providers must declare their own derivative semantics. Temperature collapse is not built into a FieldState: regional gates are an explicit higher-level integration.

## Run this README

The examples use `omnibias-fields` on Python >=3.10. Their installed-wheel
profile selects runtime features, not an editable workspace. Install the prepared
prerelease from PyPI:

```bash
python -m pip install --pre "omnibias-fields==0.2.0rc1"
```

For local development before publication, build and test the coordinated wheelhouse
using the [release guide](https://github.com/derivon-ai/omnibias/blob/main/RELEASE.md).
The package's [wheel profile](https://github.com/derivon-ai/omnibias/blob/main/packages/omnibias-fields/wheel-tests.toml)
executes the examples below outside the source checkout.

Existing published consumers may need historical primitive versions; see the
[compatibility policy](https://github.com/derivon-ai/omnibias/blob/main/RELEASE.md#published-consumer-compatibility).

## Why this package exists

A PDE residual is easier to maintain when coordinates, field components and derivative caches have explicit identities. Fields supplies that shared vocabulary so a gradient, divergence, curl or Laplacian can consume the same evaluated state. Consumers can build model families and solvers without each inventing its own field protocol.

## What you can build

- FieldState, CoordinateSpec and ComponentSpec describe evaluated fields.
- SigmaCache reuses activation derivatives within an evaluation.
- Torch and JAX operators cover scalar, vector, tensor, complex and weak-form compositions.

Use fields as the integration seam between a model and physical operators. The external PINN package owns concrete solver workflows; fields owns the reusable state and operator contracts. A component name is part of the model interface, not a guess about a tensor axis.

## A working example

```python
from omnibias.fields import CoordinateSpec, ComponentSpec, SigmaCache
from omnibias.core import eval_tanh_derivative

coordinates = CoordinateSpec(axes=("t", "x"), time_axis="t")
components = ComponentSpec(names=("u",))
assert coordinates.axis_index("x") == 1 and components.is_component("u")
cache = SigmaCache(z=0.3)
u_xx = cache.get_or_compute(2, lambda n: eval_tanh_derivative(0.3, n))
assert cache.get_or_compute(2, lambda n: 999.0) == u_xx
```

## Choose the right contract

Operators consume a FieldState with a compatible provider; they are not generic functions accepting any tensor. Rebuild state after coordinates or parameters change. Cached values must stay attached to the current computation. Numerical quadrature is an approximation unless the selected rule is exact for the integrand.

## Explore and validate

The [API guide](https://github.com/derivon-ai/omnibias/blob/main/docs/api/fields.md) contains the generated module/export
inventory. Use it to find the focused implementation rather than guessing a
symbol from another package. The [capability map](https://github.com/derivon-ai/omnibias/blob/main/docs/capabilities.md)
connects the primitives to larger scientific workflows.

From the main repository, run the package’s regression suite:

```bash
uv run pytest packages/omnibias-fields/tests -q
```

## License

Apache-2.0. See [LICENSE](https://github.com/derivon-ai/omnibias/blob/main/packages/omnibias-fields/LICENSE) and the [licensing policy](https://github.com/derivon-ai/omnibias/blob/main/LICENSING.md).
