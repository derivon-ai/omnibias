# omnibias-fields

## Write the physics in field operations.

**Named coordinates, components and reusable differential operators for models built on omnibias.**

[API reference](https://omnibias.ai/api/fields/) · [Source](https://github.com/derivon-ai/omnibias/tree/main/packages/omnibias-fields/src/omnibias/fields) · [Tests](https://github.com/derivon-ai/omnibias/tree/main/packages/omnibias-fields/tests) · [Talk to Derivon](mailto:info@derivon.ai)

## Why this package exists

A PDE residual is easier to maintain when coordinates, field components and derivative caches have explicit identities. Fields supplies that shared vocabulary so a gradient, divergence, curl or Laplacian can consume the same evaluated state. Consumers can build model families and solvers without each inventing its own field protocol.

## What you can build

- FieldState, CoordinateSpec and ComponentSpec describe evaluated fields.
- SigmaCache reuses activation derivatives within an evaluation.
- Torch and JAX operators cover scalar, vector, tensor, complex and weak-form compositions.

Use fields as the integration seam between a model and physical operators. The external PINN package owns concrete solver workflows; fields owns the reusable state and operator contracts. A component name is part of the model interface, not a guess about a tensor axis.

## Install the source edition

This README describes the current source tree. Published artifacts can lag this
branch, and this migration does not overwrite existing package versions. Build
the coordinated local wheelhouse using the [release guide](https://github.com/derivon-ai/omnibias/blob/main/RELEASE.md),
then install only this package and its selected dependencies:

```bash
python -m pip install --constraint artifacts/wheel-validation/constraints.txt "omnibias-fields"
```

Run that command from the repository containing the generated wheelhouse.
The constraint file selects the built distributions rather than silently mixing
an older published dependency with the current source. Supported Python versions,
optional features and runtime dependencies are declared in [pyproject.toml](https://github.com/derivon-ai/omnibias/blob/main/packages/omnibias-fields/pyproject.toml).

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

## Evidence, not a universal speed claim

The [performance guide](https://github.com/derivon-ai/omnibias/blob/main/docs/performance.md) separates activation derivatives,
specialized contractions and general deep-network jets. Its artifacts record
workloads, precision, compilation and independent accuracy checks, including
baseline wins. The reported speedups do not automatically transfer to an entire
training loop, another activation, a different device or this package’s every API.

## Explore and validate

The [API guide](https://github.com/derivon-ai/omnibias/blob/main/docs/api/fields.md) contains the generated module/export
inventory. Use it to find the focused implementation rather than guessing a
symbol from another package. The [capability map](https://github.com/derivon-ai/omnibias/blob/main/docs/capabilities.md)
connects the primitives to larger scientific workflows.

From the main repository, run the package’s regression suite:

```bash
uv run pytest packages/omnibias-fields/tests -q
```

Examples above are executable smoke checks, not a substitute for application
validation. For a production integration, measure the intended objective,
precision, parameter gradients, memory and wall time on representative inputs.

## License

Apache-2.0. See [LICENSE](https://github.com/derivon-ai/omnibias/blob/main/packages/omnibias-fields/LICENSE) and the [licensing policy](https://github.com/derivon-ai/omnibias/blob/main/LICENSING.md).
