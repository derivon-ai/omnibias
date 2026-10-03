# omnibias-convex

## Optimization with an inspectable certificate path.

**Differentiable LP/QP machinery, barrier solves and verified optimality bounds.**

[API reference](https://omnibias.ai/api/convex/) · [Source](https://github.com/derivon-ai/omnibias/tree/main/packages/omnibias-convex/src/omnibias/convex) · [Tests](https://github.com/derivon-ai/omnibias/tree/main/packages/omnibias-convex/tests) · [Talk to Derivon](mailto:info@derivon.ai)

## Why this package exists

Many scientific and decision workflows need a constrained solve inside a larger model. Convex separates problem data, a numerical candidate and its optimality checks. This makes it possible to inspect feasibility and use a differentiable layer where the chosen solve and its regularity assumptions support it.

## What you can build

- Linear and quadratic objectives with explicit inequality constraints.
- Log-barrier solver configuration and implicit differentiation layers.
- Dual lower bounds, optimality enclosures and reusable warm starts.

Use convex as a continuous optimization primitive or a bound backend for discrete applications. Start with a small independently solvable problem, check residuals and slacks, then integrate the layer into the surrounding training objective. Named application frontends belong outside this package.

## Install the source edition

This README describes the current source tree. Published artifacts can lag this
branch, and this migration does not overwrite existing package versions. Build
the coordinated local wheelhouse using the [release guide](https://github.com/derivon-ai/omnibias/blob/main/RELEASE.md),
then install only this package and its selected dependencies:

```bash
python -m pip install --constraint artifacts/wheel-validation/constraints.txt "omnibias-convex[torch]"
```

Run that command from the repository containing the generated wheelhouse.
The constraint file selects the built distributions rather than silently mixing
an older published dependency with the current source. Supported Python versions,
optional features and runtime dependencies are declared in [pyproject.toml](https://github.com/derivon-ai/omnibias/blob/main/packages/omnibias-convex/pyproject.toml).

## A working example

```python
import torch
from omnibias.convex.torch import solve_qp

# Minimize 0.5*x**2 - 0.5*x over -1 <= x <= 1; optimum is x=0.5.
solution = solve_qp([[1.]], [-0.5], [[1.], [-1.]], [1., 1.], x0=[0.])
assert solution.converged
assert torch.all(solution.slack > 0)
assert abs(float(solution.x[0]) - 0.5) < 1e-3
print(solution.gap)
```

## Choose the right contract

A finite barrier iterate is not automatically an exact optimizer. Inspect converged, gap and slack rather than treating a returned vector as certified. Implicit gradients need appropriate regularity, especially when active constraints change. The solver and certificate functions expose different claims.

## Explore and validate

The [API guide](https://github.com/derivon-ai/omnibias/blob/main/docs/api/convex.md) contains the generated module/export
inventory. Use it to find the focused implementation rather than guessing a
symbol from another package. The [capability map](https://github.com/derivon-ai/omnibias/blob/main/docs/capabilities.md)
connects the primitives to larger scientific workflows.

From the main repository, run the package’s regression suite:

```bash
uv run pytest packages/omnibias-convex/tests -q
```

Examples above are executable smoke checks, not a substitute for application
validation. For a production integration, measure the intended objective,
precision, parameter gradients, memory and wall time on representative inputs.

## License

AGPL-3.0-or-later **or commercial**. See [LICENSE](https://github.com/derivon-ai/omnibias/blob/main/packages/omnibias-convex/LICENSE) and the [licensing policy](https://github.com/derivon-ai/omnibias/blob/main/LICENSING.md).
Comply with the AGPL terms or obtain a signed commercial grant; commercial use alone does not require payment. [Commercial terms](https://github.com/derivon-ai/omnibias/blob/main/packages/omnibias-convex/COMMERCIAL-LICENSE.md) describe the alternative. Previously distributed Apache editions, where applicable, retain their original grants.
