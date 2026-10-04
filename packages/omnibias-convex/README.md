# omnibias-convex

**Optimize inside explicit constraints.** A barrier path approaches the constrained quadratic optimum.

![Optimize inside explicit constraints.](https://raw.githubusercontent.com/derivon-ai/omnibias/62200950132cb8fc53cc627f4614b5380058be82/packages/omnibias-convex/docs/visuals/story.gif)

[Static poster](https://raw.githubusercontent.com/derivon-ai/omnibias/62200950132cb8fc53cc627f4614b5380058be82/packages/omnibias-convex/docs/visuals/poster.png) · [Narrow-screen animation](https://raw.githubusercontent.com/derivon-ai/omnibias/62200950132cb8fc53cc627f4614b5380058be82/packages/omnibias-convex/docs/visuals/story-mobile.gif) · [How this visual is computed](https://raw.githubusercontent.com/derivon-ai/omnibias/62200950132cb8fc53cc627f4614b5380058be82/packages/omnibias-convex/docs/visuals/scene.py)

Linear or quadratic objectives and explicit constraints enter; primal candidates, dual information, implicit gradients and checked bounds leave. This is an optimization engine and certificate backend, not a generic neural-network layer factory.

The animation uses computed outputs to explain this package. Frame transitions
are illustrative unless a training step is explicitly identified; it is not a
performance comparison.


[API reference](https://omnibias.ai/api/convex/) · [Source](https://github.com/derivon-ai/omnibias/tree/main/packages/omnibias-convex/src/omnibias/convex) · [Tests](https://github.com/derivon-ai/omnibias/tree/main/packages/omnibias-convex/tests) · [Talk to Derivon](mailto:info@derivon.ai)

## The mathematical connection

The log-barrier continuation follows an interior path toward a constrained solution. It shares the idea of progressively removing a smooth approximation with temperature collapse, but its parameter and feasibility semantics are those of the barrier problem. Bias collapse is not needed for its closed-form quadratic/barrier derivatives. Finite termination tolerances must be checked.

## Run this README

The examples use `omnibias-convex[torch]` on Python >=3.10. Their installed-wheel
profile selects runtime features, not an editable workspace. Install the prepared
prerelease from PyPI:

```bash
python -m pip install --pre "omnibias-convex[torch]==0.1.0a2"
```

For local development before publication, build and test the coordinated wheelhouse
using the [release guide](https://github.com/derivon-ai/omnibias/blob/main/RELEASE.md).
The package's [wheel profile](https://github.com/derivon-ai/omnibias/blob/main/packages/omnibias-convex/wheel-tests.toml)
executes the examples below outside the source checkout.

Existing published consumers may need historical primitive versions; see the
[compatibility policy](https://github.com/derivon-ai/omnibias/blob/main/RELEASE.md#published-consumer-compatibility).

## Why this package exists

Many scientific and decision workflows need a constrained solve inside a larger model. Convex separates problem data, a numerical candidate and its optimality checks. This makes it possible to inspect feasibility and use a differentiable layer where the chosen solve and its regularity assumptions support it.

## What you can build

- Linear and quadratic objectives with explicit inequality constraints.
- Log-barrier solver configuration and implicit differentiation layers.
- Dual lower bounds, optimality enclosures and reusable warm starts.

Use convex as a continuous optimization primitive or a bound backend for discrete applications. Start with a small independently solvable problem, check residuals and slacks, then integrate the layer into the surrounding training objective. Named application frontends belong outside this package.

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

## License

AGPL-3.0-or-later **or commercial**. See [LICENSE](https://github.com/derivon-ai/omnibias/blob/main/packages/omnibias-convex/LICENSE) and the [licensing policy](https://github.com/derivon-ai/omnibias/blob/main/LICENSING.md).
Comply with the AGPL terms or obtain a signed commercial grant; commercial use alone does not require payment. [Commercial terms](https://github.com/derivon-ai/omnibias/blob/main/packages/omnibias-convex/COMMERCIAL-LICENSE.md) describe the alternative. Previously distributed Apache editions, where applicable, retain their original grants.
