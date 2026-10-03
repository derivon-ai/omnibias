# omnibias-core

## One algebra. Every backend.

**The framework-independent mathematics behind high-order neural derivatives and checked numerical enclosures.**

[API reference](https://omnibias.ai/api/core/) · [Source](https://github.com/derivon-ai/omnibias/tree/main/packages/omnibias-core/src/omnibias/core) · [Tests](https://github.com/derivon-ai/omnibias/tree/main/packages/omnibias-core/tests) · [Talk to Derivon](mailto:info@derivon.ai)

![One algebra. Every backend.](https://raw.githubusercontent.com/derivon-ai/omnibias/62c9646964ea58b00cc3de5b80bcc2134b45f511/docs/img/explain/bias-collapse.svg)

<details>
<summary>Watch the mechanism change continuously</summary>

![Animated mathematical explanation](https://raw.githubusercontent.com/derivon-ai/omnibias/62c9646964ea58b00cc3de5b80bcc2134b45f511/docs/img/explain/bias-collapse.gif)

The animation illustrates the mechanism; it is not a speed benchmark.
</details>

## Why this package exists

A derivative engine should not need three competing implementations of its mathematics. Core gives tensor backends one source for activation polynomials, Taylor composition and mixed-index bookkeeping. Exact integer and rational work stays exact until the numerical boundary, so a new identity can be implemented once and tested independently of a training framework.

## What you can build

- Activation derivative coefficients from Riccati, Eulerian and Hermite recurrences.
- Bell / Faà di Bruno composition and shared multi-index ordering.
- Outward-rounded intervals, Taylor models and finite certificate obligations.

Choose core when implementing a backend, inspecting an identity, or building a small numerical checker without importing Torch or JAX. Training tensors and device execution belong in a backend; PDE workflows belong in consumers.

## Install the source edition

This README describes the current source tree. Published artifacts can lag this
branch, and this migration does not overwrite existing package versions. Build
the coordinated local wheelhouse using the [release guide](https://github.com/derivon-ai/omnibias/blob/main/RELEASE.md),
then install only this package and its selected dependencies:

```bash
python -m pip install --constraint artifacts/wheel-validation/constraints.txt "omnibias-core"
```

Run that command from the repository containing the generated wheelhouse.
The constraint file selects the built distributions rather than silently mixing
an older published dependency with the current source. Supported Python versions,
optional features and runtime dependencies are declared in [pyproject.toml](https://github.com/derivon-ai/omnibias/blob/main/packages/omnibias-core/pyproject.toml).

## A working example

```python
from omnibias.core import eval_tanh_derivative, tanh_polynomial_coeffs
from omnibias.core.verified import Interval

assert tanh_polynomial_coeffs(1) == (1, 0, -1)
assert eval_tanh_derivative(0.0, 3) == -2.0
box = Interval(1.0, 2.0)
print(box * box)  # outward-rounded enclosure of the product
```

## Choose the right contract

A certificate digest checks integrity, not truth. Interval assumptions, floating-point rounding and a successful formal checker are separate concerns. The optional Lean kernel is not needed for ordinary derivative evaluation; a missing checker must never become a verified verdict.

## Evidence, not a universal speed claim

The [performance guide](https://github.com/derivon-ai/omnibias/blob/main/docs/performance.md) separates activation derivatives,
specialized contractions and general deep-network jets. Its artifacts record
workloads, precision, compilation and independent accuracy checks, including
baseline wins. The reported speedups do not automatically transfer to an entire
training loop, another activation, a different device or this package’s every API.

## Explore and validate

The [API guide](https://github.com/derivon-ai/omnibias/blob/main/docs/api/core.md) contains the generated module/export
inventory. Use it to find the focused implementation rather than guessing a
symbol from another package. The [capability map](https://github.com/derivon-ai/omnibias/blob/main/docs/capabilities.md)
connects the primitives to larger scientific workflows.

From the main repository, run the package’s regression suite:

```bash
uv run pytest packages/omnibias-core/tests -q
```

Examples above are executable smoke checks, not a substitute for application
validation. For a production integration, measure the intended objective,
precision, parameter gradients, memory and wall time on representative inputs.

## License

Apache-2.0. See [LICENSE](https://github.com/derivon-ai/omnibias/blob/main/packages/omnibias-core/LICENSE) and the [licensing policy](https://github.com/derivon-ai/omnibias/blob/main/LICENSING.md).
