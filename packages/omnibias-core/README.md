# omnibias-core

**One activation. A derivative tower.** Exact coefficient algebra beneath every backend.

![One activation. A derivative tower.](https://raw.githubusercontent.com/derivon-ai/omnibias/62200950132cb8fc53cc627f4614b5380058be82/packages/omnibias-core/docs/visuals/story.gif)

[Static poster](https://raw.githubusercontent.com/derivon-ai/omnibias/62200950132cb8fc53cc627f4614b5380058be82/packages/omnibias-core/docs/visuals/poster.png) · [Narrow-screen animation](https://raw.githubusercontent.com/derivon-ai/omnibias/62200950132cb8fc53cc627f4614b5380058be82/packages/omnibias-core/docs/visuals/story-mobile.gif) · [How this visual is computed](https://raw.githubusercontent.com/derivon-ai/omnibias/62200950132cb8fc53cc627f4614b5380058be82/packages/omnibias-core/docs/visuals/scene.py)

Activation metadata and an integer order enter; shared polynomial coefficients, derivative values or checked enclosures leave. Backend authors use these exact combinatorics without importing a tensor framework.

The animation uses computed outputs to explain this package. Frame transitions
are illustrative unless a training step is explicitly identified; it is not a
performance comparison.


[API reference](https://omnibias.ai/api/core/) · [Source](https://github.com/derivon-ai/omnibias/tree/main/packages/omnibias-core/src/omnibias/core) · [Tests](https://github.com/derivon-ai/omnibias/tree/main/packages/omnibias-core/tests) · [Talk to Derivon](mailto:info@derivon.ai)

## The mathematical connection

Bias collapse starts with normalized nearby shifts of an activation. Their limit is a derivative; the Riccati recurrence evaluates that limit directly instead of subtracting nearly equal samples. One activation evaluation supplies the polynomial argument, while polynomial work still grows with order. Temperature collapse belongs to decision primitives; core supplies algebra and verified arithmetic, not a regional router.

## Run this README

The examples use `omnibias-core` on Python >=3.10. Their installed-wheel
profile is [wheel-tests.toml](https://github.com/derivon-ai/omnibias/blob/codex/pinn-substrate-split/packages/omnibias-core/wheel-tests.toml); it selects runtime features, not
an editable workspace. Build the coordinated wheelhouse using the
[release guide](https://github.com/derivon-ai/omnibias/blob/main/RELEASE.md), then run
from that main checkout:

```bash
python -m pip install --constraint artifacts/wheel-validation/constraints.txt "omnibias-core"
```

After this opt-in prerelease is published, the equivalent index command is:

```bash
python -m pip install --pre "omnibias-core==0.5.0rc1"
```

Existing published consumers may need the historical primitive versions; see the
[compatibility policy](https://github.com/derivon-ai/omnibias/blob/main/RELEASE.md#published-consumer-compatibility).

## Why this package exists

A derivative engine should not need three competing implementations of its mathematics. Core gives tensor backends one source for activation polynomials, Taylor composition and mixed-index bookkeeping. Exact integer and rational work stays exact until the numerical boundary, so a new identity can be implemented once and tested independently of a training framework.

## What you can build

- Activation derivative coefficients from Riccati, Eulerian and Hermite recurrences.
- Bell / Faà di Bruno composition and shared multi-index ordering.
- Outward-rounded intervals, Taylor models and finite certificate obligations.

Choose core when implementing a backend, inspecting an identity, or building a small numerical checker without importing Torch or JAX. Training tensors and device execution belong in a backend; PDE workflows belong in consumers.

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

## Explore and validate

The [API guide](https://github.com/derivon-ai/omnibias/blob/main/docs/api/core.md) contains the generated module/export
inventory. Use it to find the focused implementation rather than guessing a
symbol from another package. The [capability map](https://github.com/derivon-ai/omnibias/blob/main/docs/capabilities.md)
connects the primitives to larger scientific workflows.

From the main repository, run the package’s regression suite:

```bash
uv run pytest packages/omnibias-core/tests -q
```

## License

Apache-2.0. See [LICENSE](https://github.com/derivon-ai/omnibias/blob/main/packages/omnibias-core/LICENSE) and the [licensing policy](https://github.com/derivon-ai/omnibias/blob/main/LICENSING.md).
