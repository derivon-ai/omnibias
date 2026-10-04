# omnibias-difference

**Differentiate the samples you have.** Sample locations and weights determine the discrete operator.

![Differentiate the samples you have.](https://raw.githubusercontent.com/derivon-ai/omnibias/62200950132cb8fc53cc627f4614b5380058be82/packages/omnibias-difference/docs/visuals/story.gif)

[Static poster](https://raw.githubusercontent.com/derivon-ai/omnibias/62200950132cb8fc53cc627f4614b5380058be82/packages/omnibias-difference/docs/visuals/poster.png) · [Narrow-screen animation](https://raw.githubusercontent.com/derivon-ai/omnibias/62200950132cb8fc53cc627f4614b5380058be82/packages/omnibias-difference/docs/visuals/story-mobile.gif) · [How this visual is computed](https://raw.githubusercontent.com/derivon-ai/omnibias/62200950132cb8fc53cc627f4614b5380058be82/packages/omnibias-difference/docs/visuals/scene.py)

Sample locations, values and an extraction order enter; stencil coefficients, derivative estimates and scoped remainder information leave. Use this package when the source is discrete data or an exact recurrence rather than a trainable tensor network.

The animation uses computed outputs to explain this package. Frame transitions
are illustrative unless a training step is explicitly identified; it is not a
performance comparison.


[API reference](https://github.com/derivon-ai/omnibias/blob/main/docs/api/difference.md) · [Source](https://github.com/derivon-ai/omnibias/tree/main/packages/omnibias-difference/src/omnibias/difference) · [Tests](https://github.com/derivon-ai/omnibias/tree/main/packages/omnibias-difference/tests) · [Talk to Derivon](mailto:info@derivon.ai)

## The mathematical connection

This package exposes the finite-spacing side of bias collapse. Normalized weighted shifts approach derivatives as the spacing vanishes; at nonzero spacing, truncation and cancellation must still be assessed. Its umbral and exact recurrence tools also operate algebraically. Temperature collapse does not define this sampling contract.

## Run this README

The examples use `omnibias-difference` on Python >=3.10. Their installed-wheel
profile selects runtime features, not an editable workspace. Install the prepared
prerelease from PyPI:

```bash
python -m pip install --pre "omnibias-difference==0.1.0a2"
```

For local development before publication, build and test the coordinated wheelhouse
using the [release guide](https://github.com/derivon-ai/omnibias/blob/main/RELEASE.md).
The package's [wheel profile](https://github.com/derivon-ai/omnibias/blob/main/packages/omnibias-difference/wheel-tests.toml)
executes the examples below outside the source checkout.

Existing published consumers may need historical primitive versions; see the
[compatibility policy](https://github.com/derivon-ai/omnibias/blob/main/RELEASE.md#published-consumer-compatibility).

## Why this package exists

Sampled data and discrete sequences need a different register from an analytic neural activation. Difference makes the step size, stencil and algebra explicit. You can construct irregular derivative estimates, reason about their truncation bounds, or recover exact rational recurrence candidates from a finite sequence.

## What you can build

- Irregular stencil construction and moment identities.
- Finite-step estimates with separately supplied regularity and remainder information.
- Umbral algebra, series transforms and one canonical exact recurrence fitter.

Use stencils when you have samples; use activation towers when you have a supported analytic model. Use recurrence fitting to propose compact sequence laws for symbolic or holonomic workflows. Keeping these modes separate prevents a numerical difference estimate from being presented as an exact derivative.

## A working example

```python
from math import factorial
from omnibias.difference.recurrence import discover_recurrence

relation = discover_recurrence([factorial(n) for n in range(11)])
assert relation is not None
assert relation.order == 1
assert relation.max_abs_residual([factorial(n) for n in range(16)]) == 0
print(relation.coefficients)  # exact rational coefficients for a_n - n*a_(n-1)
```

## Choose the right contract

A finite prefix can fit multiple recurrences. A zero residual on supplied samples is not a proof for all sequence indices; use held-out values and an independent identity argument. Likewise, a truncation certificate depends on its derivative bounds and does not automatically account for every floating-point error.

## Explore and validate

The [API guide](https://github.com/derivon-ai/omnibias/blob/main/docs/api/difference.md) contains the generated module/export
inventory. Use it to find the focused implementation rather than guessing a
symbol from another package. The [capability map](https://github.com/derivon-ai/omnibias/blob/main/docs/capabilities.md)
connects the primitives to larger scientific workflows.

From the main repository, run the package’s regression suite:

```bash
uv run pytest packages/omnibias-difference/tests -q
```

## License

Apache-2.0. See [LICENSE](https://github.com/derivon-ai/omnibias/blob/main/packages/omnibias-difference/LICENSE) and the [licensing policy](https://github.com/derivon-ai/omnibias/blob/main/LICENSING.md).
