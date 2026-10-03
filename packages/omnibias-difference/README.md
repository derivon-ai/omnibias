# omnibias-difference

## From finite steps to exact structure.

**Finite-difference extraction, irregular stencils, exact sequence algebra and recurrence fitting.**

[API reference](https://omnibias.ai/api/difference/) · [Source](https://github.com/derivon-ai/omnibias/tree/main/packages/omnibias-difference/src/omnibias/difference) · [Tests](https://github.com/derivon-ai/omnibias/tree/main/packages/omnibias-difference/tests) · [Talk to Derivon](mailto:info@derivon.ai)

![From finite steps to exact structure.](https://raw.githubusercontent.com/derivon-ai/omnibias/62c9646964ea58b00cc3de5b80bcc2134b45f511/docs/img/explain/bias-collapse.svg)

<details>
<summary>Watch the mechanism change continuously</summary>

![Animated mathematical explanation](https://raw.githubusercontent.com/derivon-ai/omnibias/62c9646964ea58b00cc3de5b80bcc2134b45f511/docs/img/explain/bias-collapse.gif)

The animation illustrates the mechanism; it is not a speed benchmark.
</details>

## Why this package exists

Sampled data and discrete sequences need a different register from an analytic neural activation. Difference makes the step size, stencil and algebra explicit. You can construct irregular derivative estimates, reason about their truncation bounds, or recover exact rational recurrence candidates from a finite sequence.

## What you can build

- Irregular stencil construction and moment identities.
- Finite-step estimates with separately supplied regularity and remainder information.
- Umbral algebra, series transforms and one canonical exact recurrence fitter.

Use stencils when you have samples; use activation towers when you have a supported analytic model. Use recurrence fitting to propose compact sequence laws for symbolic or holonomic workflows. Keeping these modes separate prevents a numerical difference estimate from being presented as an exact derivative.

## Install the source edition

This README describes the current source tree. Published artifacts can lag this
branch, and this migration does not overwrite existing package versions. Build
the coordinated local wheelhouse using the [release guide](https://github.com/derivon-ai/omnibias/blob/main/RELEASE.md),
then install only this package and its selected dependencies:

```bash
python -m pip install --constraint artifacts/wheel-validation/constraints.txt "omnibias-difference"
```

Run that command from the repository containing the generated wheelhouse.
The constraint file selects the built distributions rather than silently mixing
an older published dependency with the current source. Supported Python versions,
optional features and runtime dependencies are declared in [pyproject.toml](https://github.com/derivon-ai/omnibias/blob/main/packages/omnibias-difference/pyproject.toml).

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

Examples above are executable smoke checks, not a substitute for application
validation. For a production integration, measure the intended objective,
precision, parameter gradients, memory and wall time on representative inputs.

## License

Apache-2.0. See [LICENSE](https://github.com/derivon-ai/omnibias/blob/main/packages/omnibias-difference/LICENSE) and the [licensing policy](https://github.com/derivon-ai/omnibias/blob/main/LICENSING.md).
