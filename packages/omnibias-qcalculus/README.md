# omnibias-qcalculus

## A calculus with a scale parameter.

**q-numbers, Jackson operators and q-deformed polynomial and series algebra.**

[API reference](https://omnibias.ai/api/qcalculus/) · [Source](https://github.com/derivon-ai/omnibias/tree/main/packages/omnibias-qcalculus/src/omnibias/qcalculus) · [Tests](https://github.com/derivon-ai/omnibias/tree/main/packages/omnibias-qcalculus/tests) · [Talk to Derivon](mailto:info@derivon.ai)

## Why this package exists

Some discrete and multiplicative-scale problems are expressed more naturally by q-differences than by ordinary shifts. Qcalculus exposes that deformation explicitly and connects it to the ordinary derivative as q approaches one. Exact polynomial operations make the relation easy to inspect without numerical differencing.

## What you can build

- q-brackets, factorials, binomials and polynomial transforms.
- Jackson derivatives and antiderivatives.
- q-exponential families, series bounds and optional tensor realizations.

Use qcalculus for multiplicative sampling, q-series experiments and time-scale or symbolic consumers that need this register. Its q→1 limit is a separate mechanism from bias collapse and temperature hardening. Select the register that represents the mathematical problem rather than treating the parameters as interchangeable temperatures.

## Install the source edition

This README describes the current source tree. Published artifacts can lag this
branch, and this migration does not overwrite existing package versions. Build
the coordinated local wheelhouse using the [release guide](https://github.com/derivon-ai/omnibias/blob/main/RELEASE.md),
then install only this package and its selected dependencies:

```bash
python -m pip install --constraint artifacts/wheel-validation/constraints.txt "omnibias-qcalculus"
```

Run that command from the repository containing the generated wheelhouse.
The constraint file selects the built distributions rather than silently mixing
an older published dependency with the current source. Supported Python versions,
optional features and runtime dependencies are declared in [pyproject.toml](https://github.com/derivon-ai/omnibias/blob/main/packages/omnibias-qcalculus/pyproject.toml).

## A working example

```python
from fractions import Fraction
from omnibias.qcalculus import q_derivative_poly

# Coefficients are ordered from constant term upward: f(x) = x**2.
assert q_derivative_poly([0, 0, 1], Fraction(1, 2)) == (Fraction(0), Fraction(3, 2))
assert q_derivative_poly([0, 0, 1], Fraction(1)) == (Fraction(0), Fraction(2))
```

## Choose the right contract

Numeric series require a supported q-domain and truncation/convergence controls. Near q=1, a direct quotient may be poorly conditioned; prefer the explicit limit or polynomial path where available. Exact rational coefficients do not make arbitrary floating-point series evaluations exact.

## Explore and validate

The [API guide](https://github.com/derivon-ai/omnibias/blob/main/docs/api/qcalculus.md) contains the generated module/export
inventory. Use it to find the focused implementation rather than guessing a
symbol from another package. The [capability map](https://github.com/derivon-ai/omnibias/blob/main/docs/capabilities.md)
connects the primitives to larger scientific workflows.

From the main repository, run the package’s regression suite:

```bash
uv run pytest packages/omnibias-qcalculus/tests -q
```

Examples above are executable smoke checks, not a substitute for application
validation. For a production integration, measure the intended objective,
precision, parameter gradients, memory and wall time on representative inputs.

## License

Apache-2.0. See [LICENSE](https://github.com/derivon-ai/omnibias/blob/main/packages/omnibias-qcalculus/LICENSE) and the [licensing policy](https://github.com/derivon-ai/omnibias/blob/main/LICENSING.md).
