# omnibias-boolean

## Exact logic, inspectable structure.

**Truth tables, algebraic normal forms, Walsh spectra and Boolean differential calculus.**

[API reference](https://omnibias.ai/api/boolean/) · [Source](https://github.com/derivon-ai/omnibias/tree/main/packages/omnibias-boolean/src/omnibias/boolean) · [Tests](https://github.com/derivon-ai/omnibias/tree/main/packages/omnibias-boolean/tests) · [Talk to Derivon](mailto:info@derivon.ai)

## Why this package exists

Boolean functions have algebraic structure that a generic real-valued tensor does not expose. Boolean lets you move between exact finite representations, inspect interactions with discrete derivatives, and solve bounded algebraic systems. Optional tensor realizations connect those structures to learnable soft gates.

## What you can build

- Truth tables and algebraic normal forms over GF(2).
- Walsh transforms, influences and exact Boolean derivatives.
- Finite equation solvers, with optional Torch/JAX gate and spectrum operations.

Choose exact mode for truth-table identities, bounded logic tests and reference oracles. Choose a tensor realization when the purpose is optimization through a smooth model. Keeping the two separate lets you compare a learned gate with the exact finite behavior it is intended to approximate.

## Install the source edition

This README describes the current source tree. Published artifacts can lag this
branch, and this migration does not overwrite existing package versions. Build
the coordinated local wheelhouse using the [release guide](https://github.com/derivon-ai/omnibias/blob/main/RELEASE.md),
then install only this package and its selected dependencies:

```bash
python -m pip install --constraint artifacts/wheel-validation/constraints.txt "omnibias-boolean"
```

Run that command from the repository containing the generated wheelhouse.
The constraint file selects the built distributions rather than silently mixing
an older published dependency with the current source. Supported Python versions,
optional features and runtime dependencies are declared in [pyproject.toml](https://github.com/derivon-ai/omnibias/blob/main/packages/omnibias-boolean/pyproject.toml).

## A working example

```python
from omnibias.boolean import truth_table_from_callable, anf_from_truth_table
from omnibias.boolean import truth_table_from_anf

xor = truth_table_from_callable(lambda a, b: a ^ b, 2)
polynomial = anf_from_truth_table(xor)
assert truth_table_from_anf(polynomial) == xor
print(polynomial)
```

## Choose the right contract

Truth-table size is exponential in the number of variables. Exact enumeration is a bounded reference tool, not a polynomial-time solver for arbitrary large Boolean systems. A soft gate gradient and a Boolean derivative are different operators with different meanings.

## Explore and validate

The [API guide](https://github.com/derivon-ai/omnibias/blob/main/docs/api/boolean.md) contains the generated module/export
inventory. Use it to find the focused implementation rather than guessing a
symbol from another package. The [capability map](https://github.com/derivon-ai/omnibias/blob/main/docs/capabilities.md)
connects the primitives to larger scientific workflows.

From the main repository, run the package’s regression suite:

```bash
uv run pytest packages/omnibias-boolean/tests -q
```

Examples above are executable smoke checks, not a substitute for application
validation. For a production integration, measure the intended objective,
precision, parameter gradients, memory and wall time on representative inputs.

## License

Apache-2.0. See [LICENSE](https://github.com/derivon-ai/omnibias/blob/main/packages/omnibias-boolean/LICENSE) and the [licensing policy](https://github.com/derivon-ai/omnibias/blob/main/LICENSING.md).
