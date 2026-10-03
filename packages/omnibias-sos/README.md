# omnibias-sos

## Turn positivity into checkable algebra.

**Sum-of-squares proposals, rational reconstruction and scoped polynomial certificates.**

[API reference](https://omnibias.ai/api/sos/) · [Source](https://github.com/derivon-ai/omnibias/tree/main/packages/omnibias-sos/src/omnibias/sos) · [Tests](https://github.com/derivon-ai/omnibias/tree/main/packages/omnibias-sos/tests) · [Talk to Derivon](mailto:info@derivon.ai)

## Why this package exists

A plot that looks positive is not a proof. An SOS certificate represents a polynomial through a Gram matrix and checks the algebra and positivity conditions needed for the stated claim. The distinction between finding a candidate and validating it is central to this package.

## What you can build

- Polynomial and monomial-basis representations.
- SOS and constrained Positivstellensatz workflows.
- Rational reconstruction, interval pivot checks and optional formal obligations.

Use SOS to justify polynomial nonnegativity, construct lower bounds or support another package’s certificate. Small examples are useful for learning the workflow; higher degree and dimension increase basis and semidefinite costs. A carefully chosen basis can matter as much as the numerical proposer.

## Install the source edition

This README describes the current source tree. Published artifacts can lag this
branch, and this migration does not overwrite existing package versions. Build
the coordinated local wheelhouse using the [release guide](https://github.com/derivon-ai/omnibias/blob/main/RELEASE.md),
then install only this package and its selected dependencies:

```bash
python -m pip install --constraint artifacts/wheel-validation/constraints.txt "omnibias-sos"
```

Run that command from the repository containing the generated wheelhouse.
The constraint file selects the built distributions rather than silently mixing
an older published dependency with the current source. Supported Python versions,
optional features and runtime dependencies are declared in [pyproject.toml](https://github.com/derivon-ai/omnibias/blob/main/packages/omnibias-sos/pyproject.toml).

## A working example

```python
from omnibias.sos import Polynomial, certify_sos

x = Polynomial.variable(0, 1)
p = x*x + Polynomial.constant(1., 1)
certificate = certify_sos(p)
assert certificate.certified
assert certificate.coeff_residual == 0.0
print(certificate.status)
```

## Choose the right contract

Not every nonnegative polynomial is SOS. Failure or an inconclusive result does not prove negativity. Certificate scope records the domain and assumptions; a finite algebraic replay is not automatically a theorem about an unrepresented analytic model. Report rigorous and formal-check flags separately.

## Explore and validate

The [API guide](https://github.com/derivon-ai/omnibias/blob/main/docs/api/sos.md) contains the generated module/export
inventory. Use it to find the focused implementation rather than guessing a
symbol from another package. The [capability map](https://github.com/derivon-ai/omnibias/blob/main/docs/capabilities.md)
connects the primitives to larger scientific workflows.

From the main repository, run the package’s regression suite:

```bash
uv run pytest packages/omnibias-sos/tests -q
```

Examples above are executable smoke checks, not a substitute for application
validation. For a production integration, measure the intended objective,
precision, parameter gradients, memory and wall time on representative inputs.

## License

AGPL-3.0-or-later **or commercial**. See [LICENSE](https://github.com/derivon-ai/omnibias/blob/main/packages/omnibias-sos/LICENSE) and the [licensing policy](https://github.com/derivon-ai/omnibias/blob/main/LICENSING.md).
Comply with the AGPL terms or obtain a signed commercial grant; commercial use alone does not require payment. [Commercial terms](https://github.com/derivon-ai/omnibias/blob/main/packages/omnibias-sos/COMMERCIAL-LICENSE.md) describe the alternative. Previously distributed Apache editions, where applicable, retain their original grants.
