# omnibias-sos

**Turn positivity into a checkable witness.** A decomposition earns a scoped certificate through explicit checks.

![Turn positivity into a checkable witness.](https://raw.githubusercontent.com/derivon-ai/omnibias/62200950132cb8fc53cc627f4614b5380058be82/packages/omnibias-sos/docs/visuals/story.gif)

[Static poster](https://raw.githubusercontent.com/derivon-ai/omnibias/62200950132cb8fc53cc627f4614b5380058be82/packages/omnibias-sos/docs/visuals/poster.png) · [Narrow-screen animation](https://raw.githubusercontent.com/derivon-ai/omnibias/62200950132cb8fc53cc627f4614b5380058be82/packages/omnibias-sos/docs/visuals/story-mobile.gif) · [How this visual is computed](https://raw.githubusercontent.com/derivon-ai/omnibias/62200950132cb8fc53cc627f4614b5380058be82/packages/omnibias-sos/docs/visuals/scene.py)

A polynomial positivity problem enters; a proposed Gram decomposition and rigorously checked certificate data leave. The useful distinction is between finding a numerical candidate and earning a positivity verdict.

The animation uses computed outputs to explain this package. Frame transitions
are illustrative unless a training step is explicitly identified; it is not a
performance comparison.


[API reference](https://omnibias.ai/api/sos/) · [Source](https://github.com/derivon-ai/omnibias/tree/main/packages/omnibias-sos/src/omnibias/sos) · [Tests](https://github.com/derivon-ai/omnibias/tree/main/packages/omnibias-sos/tests) · [Talk to Derivon](mailto:info@derivon.ai)

## The mathematical connection

SOS is certificate infrastructure and directly implements neither bias collapse nor temperature collapse. A smooth or discrete consumer can use its lower bounds, but annealing success does not imply polynomial positivity. The checker’s domain, rounding and PSD obligations determine what is certified; a proposal alone is insufficient.

## Run this README

The examples use `omnibias-sos` on Python >=3.10. Their installed-wheel
profile selects runtime features, not an editable workspace. Install the prepared
prerelease from PyPI:

```bash
python -m pip install --pre "omnibias-sos==0.1.0a2"
```

For local development before publication, build and test the coordinated wheelhouse
using the [release guide](https://github.com/derivon-ai/omnibias/blob/main/RELEASE.md).
The package's [wheel profile](https://github.com/derivon-ai/omnibias/blob/main/packages/omnibias-sos/wheel-tests.toml)
executes the examples below outside the source checkout.

Existing published consumers may need historical primitive versions; see the
[compatibility policy](https://github.com/derivon-ai/omnibias/blob/main/RELEASE.md#published-consumer-compatibility).

## Why this package exists

A plot that looks positive is not a proof. An SOS certificate represents a polynomial through a Gram matrix and checks the algebra and positivity conditions needed for the stated claim. The distinction between finding a candidate and validating it is central to this package.

## What you can build

- Polynomial and monomial-basis representations.
- SOS and constrained Positivstellensatz workflows.
- Rational reconstruction, interval pivot checks and optional formal obligations.

Use SOS to justify polynomial nonnegativity, construct lower bounds or support another package’s certificate. Small examples are useful for learning the workflow; higher degree and dimension increase basis and semidefinite costs. A carefully chosen basis can matter as much as the numerical proposer.

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

## License

AGPL-3.0-or-later **or commercial**. See [LICENSE](https://github.com/derivon-ai/omnibias/blob/main/packages/omnibias-sos/LICENSE) and the [licensing policy](https://github.com/derivon-ai/omnibias/blob/main/LICENSING.md).
Comply with the AGPL terms or obtain a signed commercial grant; commercial use alone does not require payment. [Commercial terms](https://github.com/derivon-ai/omnibias/blob/main/packages/omnibias-sos/COMMERCIAL-LICENSE.md) describe the alternative. Previously distributed Apache editions, where applicable, retain their original grants.
