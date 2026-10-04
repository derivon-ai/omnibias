# omnibias-qcalculus

**Calculus on a geometric grid.** Exact q-polynomial coefficients approach ordinary derivatives.

![Calculus on a geometric grid.](https://raw.githubusercontent.com/derivon-ai/omnibias/62200950132cb8fc53cc627f4614b5380058be82/packages/omnibias-qcalculus/docs/visuals/story.gif)

[Static poster](https://raw.githubusercontent.com/derivon-ai/omnibias/62200950132cb8fc53cc627f4614b5380058be82/packages/omnibias-qcalculus/docs/visuals/poster.png) · [Narrow-screen animation](https://raw.githubusercontent.com/derivon-ai/omnibias/62200950132cb8fc53cc627f4614b5380058be82/packages/omnibias-qcalculus/docs/visuals/story-mobile.gif) · [How this visual is computed](https://raw.githubusercontent.com/derivon-ai/omnibias/62200950132cb8fc53cc627f4614b5380058be82/packages/omnibias-qcalculus/docs/visuals/scene.py)

A q-parameter and algebraic coefficients enter; q-numbers, Jackson derivatives and q-integrals leave. Multiplicative sampling supports calculations on a geometric grid, distinct from an additive finite-difference stencil.

The animation uses computed outputs to explain this package. Frame transitions
are illustrative unless a training step is explicitly identified; it is not a
performance comparison.


[API reference](https://omnibias.ai/api/qcalculus/) · [Source](https://github.com/derivon-ai/omnibias/tree/main/packages/omnibias-qcalculus/src/omnibias/qcalculus) · [Tests](https://github.com/derivon-ai/omnibias/tree/main/packages/omnibias-qcalculus/tests) · [Talk to Derivon](mailto:info@derivon.ai)

## The mathematical connection

The defining limit here is q → 1, which recovers ordinary calculus. It is distinct from both bias collapse (normalized nearby shifts) and temperature collapse (sharpening soft alternatives). Neither founding mechanism should be substituted for the q-calculus operator definition or its domain restrictions.

## Run this README

The examples use `omnibias-qcalculus` on Python >=3.10. Their installed-wheel
profile is [wheel-tests.toml](https://github.com/derivon-ai/omnibias/blob/codex/pinn-substrate-split/packages/omnibias-qcalculus/wheel-tests.toml); it selects runtime features, not
an editable workspace. Build the coordinated wheelhouse using the
[release guide](https://github.com/derivon-ai/omnibias/blob/main/RELEASE.md), then run
from that main checkout:

```bash
python -m pip install --constraint artifacts/wheel-validation/constraints.txt "omnibias-qcalculus"
```

After this opt-in prerelease is published, the equivalent index command is:

```bash
python -m pip install --pre "omnibias-qcalculus==0.1.0a2"
```

Existing published consumers may need the historical primitive versions; see the
[compatibility policy](https://github.com/derivon-ai/omnibias/blob/main/RELEASE.md#published-consumer-compatibility).

## Why this package exists

Some discrete and multiplicative-scale problems are expressed more naturally by q-differences than by ordinary shifts. Qcalculus exposes that deformation explicitly and connects it to the ordinary derivative as q approaches one. Exact polynomial operations make the relation easy to inspect without numerical differencing.

## What you can build

- q-brackets, factorials, binomials and polynomial transforms.
- Jackson derivatives and antiderivatives.
- q-exponential families, series bounds and optional tensor realizations.

Use qcalculus for multiplicative sampling, q-series experiments and time-scale or symbolic consumers that need this register. Its q→1 limit is a separate mechanism from bias collapse and temperature hardening. Select the register that represents the mathematical problem rather than treating the parameters as interchangeable temperatures.

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

## License

Apache-2.0. See [LICENSE](https://github.com/derivon-ai/omnibias/blob/main/packages/omnibias-qcalculus/LICENSE) and the [licensing policy](https://github.com/derivon-ai/omnibias/blob/main/LICENSING.md).
