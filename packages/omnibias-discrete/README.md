# omnibias-discrete

## Relax. Decode. Measure the gap.

**A shared contract for differentiable optimization over binary decisions.**

[API reference](https://omnibias.ai/api/discrete/) · [Source](https://github.com/derivon-ai/omnibias/tree/main/packages/omnibias-discrete/src/omnibias/discrete) · [Tests](https://github.com/derivon-ai/omnibias/tree/main/packages/omnibias-discrete/tests) · [Talk to Derivon](mailto:info@derivon.ai)

![Relax. Decode. Measure the gap.](https://raw.githubusercontent.com/derivon-ai/omnibias/62c9646964ea58b00cc3de5b80bcc2134b45f511/docs/img/explain/temperature-collapse.svg)

<details>
<summary>Watch the mechanism change continuously</summary>

![Animated mathematical explanation](https://raw.githubusercontent.com/derivon-ai/omnibias/62c9646964ea58b00cc3de5b80bcc2134b45f511/docs/img/explain/temperature-collapse.gif)

The animation illustrates the mechanism; it is not a speed benchmark.
</details>

## Why this package exists

Discrete applications often repeat the same infrastructure: represent an energy, relax binary variables, sharpen a temperature schedule, round a candidate and compare it with a bound. Discrete gives those steps explicit interfaces, so consumers can specialize a problem without copying its optimization plumbing.

## What you can build

- DiscreteProblem and DiscreteSolution interfaces.
- Annealed Torch/JAX relaxation and reusable proposers.
- Rounding, local descent, bounded brute-force oracles and optimality-gap certificates.

Use this primitive under QUBO, routing, logic or other discrete frontends. A feasible candidate and a lower bound are independently useful outputs: the candidate supplies an actionable decision, while the bound quantifies what remains unproven about its quality.

## Install the source edition

This README describes the current source tree. Published artifacts can lag this
branch, and this migration does not overwrite existing package versions. Build
the coordinated local wheelhouse using the [release guide](https://github.com/derivon-ai/omnibias/blob/main/RELEASE.md),
then install only this package and its selected dependencies:

```bash
python -m pip install --constraint artifacts/wheel-validation/constraints.txt "omnibias-discrete"
```

Run that command from the repository containing the generated wheelhouse.
The constraint file selects the built distributions rather than silently mixing
an older published dependency with the current source. Supported Python versions,
optional features and runtime dependencies are declared in [pyproject.toml](https://github.com/derivon-ai/omnibias/blob/main/packages/omnibias-discrete/pyproject.toml).

## A working example

```python
import numpy as np
from omnibias.discrete import AnnealSchedule, round_relaxed

schedule = AnnealSchedule(beta0=0.5, beta_growth=2.0, stages=4)
assert schedule.betas() == [0.5, 1.0, 2.0, 4.0]
probabilities = np.array([0.1, 0.8, 0.3])
candidate = round_relaxed(probabilities)
assert np.array_equal(candidate, [0., 1., 0.])
# Rounding creates a candidate; certify_gap needs a separately justified bound.
```

## Choose the right contract

Temperature hardening does not make a difficult discrete search globally optimal. Keep feasibility, objective value, bound provenance and certificate status separate. Brute-force enumeration is only a small-instance oracle. Optional SOS or convex backends provide specific bounds under their own assumptions.

## Explore and validate

The [API guide](https://github.com/derivon-ai/omnibias/blob/main/docs/api/discrete.md) contains the generated module/export
inventory. Use it to find the focused implementation rather than guessing a
symbol from another package. The [capability map](https://github.com/derivon-ai/omnibias/blob/main/docs/capabilities.md)
connects the primitives to larger scientific workflows.

From the main repository, run the package’s regression suite:

```bash
uv run pytest packages/omnibias-discrete/tests -q
```

Examples above are executable smoke checks, not a substitute for application
validation. For a production integration, measure the intended objective,
precision, parameter gradients, memory and wall time on representative inputs.

## License

AGPL-3.0-or-later **or commercial**. See [LICENSE](https://github.com/derivon-ai/omnibias/blob/main/packages/omnibias-discrete/LICENSE) and the [licensing policy](https://github.com/derivon-ai/omnibias/blob/main/LICENSING.md).
Comply with the AGPL terms or obtain a signed commercial grant; commercial use alone does not require payment. [Commercial terms](https://github.com/derivon-ai/omnibias/blob/main/packages/omnibias-discrete/COMMERCIAL-LICENSE.md) describe the alternative. Previously distributed Apache editions, where applicable, retain their original grants.
