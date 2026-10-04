# omnibias-discrete

**A decision needs more than rounding.** Represent a relaxation, decode a candidate, then measure its gap.

![A decision needs more than rounding.](https://raw.githubusercontent.com/derivon-ai/omnibias/62200950132cb8fc53cc627f4614b5380058be82/packages/omnibias-discrete/docs/visuals/story.gif)

[Static poster](https://raw.githubusercontent.com/derivon-ai/omnibias/62200950132cb8fc53cc627f4614b5380058be82/packages/omnibias-discrete/docs/visuals/poster.png) · [Narrow-screen animation](https://raw.githubusercontent.com/derivon-ai/omnibias/62200950132cb8fc53cc627f4614b5380058be82/packages/omnibias-discrete/docs/visuals/story-mobile.gif) · [How this visual is computed](https://raw.githubusercontent.com/derivon-ai/omnibias/62200950132cb8fc53cc627f4614b5380058be82/packages/omnibias-discrete/docs/visuals/scene.py)

A discrete problem, relaxation schedule and candidate enter a shared optimization interface. Relaxed values, a decoded assignment and an independently justified optimality gap leave. Front-ends supply the objective and feasible-set meaning.

The animation uses computed outputs to explain this package. Frame transitions
are illustrative unless a training step is explicitly identified; it is not a
performance comparison.


[API reference](https://github.com/derivon-ai/omnibias/blob/main/docs/api/discrete.md) · [Source](https://github.com/derivon-ai/omnibias/tree/main/packages/omnibias-discrete/src/omnibias/discrete) · [Tests](https://github.com/derivon-ai/omnibias/tree/main/packages/omnibias-discrete/tests) · [Talk to Derivon](mailto:info@derivon.ai)

## The mathematical connection

Temperature collapse provides a route from continuous relaxation toward discrete candidates. Finite β does not guarantee an integral or globally optimal result; decoding and lower bounds remain separate stages, including at ties. Bias-collapse backends may differentiate a smooth objective, but the problem/anneal/decode/certify contract is the package’s distinctive role.

## Run this README

The examples use `omnibias-discrete` on Python >=3.10. Their installed-wheel
profile selects runtime features, not an editable workspace. Install the prepared
prerelease from PyPI:

```bash
python -m pip install --pre "omnibias-discrete==0.1.0a2"
```

For local development before publication, build and test the coordinated wheelhouse
using the [release guide](https://github.com/derivon-ai/omnibias/blob/main/RELEASE.md).
The package's [wheel profile](https://github.com/derivon-ai/omnibias/blob/main/packages/omnibias-discrete/wheel-tests.toml)
executes the examples below outside the source checkout.

Existing published consumers may need historical primitive versions; see the
[compatibility policy](https://github.com/derivon-ai/omnibias/blob/main/RELEASE.md#published-consumer-compatibility).

## Why this package exists

Discrete applications often repeat the same infrastructure: represent an energy, relax binary variables, sharpen a temperature schedule, round a candidate and compare it with a bound. Discrete gives those steps explicit interfaces, so consumers can specialize a problem without copying its optimization plumbing.

## What you can build

- DiscreteProblem and DiscreteSolution interfaces.
- Annealed Torch/JAX relaxation and reusable proposers.
- Rounding, local descent, bounded brute-force oracles and optimality-gap certificates.

Use this primitive under QUBO, routing, logic or other discrete frontends. A feasible candidate and a lower bound are independently useful outputs: the candidate supplies an actionable decision, while the bound quantifies what remains unproven about its quality.

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

## License

AGPL-3.0-or-later **or commercial**. See [LICENSE](https://github.com/derivon-ai/omnibias/blob/main/packages/omnibias-discrete/LICENSE) and the [licensing policy](https://github.com/derivon-ai/omnibias/blob/main/LICENSING.md).
Comply with the AGPL terms or obtain a signed commercial grant; commercial use alone does not require payment. [Commercial terms](https://github.com/derivon-ai/omnibias/blob/main/packages/omnibias-discrete/COMMERCIAL-LICENSE.md) describe the alternative. Previously distributed Apache editions, where applicable, retain their original grants.
