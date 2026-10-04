# omnibias-struct

**Let paths compete before selecting one.** Dynamic programs expose smooth values and structured marginals.

![Let paths compete before selecting one.](https://raw.githubusercontent.com/derivon-ai/omnibias/62200950132cb8fc53cc627f4614b5380058be82/packages/omnibias-struct/docs/visuals/story.gif)

[Static poster](https://raw.githubusercontent.com/derivon-ai/omnibias/62200950132cb8fc53cc627f4614b5380058be82/packages/omnibias-struct/docs/visuals/poster.png) · [Narrow-screen animation](https://raw.githubusercontent.com/derivon-ai/omnibias/62200950132cb8fc53cc627f4614b5380058be82/packages/omnibias-struct/docs/visuals/story-mobile.gif) · [How this visual is computed](https://raw.githubusercontent.com/derivon-ai/omnibias/62200950132cb8fc53cc627f4614b5380058be82/packages/omnibias-struct/docs/visuals/scene.py)

Scores on a sequence, graph or grammar enter; soft dynamic-program values, marginals and decoded structures leave. The recurrence exploits structure rather than enumerating all paths as independent neural experts.

The animation uses computed outputs to explain this package. Frame transitions
are illustrative unless a training step is explicitly identified; it is not a
performance comparison.


[API reference](https://omnibias.ai/api/struct/) · [Source](https://github.com/derivon-ai/omnibias/tree/main/packages/omnibias-struct/src/omnibias/struct) · [Tests](https://github.com/derivon-ai/omnibias/tree/main/packages/omnibias-struct/tests) · [Talk to Derivon](mailto:info@derivon.ai)

## The mathematical connection

Temperature collapse replaces a hard max or min with smooth log-sum-exp competition. At high β, values approach the hard recurrence; tied alternatives retain shared soft mass. Bias-collapse jets differentiate supported log-sum-exp compositions at higher order. The resulting derivatives concern the soft program, not a discontinuous argmax path.

## Run this README

The examples use `omnibias-struct[torch]` on Python >=3.10. Their installed-wheel
profile selects runtime features, not an editable workspace. Install the prepared
prerelease from PyPI:

```bash
python -m pip install --pre "omnibias-struct[torch]==0.1.0a2"
```

For local development before publication, build and test the coordinated wheelhouse
using the [release guide](https://github.com/derivon-ai/omnibias/blob/main/RELEASE.md).
The package's [wheel profile](https://github.com/derivon-ai/omnibias/blob/main/packages/omnibias-struct/wheel-tests.toml)
executes the examples below outside the source checkout.

Existing published consumers may need historical primitive versions; see the
[compatibility policy](https://github.com/derivon-ai/omnibias/blob/main/RELEASE.md#published-consumer-compatibility).

## Why this package exists

A max-score path selects one discrete explanation. Replacing max with a temperature-scaled log-sum-exp retains information about alternatives, making the structured value differentiable with respect to scores. Struct combines these recurrences with shared semiring representations and derivative machinery.

## What you can build

- Soft Viterbi, alignment, DTW, planning and related structured operators.
- Semiring/hypergraph representations and selected parsing families.
- Marginals, higher derivatives, decoding and soft-versus-hard gap bounds.

Use struct when a neural model produces scores for a sequence or structured decision and the final loss should train those scores. The recurrence is the reusable primitive; task-specific tokenization, datasets, supervision and application objectives belong in the consuming project.

## A working example

```python
import torch
from omnibias.struct.torch import soft_viterbi

emissions = torch.tensor([[0.3, -0.2], [0.1, 0.5], [0.4, 0.2]], requires_grad=True)
transitions = torch.zeros((2, 2), requires_grad=True)
value = soft_viterbi(emissions, transitions, beta=3.0)
value.backward()
assert emissions.grad is not None
assert torch.allclose(emissions.grad.sum(dim=1), torch.ones(3))
```

## Choose the right contract

Temperature bounds depend on the finite alternatives being counted. A bound on the smoothed optimal value is not a proof that every learned decoder is correct. Tensor shapes and semiring conventions matter, and different structured families have different computational costs.

## Explore and validate

The [API guide](https://github.com/derivon-ai/omnibias/blob/main/docs/api/struct.md) contains the generated module/export
inventory. Use it to find the focused implementation rather than guessing a
symbol from another package. The [capability map](https://github.com/derivon-ai/omnibias/blob/main/docs/capabilities.md)
connects the primitives to larger scientific workflows.

From the main repository, run the package’s regression suite:

```bash
uv run pytest packages/omnibias-struct/tests -q
```

## License

AGPL-3.0-or-later **or commercial**. See [LICENSE](https://github.com/derivon-ai/omnibias/blob/main/packages/omnibias-struct/LICENSE) and the [licensing policy](https://github.com/derivon-ai/omnibias/blob/main/LICENSING.md).
Comply with the AGPL terms or obtain a signed commercial grant; commercial use alone does not require payment. [Commercial terms](https://github.com/derivon-ai/omnibias/blob/main/packages/omnibias-struct/COMMERCIAL-LICENSE.md) describe the alternative. Previously distributed Apache editions, where applicable, retain their original grants.
