# omnibias-struct

## Dynamic programs that participate in training.

**Smooth structured computation on sequences, paths and hypergraphs, with explicit hard-limit comparisons.**

[API reference](https://omnibias.ai/api/struct/) · [Source](https://github.com/derivon-ai/omnibias/tree/main/packages/omnibias-struct/src/omnibias/struct) · [Tests](https://github.com/derivon-ai/omnibias/tree/main/packages/omnibias-struct/tests) · [Talk to Derivon](mailto:info@derivon.ai)

![Dynamic programs that participate in training.](https://raw.githubusercontent.com/derivon-ai/omnibias/62c9646964ea58b00cc3de5b80bcc2134b45f511/docs/img/explain/temperature-collapse.svg)

<details>
<summary>Watch the mechanism change continuously</summary>

![Animated mathematical explanation](https://raw.githubusercontent.com/derivon-ai/omnibias/62c9646964ea58b00cc3de5b80bcc2134b45f511/docs/img/explain/temperature-collapse.gif)

The animation illustrates the mechanism; it is not a speed benchmark.
</details>

## Why this package exists

A max-score path selects one discrete explanation. Replacing max with a temperature-scaled log-sum-exp retains information about alternatives, making the structured value differentiable with respect to scores. Struct combines these recurrences with shared semiring representations and derivative machinery.

## What you can build

- Soft Viterbi, alignment, DTW, planning and related structured operators.
- Semiring/hypergraph representations and selected parsing families.
- Marginals, higher derivatives, decoding and soft-versus-hard gap bounds.

Use struct when a neural model produces scores for a sequence or structured decision and the final loss should train those scores. The recurrence is the reusable primitive; task-specific tokenization, datasets, supervision and application objectives belong in the consuming project.

## Install the source edition

This README describes the current source tree. Published artifacts can lag this
branch, and this migration does not overwrite existing package versions. Build
the coordinated local wheelhouse using the [release guide](https://github.com/derivon-ai/omnibias/blob/main/RELEASE.md),
then install only this package and its selected dependencies:

```bash
python -m pip install --constraint artifacts/wheel-validation/constraints.txt "omnibias-struct[torch]"
```

Run that command from the repository containing the generated wheelhouse.
The constraint file selects the built distributions rather than silently mixing
an older published dependency with the current source. Supported Python versions,
optional features and runtime dependencies are declared in [pyproject.toml](https://github.com/derivon-ai/omnibias/blob/main/packages/omnibias-struct/pyproject.toml).

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

Examples above are executable smoke checks, not a substitute for application
validation. For a production integration, measure the intended objective,
precision, parameter gradients, memory and wall time on representative inputs.

## License

AGPL-3.0-or-later **or commercial**. See [LICENSE](https://github.com/derivon-ai/omnibias/blob/main/packages/omnibias-struct/LICENSE) and the [licensing policy](https://github.com/derivon-ai/omnibias/blob/main/LICENSING.md).
Comply with the AGPL terms or obtain a signed commercial grant; commercial use alone does not require payment. [Commercial terms](https://github.com/derivon-ai/omnibias/blob/main/packages/omnibias-struct/COMMERCIAL-LICENSE.md) describe the alternative. Previously distributed Apache editions, where applicable, retain their original grants.
