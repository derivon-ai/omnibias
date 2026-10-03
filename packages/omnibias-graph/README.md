# omnibias-graph

Differentiable graph Laplacians, spectral embeddings, heat kernels, and soft assignment primitives such as Sinkhorn, sorting, and top-k relaxations.

Install with `pip install omnibias-graph`; tensor backends are optional where
listed in [pyproject.toml](pyproject.toml).

Run package tests from the repository root:

```bash
python -m pytest packages/omnibias-graph/tests -q
```

See the [main guide](../../README.md) for architecture and supported workflows.
License: Apache-2.0; see [LICENSE](LICENSE).
