# Contributing

Develop reusable primitives here; build solver products and application
experiments in sibling repositories under `../omnibias_projects/`.
The package map is in [docs/packages.md](docs/packages.md).

```bash
uv sync --all-packages --group docs
uv run pytest packages/omnibias-core/tests -q
uv run ruff check packages tests
uv run mypy --strict packages/omnibias-core/src packages/omnibias-torch/src packages/omnibias-jax/src
uv run pytest tests/test_docs_snippets.py -q
uv run mkdocs build --strict
```

Run the tests for every changed package and any affected cross-backend parity
tests. For Keras, choose `KERAS_BACKEND` before importing it. JAX 64-bit parity
requires `JAX_ENABLE_X64=1` before arrays are created.

A contribution should include a focused implementation, a regression test for
each behavior change, and executable documentation for a new public API.
Use shared core polynomial coefficients, preserve parameter gradients, and
state numerical and approximation limits. Do not change versions unless the
release task asks for it.

[AGENTS.md](AGENTS.md) records the numerical and repository contracts.
Review [LICENSING.md](LICENSING.md) before adding cross-package dependencies.

Contributions require acceptance of the existing [CLA](CLA.md); the PR bot
records the signature.
