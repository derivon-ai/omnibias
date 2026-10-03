# omnibias-fields

Backend-neutral `FieldState`, coordinate/component schemas, lazy derivative caching, and reusable PyTorch/JAX field calculus. Includes gradients, divergence, curl, Laplacians, Hessians, tensor operators, integration, norms, weak forms, and an extension registry.

Install with `pip install omnibias-fields`; tensor backends are optional where
listed in [pyproject.toml](pyproject.toml).

Run package tests from the repository root:

```bash
python -m pytest packages/omnibias-fields/tests -q
```

See the [main guide](../../README.md) for architecture and supported workflows.
License: Apache-2.0; see [LICENSE](LICENSE).
