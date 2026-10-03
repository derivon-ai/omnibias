# omnibias-boolean

Exact Boolean algebra: truth tables, ANF coefficients, Walsh spectra, Boolean derivatives, equation solving, and optional differentiable gates. Exact routines use integer algebra; soft gates use PyTorch or JAX.

Install with `pip install omnibias-boolean`; tensor backends are optional where
listed in [pyproject.toml](pyproject.toml).

Run package tests from the repository root:

```bash
python -m pytest packages/omnibias-boolean/tests -q
```

See the [main guide](../../README.md) for architecture and supported workflows.
License: Apache-2.0; see [LICENSE](LICENSE).
