# omnibias-curvature

Closed-form parameter Hessians, Gauss–Newton and Fisher matrices, KFAC factors, and regularized or natural-gradient steps. Reuses the shared activation derivative tower.

Install with `pip install omnibias-curvature`; tensor backends are optional where
listed in [pyproject.toml](pyproject.toml).

Run package tests from the repository root:

```bash
python -m pytest packages/omnibias-curvature/tests -q
```

See the [main guide](../../README.md) for architecture and supported workflows.
License: Apache-2.0; see [LICENSE](LICENSE).
