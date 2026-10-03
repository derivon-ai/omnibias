# omnibias-difference

Finite-difference stencils, certified derivative extraction, exact irregular stencils, and umbral sequence algebra. The finite-difference estimate is numerical; its limit uses the shared closed-form activation derivative tower.

Install with `pip install omnibias-difference`; tensor backends are optional where
listed in [pyproject.toml](pyproject.toml).

Run package tests from the repository root:

```bash
python -m pytest packages/omnibias-difference/tests -q
```

See the [main guide](../../README.md) for architecture and supported workflows.
License: Apache-2.0; see [LICENSE](LICENSE).
