# omnibias-sos

Polynomial sum-of-squares and Positivstellensatz certificate backends. Numerical solvers propose Gram matrices; outward-rounded interval arithmetic checks positivity. Optional Lean verification checks finite rational obligations.

Install with `pip install omnibias-sos`; tensor backends are optional where
listed in [pyproject.toml](pyproject.toml).

Run package tests from the repository root:

```bash
python -m pytest packages/omnibias-sos/tests -q
```

See the [main guide](../../README.md) for architecture and supported workflows.
License: AGPL-3.0-or-later or commercial; see [LICENSE](LICENSE).
