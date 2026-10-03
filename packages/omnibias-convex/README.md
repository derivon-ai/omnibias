# omnibias-convex

Differentiable LP/QP primitives: log-barrier solvers, implicit KKT gradients, warm starts, and verified optimality enclosures. Failed numerical or certification checks report failure rather than asserting a solution.

Install with `pip install omnibias-convex`; tensor backends are optional where
listed in [pyproject.toml](pyproject.toml).

Run package tests from the repository root:

```bash
python -m pytest packages/omnibias-convex/tests -q
```

See the [main guide](../../README.md) for architecture and supported workflows.
License: AGPL-3.0-or-later or commercial; see [LICENSE](LICENSE).
