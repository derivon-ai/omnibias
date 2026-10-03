# omnibias-discrete

Shared discrete-optimization primitives: `DiscreteProblem`, annealing schedules, differentiable relaxations, rounding, local decoding, matroid kernels, and certified optimality gaps. Application encoders live in consumer projects.

Install with `pip install omnibias-discrete`; tensor backends are optional where
listed in [pyproject.toml](pyproject.toml).

Run package tests from the repository root:

```bash
python -m pytest packages/omnibias-discrete/tests -q
```

See the [main guide](../../README.md) for architecture and supported workflows.
License: AGPL-3.0-or-later or commercial; see [LICENSE](LICENSE).
