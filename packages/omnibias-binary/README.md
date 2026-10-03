# omnibias-binary

Binary, ternary, and k-bit quantization with differentiable activation-tower surrogates and annealing schedules. Hard forward values and surrogate backward derivatives are distinct operations.

Install with `pip install omnibias-binary`; tensor backends are optional where
listed in [pyproject.toml](pyproject.toml).

Run package tests from the repository root:

```bash
python -m pytest packages/omnibias-binary/tests -q
```

See the [main guide](../../README.md) for architecture and supported workflows.
License: Apache-2.0; see [LICENSE](LICENSE).
