# omnibias-jax

JAX activation derivatives, directional and mixed Taylor jets, and reusable PINN architectures. Kernels support JIT and differentiation through model parameters. Enable `JAX_ENABLE_X64=1` for double-precision comparisons.

Install with `pip install omnibias-jax`; tensor backends are optional where
listed in [pyproject.toml](pyproject.toml).

Run package tests from the repository root:

```bash
python -m pytest packages/omnibias-jax/tests -q
```

See the [main guide](../../README.md) for architecture and supported workflows.
License: Apache-2.0; see [LICENSE](LICENSE).
