# omnibias-torch

PyTorch activation derivatives, directional and mixed Taylor jets, and reusable PINN architectures. Generic residual optimizers and exact parameter-curvature helpers live in `omnibias.torch.optim`.

Install with `pip install omnibias-torch`; tensor backends are optional where
listed in [pyproject.toml](pyproject.toml).

Run package tests from the repository root:

```bash
python -m pytest packages/omnibias-torch/tests -q
```

See the [main guide](../../README.md) for architecture and supported workflows.
License: Apache-2.0; see [LICENSE](LICENSE).
