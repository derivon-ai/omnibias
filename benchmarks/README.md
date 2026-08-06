# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon

# Public benchmarks

Every number the root [`README.md`](../README.md) quotes from this suite is
produced by one of the scripts below and committed as JSON under
[`docs/benchmarks/`](../docs/benchmarks/). Regenerate on any commodity CPU:

```bash
uv sync --all-extras --dev
# folx is optional for the Laplacian / polylaplacian scripts
uv pip install 'folx>=0.2'

uv run python benchmarks/laplacian_scaling.py
uv run python benchmarks/polylaplacian_order.py
uv run python benchmarks/derivative_order.py
uv run python benchmarks/optimizer_pinn.py
uv run python benchmarks/burgers_shock_conservation.py
uv run python benchmarks/hard_conditions_solver.py
```

Then rebuild the figures that plot those JSON files:

```bash
uv run python docs/img/generate_figures.py
```

## What each script measures

| Script | Artifact | Comparison |
|---|---|---|
| `laplacian_scaling.py` | `docs/benchmarks/laplacian_scaling.json` | omnibias / folx / `jax.hessian` / `torch.func.hessian` Laplacian cost vs input dimension `D` |
| `polylaplacian_order.py` | `docs/benchmarks/polylaplacian_order.json` | closed-form `Δᵏ` vs nested folx / nested dense Hessian; records OOM / timeout honestly |
| `derivative_order.py` | `docs/benchmarks/derivative_order.json` | `σ^(n)` closed-form vs nested autograd vs finite differences (cost + accuracy) |
| `optimizer_pinn.py` | `docs/benchmarks/optimizer_pinn.json` | 1-D Poisson PINN: Adam / L-BFGS vs Gauss–Newton / cubic GN / trust-region Newton-CG |
| `burgers_shock_conservation.py` | `docs/benchmarks/burgers_shock_conservation.json` | shock-capturing Burgers PINN, conservative flux-form cage vs non-conservative arm at identical architecture / budget / seed, swept over 6 viscosities x 5 seeds |
| `hard_conditions_solver.py` | `docs/benchmarks/hard_conditions_solver.json` | Poisson / heat / wave with boundary + initial conditions absorbed into the ansatz vs kept as loss terms, at identical architecture / budget / seed over 5 seeds; reports boundary violation and interior relative L2 |

All runs are **float64**, **CPU** (`JAX_PLATFORMS=cpu`). Each JSON carries
`generated_utc`, `hardware_class`, library versions, and the exact config.

## Hardware tiers

| Tier | Where numbers live | Reproducible from this repo? |
|---|---|---|
| **CPU (this suite)** | `docs/benchmarks/*.json` | **Yes** — run the scripts above |
| **Data-center GPU (off-band)** | tables in [`docs/complexity.md`](../docs/complexity.md) labelled *off-band* | Measured on a separate GPU host; scripts not required for the CPU claims |

The README keeps these tiers visually separate. Do not paste an off-band GPU
number into a "reproduce this" block.

## Dependencies

- Required: `omnibias-core`, `omnibias-torch`, `omnibias-jax`, `numpy`, `torch`, `jax`.
- Optional: `folx` (Laplacian and polylaplacian scripts). Without it those two
  scripts exit with an import error — install it to regenerate those artifacts.
