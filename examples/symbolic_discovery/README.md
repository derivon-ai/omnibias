# Symbolic / equation-discovery examples

Recovered research demos that showcase `omnibias.symbolic` (neural-jet equation
discovery) and `omnibias.torch` (the joint operator regressor) on controlled and
real-world scientific-discovery tasks.

These are the worked experiments behind the example notebooks
(`notebooks/13_*`–`notebooks/18_*`). The engine itself now lives in the tracked
`omnibias-symbolic` package; this directory holds the experiment drivers,
baselines, dataset loaders, and tests.

## Experiments

| Experiment | What it shows | Data | Reproducible offline |
| --- | --- | --- | --- |
| `synthetic_feature_discovery` | Sparse closed-form law recovery in many irrelevant dimensions | synthetic | yes |
| `battery_law_discovery` | Interpretable capacity-fade law from cycle data | synthetic + Severson (2019) | yes (synthetic path) |
| `cmapss_feature_discovery` | Functional/calculus features for turbofan RUL | NASA C-MAPSS | needs dataset |
| `financial_signal_discovery` | Calculus channels for FI-2010 mid-price movement | FI-2010 / DeepLOB | needs dataset (tests use a synthetic FI-2010 zip) |
| `joint_operator_regressor` | One-shot operator gating + readout (vs ridge / dictionary) | synthetic + C-MAPSS | yes (synthetic path) |
| `causal_term_discovery` | Directed parent ranking of terms (MI + NOTEARS-lite) on a known SEM | synthetic | yes |
| `dimensional_groups` | Buckingham-Pi dimensionless groups (Reynolds, pendulum) via exact integer null-space | none (closed form) | yes |
| `latent_ode_discovery` | Hidden oscillator law from one observed coordinate (Takens + autoencoder + FieldLawDiscoverer) | synthetic | yes |
| `public_csv_discovery` | Lotka–Volterra `xy` signs from a train-only cubic-spline interpolant + Huber/ridge STLSQ, scored by RK4 rollout (RF jet reported; FD / linear are named baselines) | Hudson Bay lynx–hare CSV (committed) + synthetic orbit | yes |

The fully-reproducible synthetic paths are exercised by the test suite and the
notebooks. The real-data paths require downloading the corresponding datasets.

## Datasets (real-data paths)

All caches default to a git-ignored top-level `data/` directory.

- Severson battery (2019): `python -m examples.symbolic_discovery.battery_law_discovery.download_severson`
  (downloads the MATR structured cells; large). Loader:
  `battery_law_discovery/severson_loader.py` (with a `make_synthetic_cycle_table`
  fallback).
- NASA C-MAPSS (FD001): pulled by `cmapss_feature_discovery/benchmark.py`
  (`ensure_dataset`) into `data/cmapss_fd001`.
- Hudson Bay lynx–hare (1900–1920): committed at
  `public_csv_discovery/data/lynx_hare.csv` with provenance. Offline.
- FI-2010 / DeepLOB: fetched by `financial_signal_discovery/benchmark.py`
  (`fetch_deeplob_fi2010_zip`) into `data/financial_signal_discovery`.

## Running

```bash
# Fully reproducible (no external data):
python -m examples.symbolic_discovery.synthetic_feature_discovery.run_demo
python -m examples.symbolic_discovery.battery_law_discovery.run_demo
python -m examples.symbolic_discovery.joint_operator_regressor.run_demo
python -m examples.symbolic_discovery.causal_term_discovery.run_demo
python -m examples.symbolic_discovery.dimensional_groups.run_demo
python -m examples.symbolic_discovery.latent_ode_discovery.run_demo
python -m examples.symbolic_discovery.public_csv_discovery.run_demo --quick

# Tests (run from the repo root so `examples.*` is importable):
python -m pytest examples/symbolic_discovery -q
```

Generated metrics/reports land under a git-ignored `results/` directory.
