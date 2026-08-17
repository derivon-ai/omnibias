# Parity report: recovered symbolic / equation-discovery suite

This table records the previously-reported behavior (from the original research
runs, now encoded as the test thresholds) against the numbers reproduced after
recovery on this machine (CPU, `float64` where applicable; numpy 2.3, jax 0.10,
torch 2.9, sklearn 1.6). "Reproduced" values come from re-running the engine and
benchmarks; every row is at least as good as the prior result.

## omnibias.symbolic (NeuralJetDiscoverer + AutoML + PDE + Blasius)

| Task | Prior (target) | Reproduced |
| --- | --- | --- |
| exp identity (library-free) | `dy = y`, test RMSE < 1e-8 | `dy = 1*y`, 3.65e-13 |
| sin identity (library-free) | `d2y = -y`, test RMSE < 1e-8 | `d2y = -1*y`, 4.62e-13 |
| tanh Riccati identity | `dy = 1 - y^2`, test RMSE < 1e-8 | `dy = 1 - 1*y^2`, 1.05e-13 |
| AutoML surrogate (noise 0.02) | recovers `x1^2, x2*x3, sin(2*x4), cos(x4)` | all four recovered; family `taylor_fourier`; test RMSE 1.87e-2 |
| PDE operator (heat) | `u_t = 0.12*u_xx`, coeff err < 1e-4, RMSE < 1e-8 | `u_t = 0.12*u_xx`, RMSE 2.87e-14 |
| High-dim sparse (50 dims) | recovery rate 1.0, RMSE < 0.08 | rate 1.0 (`x3^2, x17*x42, sin(x8), x5`), RMSE 0.020 |
| Blasius shooting | `f''(0) ~= 0.332057` | 0.332059 |
| Blasius identity | `d3f = -0.5*f*d2f`, test RMSE < 1e-10 | `d3f = -0.5*f*d2f`, 9.08e-17 |
| Diabetes tabular | interpretable surrogate, RMSE < 80 | passes (see test) |

Validated by `packages/omnibias-symbolic/tests/test_symbolic_discovery.py`
(21 tests).

## examples/symbolic_discovery (reproducible synthetic paths)

| Experiment | Prior (target) | Reproduced |
| --- | --- | --- |
| synthetic_feature_discovery | discover `x1^2, x2*x3, sin(x4)`; beat raw baseline | discovered exactly those 3; `omnibias_discovered_linear` RMSE 0.102 vs `raw_linear` 4.51 (noise floor 0.084) |
| battery_law_discovery (synthetic) | recover decaying capacity law, correct sign | `dq/dn` fit RMSE 3.2e-7, `coef[q] < 0` |
| joint_operator_regressor (synthetic) | rank `x1^2, x2*x3, sin(x4)`; near dictionary | top operators `x2*x3, x1^2, sin(x4)`; RMSE 0.083 (= noise floor 0.08) vs raw ridge 4.51 |
| cmapss_feature_discovery | structural feature pipeline | tests pass on synthetic FD table |
| financial_signal_discovery | FI-2010 pipeline | tests pass on synthetic FI-2010 zip |
| public_csv_discovery | recover LV `xy` signs; interpolant ≤ 1.25× FD; rollout beats linear; public rollout gates | synthetic signs + auto interpolant + Hudson Bay smoke JSON |

Validated by `examples/symbolic_discovery/**/tests` (49 tests). Real-data
reproductions (Severson battery, NASA C-MAPSS, FI-2010) require downloading the
datasets; see `README.md`.
