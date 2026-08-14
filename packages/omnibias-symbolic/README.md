# omnibias-symbolic

**Status: Alpha (0.1.0).** Neural-jet equation discovery and interpretable
surrogate modeling, built on omnibias's closed-form activation derivative towers.

Where most symbolic-regression / SINDy tooling needs a hand-picked dictionary of
named basis functions (`sin`, `exp`, `tanh`, ...), `omnibias-symbolic` exploits the
fact that every omnibias activation exposes an *exact* n-th derivative fastpath.
That lets it build the full derivative jet `y, dy, d2y, ...` of a fitted field in
closed form and then search for compact *implicit* relations among those generic
jet coordinates — recovering the governing identity without ever being told which
named function generated the data.

## Highlights

- **`NeuralJetDiscoverer`** — library-free differential-identity discovery. From
  exact jets it recovers `dy = y` (exp), `d2y = -y` (sin/cos), and the Riccati
  relation `dy = 1 - y^2` (tanh) to machine precision, using only generic
  `x, y, dy, d2y` columns.
- **AutoML surrogates** (`discover_interpretable_surrogate`) — selects among
  Taylor / Fourier / hybrid libraries with train/validation/test fairness splits;
  recovers `1.5 x1^2 - 2 x2 x3 + sin(2 x4) + 0.4 cos(x4)`.
- **PDE operator recovery** (`discover_pde_operator_law`) — recovers the
  heat-equation coefficient on `u_xx`.
- **Fractional order discovery** (`discover_fractional_order_law`) — searches a
  grid of candidate orders and recovers the *fractional* order `alpha` of a
  fractional differential law, building closed-form `D^alpha` columns
  (`build_jet_fractional_features_closed_form`, the analytic-class twin of the
  Grünwald–Letnikov `build_jet_fractional_features`) from the jet tower and
  selecting by STLSQ; exact on polynomial signals.
- **Piecewise / hybrid-automaton discovery** (`omnibias.symbolic.piecewise`) —
  per-region STLSQ on a partition of unity; `fit_learned_piecewise_ode` learns
  the gates (differentiable soft-weighted residual) then hardens + polishes.
  A SoftTree or Arrangement trained on the trajectory's finite-difference
  ``du`` can be hardened from the **fitted** split as the partition (distinct
  from the learned-gate path). The Arrangement constructor is **unplanted**
  (random ``W``, no ``e_0`` axis init); STLSQ still uses the field jet. Vector
  systems share gates; the learned vector hybrid does not take an oracle
  partition. Oracle partitions remain the control.
- **High-dimensional sparse recovery** in many irrelevant dimensions.
- **Blasius boundary layer** (`omnibias.symbolic.blasius`) — shooting solve,
  recovery of `f''' = -0.5 f f''` from numerical and *blind neural* jets, and
  explicit rational / ODE-recurrence Pade surrogates with analytic derivatives.
- **Real-world tabular** interpretable surrogates on bundled sklearn datasets.

## Install

```bash
pip install omnibias-symbolic[all]   # numpy + jax (jets) + scikit-learn (tabular)
```

The sparse-regression, AutoML, PDE, and analytic-Blasius paths need only numpy.
The neural-field / jet-extraction paths additionally use `omnibias-jax` for the
closed-form activation fastpaths; the tabular validation uses `scikit-learn`.

## Quickstart

```python
from omnibias.symbolic import discover_activation_identity

# Recover d2y = -y from sin, with no named "sin" in the library.
result = discover_activation_identity("sin", x_range=(-3.14159, 3.14159),
                                      candidate_lhs_orders=(2,))
print(result.formula())          # d2y = -1*y
print(result.test_rmse)          # < 1e-8
```

See the example notebooks (`notebooks/13_*`–`notebooks/18_*`) for worked,
visual walkthroughs, and `docs/roadmap.md` for how this package fits the stack.

## License

Apache-2.0. See [`LICENSE`](LICENSE) and [`../../LICENSING.md`](../../LICENSING.md).
You never need a commercial licence for this package.
