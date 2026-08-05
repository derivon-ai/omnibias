---
name: omnibias-symbolic
description: Use omnibias to discover equations from data -- fit neural fields and read off exact jets, recover ODE / activation identities, and discover PDEs (heat / wave / Burgers / Laplace). Use when using omnibias for equation discovery, symbolic regression on a fitted field, or PDE identification, or when the user mentions neural-jet discovery, SINDy-style discovery, or field-law discovery.
---

# Using omnibias: neural-jet equation discovery

`omnibias.symbolic` fits a smooth neural field, reads its **exact closed-form
jet**, and recovers the governing ODE / PDE. Everything is re-exported from the
package root, so `from omnibias.symbolic import <name>` works for the names below.

Read the dense cheat-sheet first:
[docs/handbook/ai-quickstart.md](https://github.com/derivon-ai/omnibias/blob/main/docs/handbook/ai-quickstart.md).

## Import map (all from `omnibias.symbolic`)

| You want | Key entry points |
| --- | --- |
| Discover a 1-D ODE / activation identity | `discover_activation_identity`, `NeuralJetDiscoverer` |
| Fit a smooth field + its jet | `fit_neural_field_1d`, `fit_neural_field_nd`, `extract_field_jet` |
| Gradient / Hessian / Laplacian of a fitted field | `field_gradient`, `field_laplacian` |
| Discover a PDE (heat / wave / Burgers / Laplace) | `FieldLawDiscoverer`, `discover_field_pde_law`, `make_heat_field_split` |
| Curvature / metric on learned charts | `MetricField`, `laplace_beltrami`, `pullback_metric_field`, `scalar_curvature` |

## Minimal example (verified)

```python
from omnibias.symbolic import discover_activation_identity

# `exp` secretly obeys y' = y; recovered from the closed-form jet.
result = discover_activation_identity("exp", candidate_lhs_orders=(1,))
print(result.formula())            # -> "dy = 1*y"
```

## Gotchas that bite

- **A random-feature field is only accurate inside the support of its training points.** Use `n_features` in the hundreds for smooth targets and keep evaluation points inside the training box; extrapolation breaks the (exact-for-the-fitted-field) derivatives.
- **Request the order you need.** A jet carries partials up to its `max_order`; a Hessian needs `order=2`.
- **Pass `random_state=<int>`** to every `fit_*` / `make_*` / discoverer for bit-reproducible fields, jets, and discovered equations on a given platform.

## More detail

- API: [symbolic](https://github.com/derivon-ai/omnibias/blob/main/docs/api/symbolic.md)
- Handbook chapters 1-7 (neural-jet discovery through information geometry): [handbook index](https://github.com/derivon-ai/omnibias/blob/main/docs/handbook/index.md)
