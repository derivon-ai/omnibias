# Pack Fisher metric

The two-bias logistic pack has Fisher component
`G_{delta,delta} ~ delta^2 / 720` as `delta -> 0`. That is the
founding bias collapse, not temperature collapse. Near collapse the
spread chart is unidentifiable; the discrete order is the
well-conditioned coordinate.

```python
from omnibias.curvature.information import (
    NotADensityError,
    finite_difference_family,
    fisher_metric,
    honesty_payload,
    worked_example,
)

ex = worked_example()
assert abs(ex["g_over_delta2"] - 1.0 / 720.0) / (1.0 / 720.0) < 1e-6
assert ex["recommendation"] == "reparameterize by order"
assert abs(ex["unit_location_distance"] - 1.0) < 1e-12
raised = False
try:
    fisher_metric(finite_difference_family(n_biases=3), [0.2])
except NotADensityError:
    raised = True
assert raised
assert honesty_payload()["k_ge_3_fisher"] == "inapplicable_not_a_density"
assert honesty_payload()["temperature_collapse"] is False
```
