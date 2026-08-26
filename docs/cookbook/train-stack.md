# Recommended 08-01 trainer stack

Closed-form Newton plus a jet line search on a one-layer teacher /
student. The stack must beat a short gradient-descent budget and the
zero predictor. An indefinite Hessian is damped by `|λ_min|`.
Kantorovich (`use_kantorovich=True`) accepts a unique `φ'` ball near
the teacher and may later return `empty` — a halt, not a failure.
The bake-off below turns that filter off. Bias collapse (`delta -> 0`)
supplies the tower.

```python
from omnibias.core.train_stack import honesty_payload, worked_example

ex = worked_example()
assert ex["first_step_never_worse"] is True
assert ex["used_composed"] is False
assert ex["beat_start"] is True
assert ex["beat_gd"] is True
assert ex["skill_positive"] is True
payload = honesty_payload()
assert payload["skip_chain_rule"] is False
assert payload["full_parameter_jacobian"] is False
assert payload["global_min_claim"] is False
assert payload["stretch_claim"] is False
```
