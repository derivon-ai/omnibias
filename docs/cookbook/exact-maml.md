# Exact MAML

One inner Newton step on `L = 0.5 (theta - alpha)^2` lands on
`alpha`. The IFT meta-gradient matches the closed-form `d theta'/d alpha = 1`.

```python
from omnibias.core.exact_maml import honesty_payload, quadratic_worked_example

ex = quadratic_worked_example()
assert ex["inner_err"] < 1e-12
assert ex["ift_err"] < 1e-10
assert honesty_payload()["imagenet_claim"] is False
assert honesty_payload()["stretch_claim"] is False
```
