# Sliced optimal transport

Two logistic bumps at `±1` versus one at `0` have an exact
`W_1` from three `softplus` evaluations. That is founding bias
collapse, not temperature collapse. Exact per slice, not sample-free.

```python
from omnibias.measure.transport import named_worked_pair, w1_exact, worked_w1

mu, nu = named_worked_pair()
w1 = w1_exact(mu, nu)
assert abs(w1 - worked_w1()) < 1e-12
assert abs(w1 - 0.240229013916555) < 1e-12
assert abs(w1_exact(mu, nu) - w1_exact(nu, mu)) < 1e-15
assert w1_exact(mu, mu) < 1e-15
```

Sliced Wasserstein averages those exact 1-D distances over directions
and always reports `direction_stderr`.
