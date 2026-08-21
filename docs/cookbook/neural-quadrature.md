# Neural quadrature

Two-point Gauss–Legendre on `[-1, 1]` is recovered from the moment
system. On `x^4` the Peano bound is attained.

```python
from omnibias.core.cubature import (
    MomentSystem,
    apply_rule,
    certified_error,
    solve_rule,
)
from omnibias.core.verified.interval import Interval

rule = solve_rule(MomentSystem.lebesgue(6), nodes=2, free_nodes=True)
assert abs(rule.nodes[0] + 3.0**0.5 / 3.0) < 1e-12 or abs(rule.nodes[0] - 3.0**0.5 / 3.0) < 1e-12
got = apply_rule(rule, lambda x: x**4)
assert abs(got - 2.0 / 9.0) < 1e-12
enc = certified_error(rule, deriv_bound=Interval.point(24.0), degree=3)
assert enc.contains(got - 0.4)
try:
    certified_error(rule, deriv_bound=None, degree=3)
except ValueError as exc:
    assert "refuses" in str(exc)
```

Pack functionals cancel the `O(scale^2)` bump moment instead of treating
smoothing as an error.
