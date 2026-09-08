# Weighted coefficient class (07-11)

Fixed-order seminorm bounds `W_α`, `M_α`, `S_α` plus an optional
finite-order Gevrey majorant `|a_k| ≤ C k!^s ρ^k` only for
`k ≤ k_max`. Infinite-class membership stays out of Lean. The
geometric `ValidatedSeries` path is unchanged.

Status is **shipped**. G1–G5 are CI-gated
(`benchmarks/weighted_class.py`). This is not a Gevrey-class theorem
and not a Navier-Stokes class-membership claim. See theory spec
[07-11](https://github.com/derivon-ai/omnibias/blob/main/theory/07-frontier/11-weighted-coefficient-class.md).

Home: `omnibias.core.verified.weighted_class`.

```python
from fractions import Fraction

from omnibias.core.verified.weighted_class import (
    check_pointwise_bound,
    gevrey_majorant,
    honesty_payload,
    locked_bound,
    locked_samples,
    violating_sample,
)

assert check_pointwise_bound(locked_samples(), locked_bound()).holds
failed = check_pointwise_bound((violating_sample(),), locked_bound())
assert failed.holds is False
assert "too_large" in failed.failing
assert gevrey_majorant(
    (Fraction(1), Fraction(1, 2), Fraction(1, 4)),
    s=0,
    rho=Fraction(1, 2),
    k_max=2,
).holds
assert honesty_payload()["gevrey_class_claim"] is False
```
