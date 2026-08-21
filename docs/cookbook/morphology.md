# Differentiable morphology

A 3-tap flat dilation of `(0, 1, 3, 1, 0)` is hard `(1, 3, 3, 3, 1)`.
At `beta = 2` the center is `3.017988`, an excess of `0.018` inside
`log(3)/2`. That is temperature collapse, not founding bias collapse.

```python
import math

from omnibias.shape.morphology import (
    StructuringElement,
    dilate,
    hard_dilate,
    morphology_gap_bound,
    named_worked_signal,
    named_worked_soft_center,
)

f = named_worked_signal()
se = StructuringElement.flat(1)
hard = hard_dilate(f, se)
assert list(hard) == [1.0, 3.0, 3.0, 3.0, 1.0]
soft = named_worked_soft_center(beta=2.0)
assert abs(soft - 3.017988) < 5e-6
assert 0.0 <= soft - 3.0 <= morphology_gap_bound(size=3, beta=2.0)
assert abs(float(dilate(f, se, beta=2.0).value[2]) - soft) < 1e-12
assert math.isfinite(float(dilate(f, se, beta=50.0).value[2]))
```

An opening reports `compositions=2`. Soft max-pool is the same
homotopy on unfold patches, not a new operator role.
