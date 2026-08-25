# Inverse imaging

A piecewise-linear kink at `tau* = 0.37` is a jump of size 2 in `u'`.
Channel `n = 3` localizes it. The enclosure is a Krawczyk box of the
noiseless response; it must not be added to a conformal slab.

```python
from omnibias.core.verified.interval import Interval
from omnibias.pinn.inverse import (
    GuaranteeKindError,
    honesty_payload,
    merge_guarantees,
    worked_example,
)

ex = worked_example()
assert abs(ex["tau_hat"] - 0.37) < 1e-3
assert ex["unique"] is True
assert ex["contains_truth"] is True
assert honesty_payload()["ill_posedness_removed"] is False
assert honesty_payload()["temperature_collapse"] is False
raised = False
try:
    merge_guarantees(Interval(0.36, 0.38), {"kind": "conformal"})
except GuaranteeKindError:
    raised = True
assert raised
```
