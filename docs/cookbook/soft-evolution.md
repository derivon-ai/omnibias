# Soft-population evolution

Selection pressure is a temperature. At `beta = 1` the soft mean of
`E = (2.0, 2.3, 3.1, 5.0)` sits `0.347` above the best member; the
certified gap is `log(4)/beta`. That is temperature collapse, not
founding bias collapse.

```python
import math

import numpy as np
from omnibias.discrete.evolution import (
    selection_gap_bound,
    selection_stats,
    soft_weights,
)

e = np.array([2.0, 2.3, 3.1, 5.0])
w, e_soft, e_best, gap = selection_stats(e, beta=1.0)
assert e_best == 2.0
assert abs(e_soft - 2.347435) < 5e-6
assert gap <= selection_gap_bound(population=4, beta=1.0)
assert abs(selection_gap_bound(population=4, beta=1.0) - math.log(4.0)) < 1e-12
assert abs(float(np.sum(soft_weights(e, beta=1.0))) - 1.0) < 1e-15
```

A certified discrete run stops when a sound lower bound meets the
best decoded energy. That is an instance-wise sandwich, not a
complexity claim.
