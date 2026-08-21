# Arrangement LP

The named pentagon `min -x-y` subject to `x<=2`, `y<=2`, `x+y<=3`,
`x>=0`, `y>=0` is degenerate on purpose: the optimum is the edge from
`(2,1)` to `(1,2)`, value `-3`. Soft membership at the midpoint is
differentiable. The dual `y=(0,0,1,0,0)` meets the primal (G5).

```python
import math

import numpy as np
from omnibias.convex.arrangement import (
    dual_objective,
    duality_holds,
    known_dual_pentagon,
    named_pentagon,
    soft_cell_gap_bound,
    soft_membership,
    vertex_optimum,
)

poly, c, opt = named_pentagon()
w = float(soft_membership(poly, np.array([1.5, 1.5]), beta=5.0))
assert abs(w - 0.4265470) < 2e-6
assert abs(soft_cell_gap_bound(n_facets=5, beta=5.0) - math.log(5.0) / 5.0) < 1e-12
_x, value, verts = vertex_optimum(poly, c)
assert abs(value - opt) < 1e-10
assert verts.shape[0] == 5
y = known_dual_pentagon()
assert duality_holds(poly, c, y)
assert abs(dual_objective(poly, y) - opt) < 1e-12
```

Selection of the feasible cell is temperature collapse, not founding
bias collapse. The lower bound is Neumaier-Shcherbina only.
