# Constraint satisfaction by collapse

Colour a triangle with three colours. At the uniform point every edge
has satisfaction `2/3`, so `E = 1`. A proper colouring is a vertex
with `E = 0` exactly. Both temperature knobs are temperature
collapse, not founding bias collapse.

```python
import math

import numpy as np
from omnibias.discrete.csp import (
    assignment_onehot,
    certify_csp,
    clause_gap_bound,
    triangle_colouring,
)

csp = triangle_colouring(n_colours=3)
uni = csp.pack([np.full(3, 1.0 / 3.0) for _ in range(3)])
assert abs(float(csp.violation_energy(uni)) - 1.0) < 1e-12
p1 = np.array([0.8, 0.1, 0.1])
p2 = np.array([0.1, 0.8, 0.1])
p3 = np.full(3, 1.0 / 3.0)
e = float(csp.violation_energy(csp.pack([p1, p2, p3])))
assert abs(e - 0.8366666666666666) < 2e-6
rgb = assignment_onehot(csp, (0, 1, 2))
assert float(csp.energy(rgb)) == 0.0
cert = certify_csp(csp, (0, 1, 2), beta=20.0)
assert cert.claimed_sat is True
assert abs(clause_gap_bound(csp, beta=20.0) - 3.0 * math.log(6.0) / 20.0) < 1e-12
```

`certify_csp` claims SAT only when the decoded vertex has energy 0.
It never proves unsatisfiability.
