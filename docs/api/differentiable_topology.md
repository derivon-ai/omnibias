# Differentiable topology (03-09)

Cell counts and Euler characteristics are integers, so **no
differentiable function equals a Betti number**. Soft cell masses
and 1-D Morse persistence values are surrogates with a stated
gap to the integer truth. Status is **shipped**.
G1–G6 are CI-gated.

`beta -> inf` is **temperature collapse** (feasibility). The
**founding bias collapse** (`delta -> 0`) appears only in the
underlying field. Do not conflate the two.

Persistence is differentiable **almost everywhere** (pair swaps).
The Euler gap is a **sum over faces** and is loose for large
arrangements. Certified integer counts require a separating
spectral enclosure; otherwise the answer is `Inconclusive`.
Not a continuum topology theorem and not a persistent-homology
library.

Home: `omnibias.shape.topology`. Reuses
`count_eigenvalues_below`. Face masses may come from
`partition_weights`. No new package.

Spec 05-02 G6/G7 compose this module: `field_euler_characteristic`
and `field_euler_pair` return the soft Euler **and** its gap bound
together (`SoftCount` / a 2-tuple; never a bare float).
`regularize_occupancy` is a temperature-smoothed Euler prior, not
an integer Betti number. Smoke:
`docs/benchmarks/shape_topology_smoke.json`.

```python
import numpy as np
from omnibias.shape.topology import field_euler_pair

xs = np.linspace(-1.0, 1.0, 15)
value, bound = field_euler_pair(
    lambda x, y: 0.36 - x * x - y * y, beta=12.0, grid=xs
)
assert bound >= 0.0
assert isinstance(value, float)
```

## API

::: omnibias.shape.topology
    options:
      show_root_heading: false
      heading_level: 3
