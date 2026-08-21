# Differentiable topology

A three-cluster Laplacian has a soft component count near 3 at
`beta = 50`, and a certified inertia count of exactly 3 when the
gap separates.

```python
from omnibias.shape.topology import (
    Inconclusive,
    certified_component_count,
    cluster_laplacian,
    soft_component_count,
)

lap = cluster_laplacian(3, n_per=2, gap=0.205)
ev = (0.0, 0.0, 0.0, 0.41, 0.55, 0.72)
soft = soft_component_count(ev, epsilon=0.2, beta=50.0)
assert abs(soft.value - 3.0) < 2e-4
assert abs(soft.value - 3.0) <= soft.gap_bound
got = certified_component_count(lap, epsilon=0.2)
assert got == 3
```

A threshold on the spectrum is `Inconclusive`, never a guessed
integer.

```python
from omnibias.shape.topology import Inconclusive, certified_component_count, cluster_laplacian

path = cluster_laplacian(1, n_per=2, gap=0.5)
got = certified_component_count(path, epsilon=1.0)
assert isinstance(got, Inconclusive)
```
