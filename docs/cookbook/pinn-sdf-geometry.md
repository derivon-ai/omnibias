# SDF geometry with hard Dirichlet cages

A negative-inside SDF vanishes on the boundary, so the multiplicative cage
``u = g + phi * NN`` makes Dirichlet data exact by construction. Interior
accuracy is still optimised; boundary identity is not.

Measured gates (disk skill, boundary max gap, nonconvex boundary identity)
are in
[`docs/benchmarks/pinn_four_gap_matrix.md`](../benchmarks/pinn_four_gap_matrix.md).

```python
import torch
from omnibias.pinn import ComponentSpec, CoordinateSpec
from omnibias.pinn.domain import Sphere, evaluate_sdf
from omnibias.pinn.domain.torch import DistanceConstrainedField
from omnibias.pinn.torch import ops as tops
from omnibias.pinn.torch.fields import OneLayerVectorField

torch.set_default_dtype(torch.float64)
cs = CoordinateSpec(("x", "y"), domain=((-1.5, 1.5), (-1.5, 1.5)))
base = OneLayerVectorField(
    coordinate_spec=cs,
    components=ComponentSpec(("u",)),
    hidden=16,
    base="tanh",
)
disk = Sphere(center=(0.0, 0.0), radius=1.0)
field = DistanceConstrainedField(base=base, sdf=disk)

# Boundary identity: every point with phi ~= 0 must yield u = 0.
boundary = torch.tensor(
    [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]],
    dtype=torch.float64,
)
assert float(evaluate_sdf(disk, boundary.numpy()).max()) < 1e-12
u_b = tops.value(field(boundary), "u")
assert float(u_b.abs().max()) < 1e-12
```

Interior residual training uses the same cage: sample inside ``phi < 0``,
penalise ``Delta u - f``, and keep the boundary free of soft weights. At CSG
junctions where normals are undefined the domain layer raises
``NonSmoothBoundaryError`` rather than inventing a normal -- that is the
honest structural limit, not a soft failure.
