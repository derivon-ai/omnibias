# Cone-field hyperbolicity

`omnibias.dynamics._core.cone` certifies that a **finite orbit segment** of
interval Jacobians expands a cone with uniform factor `eta > 1`, and returns
the topological-entropy lower bound `log(eta)`. A rotation (harmonic-oscillator
step) is refused. This is not an Anosov or continuum chaos claim.

```python
from omnibias.dynamics import (
    certified_cone_hyperbolicity,
    doubling_map_jacobian,
    rotation_jacobian,
)

ok = certified_cone_hyperbolicity([doubling_map_jacobian()], [[1.0]], eta=1.5)
assert ok.certified
assert ok.entropy_lower > 0.0
assert ok.continuum_claim is False

refused = certified_cone_hyperbolicity([rotation_jacobian()], [[1.0, 0.0]], eta=1.1)
assert refused.certified is False
```

## API

::: omnibias.dynamics._core.cone
    options:
      show_root_heading: false
      heading_level: 3
