# Volume-uniform strong-coupling family

`volume_uniform_strong_coupling_family` evaluates the existing polymer
glueball bound on a declared growing finite-lattice family of spacetime
dimensions. It does not rebuild the polymer majorant.
`yang_mills_claim` stays `False`.

```python
from omnibias.geometry.gauge.transfer.strong_coupling import (
    volume_uniform_strong_coupling_family,
)

family = volume_uniform_strong_coupling_family(0.1, spacetime_dims=(2, 3, 4))
assert family.yang_mills_claim is False
assert family.certified
assert family.min_gap > 0.0
```

## API

::: omnibias.geometry.gauge.transfer.strong_coupling
    options:
      show_root_heading: false
      heading_level: 3
      members:
        - VolumeUniformStrongCouplingFamily
        - volume_uniform_strong_coupling_family
