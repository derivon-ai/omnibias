# Finite-lattice ground state

`omnibias.core.verified.lattice_ground_state` sandwiches the lowest eigenvalue
of one finite Hermitian matrix (Ritz upper + Temple lower + blocked `LDL^T`
inertia). The default slice is the **two-site open Hubbard chain** at
half filling `(N_up, N_down) = (1, 1)`. A numerical `eigh` is an oracle / trial
vector only. Not a thermodynamic-limit claim.

```python
from omnibias.core.verified.lattice_ground_state import (
    hubbard_half_filled_ground_state,
    two_site_hubbard_exact_energy,
)

exact = two_site_hubbard_exact_energy(hopping=1.0, u=4.0)
cert = hubbard_half_filled_ground_state(2, hopping=1.0, u=4.0, tolerance=0.25)
assert cert.certified
assert cert.sector == (1, 1)
assert cert.lower <= exact <= cert.upper
assert cert.continuum_claim is False
```

## API

::: omnibias.core.verified.lattice_ground_state
    options:
      show_root_heading: false
      heading_level: 3
