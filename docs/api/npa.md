# NPA moment hierarchy

`omnibias.sos.npa` is a **submodule** (not a new distribution): noncommutative
words, a moment matrix from a linear functional, certified PSD via interval
`LDL^T`, one `Z_2` isotypic split, and a linear-Hamiltonian lower bound
`E_0 >= -sum |c_i|` for involutive generators. Lower bounds only; no full
diagonalization claim.

```python
from omnibias.sos.npa import (
    npa_linear_hamiltonian_bound,
    pauli_pair_generators,
)

gens = pauli_pair_generators()
bound = npa_linear_hamiltonian_bound(gens, (-1.0, -1.0), level=1)
assert bound.certified
assert bound.lower_bound == -2.0
assert bound.full_diagonalization_claim is False
```

## API

::: omnibias.sos.npa
    options:
      show_root_heading: false
      heading_level: 3
