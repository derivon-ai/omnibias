# Combinatorial SOS degree certificates

`omnibias.sos.combinatorial` attaches a degree-indexed SOS residual to a
**planted-clique graph** or a **tiny 3-XOR instance**, with a brute-force
oracle. Distinct from `SosDegreeFamily`'s two-variable planted polynomial.
Not P vs NP.

```python
from omnibias.sos.combinatorial import (
    brute_force_max_clique,
    planted_clique_adjacency,
    planted_clique_degree_certificate,
)

adj = planted_clique_adjacency(4, 3)
assert brute_force_max_clique(adj) == 3
cert = planted_clique_degree_certificate(n=4, clique_size=3, half_degree=1)
assert cert.oracle_omega == 3
assert cert.sos_proved
```

## API

::: omnibias.sos.combinatorial
    options:
      show_root_heading: false
      heading_level: 3
