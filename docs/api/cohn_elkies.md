# Cohn–Elkies 1-D Hermite bound

`omnibias.core.verified.cohn_elkies` certifies a **fixed-dimension** (here: 1-D)
Cohn–Elkies packing-density upper bound from a finite even Hermite combination
with an exact Fourier transform. Sign constraints are checked on a declared
sample grid. Compared to the published 1-D packing density `1`. Not `d -> inf`.

```python
from omnibias.core.verified.cohn_elkies import (
    PUBLISHED_1D_PACKING_DENSITY,
    cohn_elkies_hermite_bound,
)

result = cohn_elkies_hermite_bound()
assert result.certified
assert result.dimension == 1
assert result.density_upper >= PUBLISHED_1D_PACKING_DENSITY
assert result.high_dimension_limit_claim is False
```

## API

::: omnibias.core.verified.cohn_elkies
    options:
      show_root_heading: false
      heading_level: 3
