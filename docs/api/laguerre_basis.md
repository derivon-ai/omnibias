# Laguerre basis

`omnibias.core.verified.laguerre_basis` is the **Laguerre sibling** of the
physicists' Hermite basis: exact rational `L_n` coefficients, interval Horner
evaluation of `ell_n(x) = L_n(x) exp(-x/2)` for `x >= 0`, and a geometric
tail under a caller-supplied uniform bound. Laguerre functions are **not**
Fourier eigenfunctions (no exact `(-i)^n` self-duality).

```python
from fractions import Fraction
from omnibias.core.verified.laguerre_basis import (
    laguerre_function,
    laguerre_poly_coeffs_exact,
)

assert laguerre_poly_coeffs_exact(1) == (Fraction(1), Fraction(-1))
ell0 = laguerre_function(0, 0.0)
assert ell0.contains(1.0)
```

## API

::: omnibias.core.verified.laguerre_basis
    options:
      show_root_heading: false
      heading_level: 3
