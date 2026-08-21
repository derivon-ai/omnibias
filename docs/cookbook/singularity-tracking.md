# Jet-Padé singularity tracking

A simple pole at `x = 0.3` is recovered from a geometric jet.
The result is a diagnostic, not a blow-up proof.

```python
from omnibias.difference.singularity import (
    DISCLAIMER,
    certified_singularity_annulus,
    domb_sykes,
    pade_estimate,
)
from omnibias.core.verified.interval import Interval

coeffs = tuple((10.0 / 3.0) ** k for k in range(5))
ds = domb_sykes(coeffs)
assert not ds.failed
assert ds.location is not None
assert abs(ds.location.real - 0.3) < 1e-8
assert abs((ds.exponent or 0.0) - 1.0) < 1e-8
pd = pade_estimate(coeffs, numer_deg=0, denom_deg=1)
assert pd.location is not None
assert abs(pd.location.real - 0.3) < 1e-8
enc = certified_singularity_annulus(
    [Interval.from_value(c) for c in coeffs],
    tail_bound=Interval.point(1.0),
    tail_ratio=10.0 / 3.0,
)
assert enc.lo <= 0.3 <= enc.hi
assert DISCLAIMER.startswith("diagnostic")
```

`exp` has an essential singularity at infinity, so Domb-Sykes
reports failure instead of a confident wrong pole.

```python
from math import factorial
from omnibias.difference.singularity import domb_sykes

ess = tuple(1.0 / factorial(k) for k in range(12))
got = domb_sykes(ess)
assert got.failed
```
