# Weak-form residual enclosure

Taylor-Green on a 16×16 cell mesh has an exact residual of
zero. The enclosure width is the method remainder. The weak
form drops the Gauss term, so the dominant (quadrature)
piece is half as wide.

```python
from omnibias.pinn.certified.weak_form import (
    DISCLAIMER,
    certified_weak_residual,
    width_decomposition,
)

strong = certified_weak_residual(form="strong")
weak = certified_weak_residual(form="weak")
ws = width_decomposition(strong)
ww = width_decomposition(weak)
assert ws.dominant == "quad"
assert ww.dominant == "quad"
assert ws.w_quad >= 2.0 * ww.w_quad
assert weak["honesty"]["continuum_navier_stokes_claim"] is False
assert "not a continuum" in DISCLAIMER
```

A residual that is orthogonal to constants is still not
zero. The completeness remainder stays in the enclosure.

```python
from omnibias.pinn.certified.weak_form import manufactured_residual

assert abs(manufactured_residual(0.25, 0.25)) > 0.1
```
