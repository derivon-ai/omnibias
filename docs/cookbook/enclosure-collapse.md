# Enclosure Collapse cookbook

Enclosure Collapse is the `width -> 0` limit of a **sound enclosure**.
The closed-form object is the Width Law. The operation is squeeze.
The output is a point plus a proof, or `Inconclusive`. This is not
bias collapse (`delta -> 0`) and not temperature collapse
(`beta -> inf`).

## Width Law (mean-value)

```python
from omnibias.core.verified.enclosure_collapse import (
    width_law,
    predicted_width,
    measured_mean_value_width,
    rounding_floor_witness,
)

law = width_law("sigmoid", 0, 0.5, kind="mean_value")
enc = measured_mean_value_width("sigmoid", 0, 0.5, 1e-4)
pred = predicted_width(law, 1e-4)
assert pred.lo >= 0.0
assert enc.width > 0.0
floor = rounding_floor_witness()
assert floor.lo < floor.hi
```

## Residual squeeze (constant Helmholtz residual)

`u = cos(w · x)` solves Helmholtz exactly. Positive width is wrapping,
and splits drive it down.

```python
import math
from omnibias.core.verified.pde_certificate import helmholtz
from omnibias.verify.enclosure_collapse import squeeze_residual

w1, w2 = 1.3, -0.7
k = math.hypot(w1, w2)
layers = [([[w1, w2]], [0.0], "cos")]
domain = [(0.0, 1.0), (0.0, 1.0)]
pde = helmholtz(2, k)
widths = []
for splits in (1, 4, 16, 64):
    report = squeeze_residual(layers, domain, pde, splits=splits)
    assert report.honesty["continuum_navier_stokes_claim"] is False
    widths.append(report.width)
assert widths[0] > widths[-1]
```

## Peak squeeze

```python
from omnibias.core.verified.interval import Interval
from omnibias.verify.enclosure_collapse import squeeze_peak
from omnibias.verify.localization import ScanResponse

report = squeeze_peak(
    ScanResponse.sech2_peak(-0.3, alpha=5.0),
    box=Interval(-0.40, -0.20),
)
assert report.status == "certified"
assert report.scope == "local_box"
```
