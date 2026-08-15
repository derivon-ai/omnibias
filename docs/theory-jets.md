# Jet-bundle vocabulary (01-10)

This page is a **dictionary**, not a discovery. It names what omnibias
already computes. There is no `omnibias-jetbundle` package.

Status is **gated** (the two contact tests). Founding `delta -> 0`
produces fiber coordinates. Temperature collapse (`beta -> inf`) acts
on the base stratification, not on the fiber.

## Dictionary

| Program object | Jet-bundle statement |
|---|---|
| collapse of a `K`-pack | evaluation of the `(K-1)` fiber coordinate of `j^N sigma` along `w` |
| multi-pack (01-01) | a linear functional on a scattered jet |
| finite gap (`band`, `integral`) | a **fiber interval**: difference of the order `-1` coordinate at two base points. Band and integral are **not** local jet coordinates |
| bias scan (01-02) | the section pulled back along a translation of the base in the `w` direction |
| arrangement (01-03) | a **stratification** of the base by sign data of affine functions |
| equality locus (01-09) | the **fiber product** of two sections, a subvariety cut out by a jet condition |
| PDE | a subvariety of `J^N(E)`; a solution is a section whose prolongation lies inside it |
| Lie symmetry (03-11) | a vector field on `E` whose prolongation is tangent to that subvariety (stays designed) |

A tower of numbers is not automatically a jet. If derivative estimates
are assembled from mixed sources, they may violate the contact ideal.

## Contact test

```python
import math
from omnibias.core.jets import contact_residual, is_holonomic
from omnibias.core.polynomials import tanh_polynomial_coeffs


def _horner(coeffs, x):
    acc = 0.0
    for c in reversed(coeffs):
        acc = acc * x + c
    return acc


def tanh_jet(x, order=2):
    t = math.tanh(x)
    out = [t]
    for n in range(1, order + 1):
        out.append(_horner(tanh_polynomial_coeffs(n), t))
    return tuple(out)

x, h = 0.5, 1e-4
res = contact_residual(tanh_jet(x), tanh_jet(x + h), h=h)
assert abs(res[0]) < 1e-8
assert is_holonomic(tanh_jet, x, h=h)
```

`is_holonomic` is a rate test: genuine prolongations drop the residual
by about `4x` when `h` is halved; a corrupted tower drops by about `2x`.

See theory spec 01-10.

## Helpers

::: omnibias.core.jets
    options:
      show_root_heading: false
      heading_level: 3
