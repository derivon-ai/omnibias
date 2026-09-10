# De Bruijn–Newman `H_t` and a named `Lambda` attempt

`omnibias.core.verified.debruijn_newman` encloses the genuine `Phi(u)` kernel
and `H_t(z)` on one finite rectangle. A named `Lambda <= t0` attempt with
pre-registered `t0 = 0.2 < 0.22` records a cited far-field premise
(Polymath15 / Rodgers–Tao) as an **external obligation**. The in-tree attempt
does not close: `certified=False`. A bounded real-axis rectangle pack reports
local `n_certified` / `n_blocked` counts; `finite_cover_certified` stays
false (a local pack is not a cover of `R` for all `t in [0, t0]`). It never
infers the Riemann Hypothesis and does not import `dirichlet`.

```python
from omnibias.core.verified.debruijn_newman import (
    PRE_REGISTERED_T0,
    attempt_named_lambda_bound,
    finite_ht_rectangle_pack,
    phi_enclosure,
)
from omnibias.core.verified.interval import Interval

phi0 = phi_enclosure(Interval.point(0.0), 6)
assert phi0.lo > 0.0

attempt = attempt_named_lambda_bound()
assert attempt.t0 == PRE_REGISTERED_T0
assert attempt.t0 < 0.22
assert attempt.certified is False
assert attempt.rh_claim is False
assert "far-field" in attempt.missing_piece.lower()

pack = finite_ht_rectangle_pack(
    boxes=((2.0, 1.0),),
    half_height=0.2,
    truncation=1.0,
    phi_terms=4,
    panels=8,
    contour_segments=8,
)
assert pack.finite_cover_certified is False
assert pack.rh_claim is False
assert pack.n_certified + pack.n_blocked == 1
```

## API

::: omnibias.core.verified.debruijn_newman
    options:
      show_root_heading: false
      heading_level: 3
