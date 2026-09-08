# Swirl-heat identity and pulse envelope (07-12)

The isotropic exterior `K = 1/r` (`h = 0`, `H ≡ 1`) solves the radial
`m = 1` heat equation exactly over `Q`. The holonomic annihilator is
accepted via `prove("identity")`. The pulse envelope `P = s t` has
exact `P'` from the sigmoid tower, not a finite difference. Mollifier
tails reuse 01-05.

The paper uses `h > 0`; that scaling is leftover-recorded. This is not
a 3-D heat theorem. See theory spec
[07-12](https://github.com/derivon-ai/omnibias/blob/main/theory/07-frontier/12-heat-exterior-pulse-envelope.md).

Homes: `omnibias.core.verified.swirl_heat`,
`omnibias.holonomic.swirl_heat`, `omnibias.core.pulse_envelope`.

```python
from fractions import Fraction

from omnibias.core.pulse_envelope import (
    GROWTH_RATE,
    locked_growth_envelope,
    mollifier_tail_contains_truth,
)
from omnibias.core.verified.swirl_heat import swirl_heat_residual
from omnibias.holonomic.swirl_heat import prove_swirl_heat_identity

assert swirl_heat_residual() == 0
assert prove_swirl_heat_identity().proved
growth = locked_growth_envelope()
assert growth.derivative() == growth.tower_derivative()
assert growth.derivative() == GROWTH_RATE * growth.value()
assert mollifier_tail_contains_truth(half_width=3.0)
assert Fraction(1, 2) == GROWTH_RATE
```
