# Adaptive pack refinement

A pack bank can add a zero-weight pack (birth), add a higher-order
zero-weight sibling (growth), or remove a pack whose contribution is
tiny (death). Birth and growth leave the output bit-identical. Death
reports the perturbation. This is theory spec 03-13. Scale `alpha` is
the tempering scale, not temperature collapse.

## Worked boundary-layer jet (spec §5)

Taylor coefficients of `exp(-100 x)` about `x = 0.005` have ratios
`a_k / a_{k-1} = -100 / k`. The scale estimator recovers `-100`. A
residual indicator, given only existing packs at scale `2`, does not.

```python
import math
from omnibias.core.refine import (
    Indicator,
    RefinePolicy,
    RefinedPack,
    hp_decision,
    local_scale_from_derivatives,
    propose_refinement,
)

derivs = tuple((-100.0) ** k * math.exp(-0.5) for k in range(7))
assert abs(local_scale_from_derivatives(derivs) + 100.0) < 1e-9
assert hp_decision(derivs, existing_scale=2.0) == "h"
assert hp_decision(derivs, existing_scale=-100.0) == "p"

packs = (
    RefinedPack(order=0, center=0.25, weight=1.0, scale=2.0),
    RefinedPack(order=0, center=0.75, weight=1.0, scale=2.0),
)
residual = propose_refinement(
    packs,
    policy=RefinePolicy(indicator=Indicator.RESIDUAL, hysteresis=1.0, min_scale_ratio=1.0),
    peak_location=0.005,
    peak_value=1.0,
    derivatives=derivs,
    birth_score=1.0,
)
sing = propose_refinement(
    packs,
    policy=RefinePolicy(indicator=Indicator.SINGULARITY, hysteresis=1.0, min_scale_ratio=1.0),
    peak_location=0.005,
    peak_value=1.0,
    derivatives=derivs,
    birth_score=1.0,
)
assert residual is not None and abs(residual.scale) == 2.0
assert sing is not None and abs(sing.scale + 100.0) < 1e-9
```

## Birth is bit-identical

```python
import torch
from omnibias.core.refine import Indicator, RefinePolicy, RefinedPack
from omnibias.torch.refine import AdaptivePackBank, refine

torch.set_default_dtype(torch.float64)
bank = AdaptivePackBank(
    (
        RefinedPack(order=0, center=0.25, weight=1.0, scale=2.0),
        RefinedPack(order=0, center=0.75, weight=0.5, scale=2.0),
    ),
    max_packs=8,
    base="exp",
)
x = torch.linspace(0.0, 1.0, 17)
before = bank(x).clone()

def residual(z):
    return torch.ones_like(z)

report = refine(
    bank,
    residual,
    x,
    RefinePolicy(
        indicator=Indicator.SINGULARITY,
        death_threshold=0.0,
        min_age=10_000,
        hysteresis=1.0,
        min_scale_ratio=1.0,
    ),
    step=0,
    probe_jet=(0.005, tuple((-100.0) ** k * __import__("math").exp(-0.5) for k in range(7))),
)
assert report.born
assert torch.equal(before, bank(x))
```
