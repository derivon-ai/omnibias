# Einselection collapse cookbook

Einselection collapse decides, on a sound enclosure of a pure-dephasing
density matrix, whether the off-diagonal coherence stays below a
declared sensitivity `eps`. `PROVED` means the einselected distribution
`{|c_i|^2}` survives -- not that "the wave function collapsed." The
global state stays pure and entangled throughout; this is never a
single-outcome claim.

## A two-level dephasing model decoheres past `t = 20`

```python
import math

from omnibias.core.collapse import DephasingModel, einselection_collapse
from omnibias.core.verified.complex_interval import ComplexInterval
from omnibias.core.verified.interval import Interval

amplitude = ComplexInterval.point(1.0 / math.sqrt(2.0))
model = DephasingModel(
    amplitudes=(amplitude, amplitude),
    rates=(
        (Interval.point(0.0), Interval.point(1.0)),
        (Interval.point(1.0), Interval.point(0.0)),
    ),
)

still_detectable = einselection_collapse(model, time=5.0, coherence_budget=1e-6)
assert still_detectable.status == "DISPROVED"

einselected = einselection_collapse(model, time=20.0, coherence_budget=1e-6)
assert einselected.status == "PROVED"
assert einselected.outcome.surviving == "einselected_distribution"
assert einselected.outcome.honesty["wave_function_collapse_claim"] is False
assert einselected.outcome.honesty["single_outcome_claim"] is False
```

## The einselected distribution is the initial populations, not a new computation

```python
from omnibias.core.collapse import einselected_distribution

populations = einselected_distribution(model)
assert all(box.contains(0.5) for box in populations)
```

## `Gamma = 0` never proves: coherence never decays

```python
never_proves = einselection_collapse(
    DephasingModel(
        amplitudes=(amplitude, amplitude),
        rates=(
            (Interval.point(0.0), Interval.point(0.0)),
            (Interval.point(0.0), Interval.point(0.0)),
        ),
    ),
    time=1000.0,
    coherence_budget=0.4,
)
assert not never_proves.proved
```

## Pointer-basis check: does a candidate observable commute with `H`?

```python
from omnibias.core.collapse import pointer_basis_verdict

diagonal_a = (
    (ComplexInterval.point(1.0), ComplexInterval.point(0.0)),
    (ComplexInterval.point(0.0), ComplexInterval.point(-1.0)),
)
diagonal_h = (
    (ComplexInterval.point(2.0), ComplexInterval.point(0.0)),
    (ComplexInterval.point(0.0), ComplexInterval.point(3.0)),
)
commutes = pointer_basis_verdict(diagonal_a, diagonal_h)
assert commutes.status == "PROVED"
assert commutes.outcome.surviving == "commuting_pointer_basis"

sigma_x = (
    (ComplexInterval.point(0.0), ComplexInterval.point(1.0)),
    (ComplexInterval.point(1.0), ComplexInterval.point(0.0)),
)
sigma_z = (
    (ComplexInterval.point(1.0), ComplexInterval.point(0.0)),
    (ComplexInterval.point(0.0), ComplexInterval.point(-1.0)),
)
does_not_commute = pointer_basis_verdict(sigma_x, sigma_z)
assert does_not_commute.status == "DISPROVED"
```

`propose_pointer_basis` is a float heuristic ranking of candidate basis
indices by population weight. It never gates either verdict above --
exactly as rank collapse's `sigma_min` only proposes a kernel vector.

```python
from omnibias.core.collapse import propose_pointer_basis

ranking = propose_pointer_basis(model)
assert set(ranking) == {0, 1}
```
