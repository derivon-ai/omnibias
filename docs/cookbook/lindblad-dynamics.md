# Open-system Lindblad dynamics

A time-independent GKSL generator is a linear semigroup. Arbitrary-order
time derivatives come from one propagator plus matrix powers of `L` --
the linear-semigroup analogue of the sigma tower, not the activation
tower. See [the API reference](../api/lindblad.md) and the
[theory spec](https://github.com/derivon-ai/omnibias/blob/main/theory/09-inventions/32-open-system-lindblad-dynamics.md).

## Qubit amplitude damping matches the Bloch closed form

```python
from omnibias.core.lindblad import (
    density_matrix,
    qubit_bloch_solution,
    qubit_thermal,
)

model = qubit_thermal(omega=1.0, beta=0.8, gamma_down=0.4)
rho0 = ((0.2 + 0j, 0.1 + 0j), (0.1 + 0j, 0.8 + 0j))
got = density_matrix(model, rho0, 1.0)
closed = qubit_bloch_solution(
    omega=1.0,
    gamma_down=0.4,
    gamma_up=model.rates[1],
    gamma_phi=0.0,
    rho0=rho0,
    time=1.0,
)
assert abs(got[1, 1] - closed[1, 1]) < 1e-12
```

## The thermal steady population is the shipped Fermi occupancy

```python
from omnibias.core.lindblad import thermal_steady_population
from omnibias.core.occupancy import FermiModel, occupancy

omega, beta = 1.3, 0.7
p_e = thermal_steady_population(omega=omega, beta=beta)
assert abs(p_e - occupancy(FermiModel(beta=beta, mu=0.0), omega)) < 1e-15
```

## Relaxation collapse disagrees with einselection on pure dephasing

On a pure-dephasing model every diagonal state is fixed: einselection
can prove an einselected distribution, while relaxation must refuse
because there is a steady-state manifold, not a unique fixed point.

```python
import math

from omnibias.core.collapse import (
    DephasingModel,
    einselection_collapse,
    relaxation_collapse,
)
from omnibias.core.lindblad import qubit_pure_dephasing
from omnibias.core.verified.complex_interval import ComplexInterval
from omnibias.core.verified.interval import Interval

amplitude = ComplexInterval.point(1.0 / math.sqrt(2.0))
dephasing = DephasingModel(
    amplitudes=(amplitude, amplitude),
    rates=(
        (Interval.point(0.0), Interval.point(1.0)),
        (Interval.point(1.0), Interval.point(0.0)),
    ),
)
ein = einselection_collapse(dephasing, time=20.0, coherence_budget=1e-6)
assert ein.status == "PROVED"
relax = relaxation_collapse(
    qubit_pure_dephasing(rate=1.0), time=20.0, distance_budget=1e-6
)
assert relax.status == "BLOCKED"
assert relax.outcome.honesty["wave_function_collapse_claim"] is False
```
