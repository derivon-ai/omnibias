# Spectral floors and adapted SOS bases

A Rayleigh quotient is a free upper bound on the lowest
eigenvalue. A lower bound needs a separator and a trial
space that sees the ground state. Alignment is reported on
every bound. This is one fixed matrix, not a continuum
spectral gap.

```python
from omnibias.core.verified.trial_spaces import (
    DISCLAIMER,
    certified_floor,
    honesty_payload,
    sine_trial_space,
    synthetic_diagonal,
    trial_space_alignment,
)

matrix, evec, rho = synthetic_diagonal(4, seed=1)
nodes = [0.0, 1.0, 2.0, 3.0]
trial = sine_trial_space(1, domain=(0.0, 3.0))
cert = certified_floor(matrix, trial, nodes, reference_vector=evec, rho=rho)
assert 0.0 <= cert.alignment <= 1.0
assert cert.alignment == trial_space_alignment(trial, evec, nodes)
assert cert.honesty["yang_mills_mass_gap"] is False
assert honesty_payload()["theorem_prover_verified"] is False
assert "not a Yang-Mills mass gap" in DISCLAIMER
```

An arrangement-adapted monomial basis can certify a high
even power at a lower ambient degree than the total-degree
basis.

```python
from omnibias.sos.certify import certify_sos
from omnibias.sos.monomials import arrangement_adapted_basis
from omnibias.sos.problem import Polynomial

p = Polynomial.monomial((6,), 1.0) + Polynomial.constant(1.0, 1)
adapted = arrangement_adapted_basis(p, ((1.0,),), degree=1)
assert certify_sos(p, basis=adapted).certified
assert not certify_sos(p, half_degree=1).certified
```
