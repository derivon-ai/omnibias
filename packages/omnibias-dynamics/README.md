# omnibias-dynamics

**Status: Alpha (0.1.0a1).**

Computer-assisted dynamics on the omnibias *validated* tower. Rigorous
(interval / Taylor-model) tools for nonlinear ODEs, built on the QR-Lohner
validated flow, the Newton-Kantorovich / radii-polynomial existence machinery,
and the closed-form variational tower in `omnibias.core.verified`.

Capabilities (CAPD-class results, on the exact variational tower):

- **Validated variational / monodromy flow** -- propagate a state *and* its
  fundamental matrix rigorously; the basis for Floquet multipliers and stability.
- **Poincare-section enclosures** -- a rigorous return map across a hyperplane.
- **Certified Lyapunov-exponent bounds** -- two-sided enclosures of the leading
  exponent from the validated variational flow.
- **Periodic-orbit existence** -- a radii-polynomial proof that a *true* periodic
  orbit lives in an explicit ball around a numerical guess.

## What this is and is not

- Every enclosure is **sound by construction**: it provably contains the true
  trajectory / matrix / exponent for the whole time interval. An existence claim
  is a **proof**, never a heuristic.
- It is **not** a fast non-rigorous integrator: validated integration trades
  speed for certainty, and a too-large step or too-chaotic a system makes the
  enclosure blow up (reported honestly, never silently widened away).
- Pure Python; depends only on `omnibias-core` (no torch / jax).

## Public API

```python
from omnibias.dynamics import (
    variational_flow, monodromy_matrix, VariationalState,
    poincare_map, PoincareSection,
    certified_lyapunov_exponent, LyapunovBounds,
    prove_periodic_orbit, PeriodicOrbitCertificate,
)
```

## Tests

```bash
python -m pytest packages/omnibias-dynamics/tests -q
```

## License

Dual-licensed: AGPL-3.0-or-later OR a commercial licence from Derivon
(`LicenseRef-omnibias-Commercial`). See [`LICENSE`](LICENSE),
[`../../LICENSING.md`](../../LICENSING.md), and
[`../../COMMERCIAL-LICENSE.md`](../../COMMERCIAL-LICENSE.md). Contact
info@derivon.ai for commercial terms.
