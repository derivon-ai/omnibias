# omnibias-sos

**Status: Alpha (0.1.0a1).**

Certified *universal* positivity by optimization. A polynomial `p(x) >= 0` for
**all** `x` if and only if it admits a Sum-of-Squares (SOS) decomposition
`p(x) = z(x)^T Q z(x)` with a positive-semidefinite Gram matrix `Q`. omnibias-sos
finds `Q` with a floating-point semidefinite program (a *proposer*) and then
**proves** the result with a rigorous interval LDL^T positive-definiteness
certificate built on `omnibias.core.verified` -- the same certificate the
Mathlib-free Lean kernel already re-checks, so a sealed SOS certificate can earn
`theorem_prover_verified`.

This is the one shape of "optimization that proves a `for all` statement"
soundly: the optimizer only proposes; the interval algebra and the Lean kernel
do the proving.

Capabilities:

- **Global SOS positivity** -- a sound proof that a polynomial is nonnegative
  everywhere on `R^n`.
- **Positivstellensatz** -- constrained positivity `p >= 0` on a semialgebraic
  set `{g_i >= 0}` via `p = s_0 + sum s_i g_i` with each `s_i` a certified SOS
  (fixed-degree Putinar form).
- **Auxiliary-functional (background) method** -- a certified `for all initial
  data` upper bound on the infinite-time average of an observable for a
  **polynomial ODE / Galerkin-truncated** system, from an SOS-certified auxiliary
  functional.

## What this is and is not

- The SDP solve is a floating-point **proposer**; it never touches the proof. The
  proof is the outward-rounded interval LDL^T certificate (soundness over
  completeness): if the rational rounding or the positive-definite margin fails,
  the result is **inconclusive**, never a false positivity claim.
- SOS / Positivstellensatz positivity is a genuine `for all x` statement about a
  **polynomial**.
- The auxiliary-functional bound is a `for all data` statement about a
  **finite-dimensional / Galerkin-truncated** system. It is **not** a continuum
  PDE regularity statement and **not** a open-problem claim
  (`unproven_claim = False` on every certificate).
- Pure Python + numpy; depends only on `omnibias-core` (no torch / jax). The
  optional Lean check degrades gracefully when no toolchain is present.

## Public API

```python
from omnibias.sos import (
    Polynomial, monomial_basis,
    certify_sos, SOSCertificate,
    certify_nonneg_on_set,
    PolynomialSystem, certify_time_average_bound,
    seal_sos_certificate, lean_check_sos,
)
```

## Tests

```bash
python -m pytest packages/omnibias-sos/tests -q
```

## License

Dual-licensed: AGPL-3.0-or-later OR a commercial licence from Derivon
(`LicenseRef-omnibias-Commercial`). See [`LICENSE`](LICENSE),
[`../../LICENSING.md`](../../LICENSING.md), and
[`../../COMMERCIAL-LICENSE.md`](../../COMMERCIAL-LICENSE.md). Contact
info@derivon.ai for commercial terms.
