# omnibias-control

**Status: Alpha (0.1.0a1).**

Differentiable control for omnibias with a **model-relative** safety certificate. A control-barrier-function
(CBF) **safety filter** takes a nominal (task) action `a_nom` and returns the *nearest*
action that keeps the system inside a safe set:

\[
a^\star(s) \;=\; \arg\min_a \tfrac12\lVert a - a_{\text{nom}}\rVert^2
\quad\text{s.t.}\quad G(s)\,a \le h(s),
\]

a **state-dependent** QP whose rows are the CBF inequality (*brake before you hit the
obstacle*) plus the actuator box `||a||_inf <= a_max`. This is the standard CBF-QP
safety filter (Ames et al. 2019) -- but two things that have never been combined in one
object come straight from omnibias:

- **Differentiable + batched + trainable end to end.** The filter is a
  temperature-collapse / hard-hinge projection layer generalised to *per-sample*
  constraints `(G(s), h(s))`: one `jit`
  call solves a whole batch by accelerated (Nesterov) gradient descent on a hard-hinge
  penalty, so you can backprop *through the safety filter* and train the policy to
  anticipate it (feasible-by-construction, not fighting the filter).
- **A rigorous safety certificate.** `omnibias-verify` (interval branch-and-bound)
  certifies the *recoverable set*: the region of states from which an actuator-admissible
  action provably keeps the barrier non-negative. On the certified region, forward
  invariance is a theorem, not a test-set statistic -- and the same register yields a
  **certified (model-relative) safe speed limit**.

Prior safety layers give feasibility but no certificate and no GPU-scale batched training;
OptNet / differentiable-MPC give gradients but no proof. The differentiable-*and*-certified
combination is what this package offers.

## What's in the box

```python
import jax.numpy as jnp
from omnibias.control import CBFSpec, FilterSchedule
from omnibias.control.jax import (
    cbf_filter,               # differentiable per-sample projection a_nom -> a*
    control_affine_cbf_rows,  # autodiff Lie-derivative CBF rows for x_dot = f(x)+g(x)a
    lagrangian_cbf_rows,      # same, for dynamics from a (learned) Lagrangian
    safe_rollout,             # differentiable safe closed loop (train THROUGH the filter)
)
from omnibias.control import certify_disc_recoverable   # rigorous recoverable-set proof
```

Every function has a bit-identical `omnibias.control.torch` twin (parity `~1e-12`).

- **`cbf_filter(a_nom, G, h, schedule)`** -- the projection. `G (B,m,d)`, `h (B,m)`,
  `a_nom (B,d)`; returns `a* (B,d)`. `FilterSchedule()` is eval-quality;
  `FilterSchedule.fast()` is enough to train through.
- **`control_affine_cbf_rows(f, g, barrier, x, spec)`** -- builds `G(x) a <= h(x)` from
  the drift `f`, input map `g`, and barrier `h` by autodiff (relative degree 1 or 2),
  appending the actuator box when `spec.a_max` is set.
- **`lagrangian_cbf_rows(L, B, barrier, q, qdot, t, spec)`** -- the same, with `f, g`
  assembled from an `omnibias.variational` Lagrangian (`g = M(q)^{-1} B`) -- so a
  **learned** Lagrangian (LNN) plugs straight in.
- **`certify_disc_recoverable(center, radius, gains, a_max, ranges, vmax, g=...)`** --
  a rigorous `RecoverableCertificate` (`certified` iff `f_lower >= 0`). Pass the model
  matrix `g = M^{-1} B` (the *learned* one for a learned-dynamics filter).

## Honest scope

- The certificate is **model-relative**: rigorous for the model you pass (closed-form
  `phi` for the disc obstacle). For a learned model, report its empirical error separately;
  the guarantee transfers to the true system only up to that error. This is not a robust
  certificate against arbitrary model mismatch.
- CBF conservativeness is set by the (design-chosen) class-K `gains`; the certified safe
  speed is honest but not the true viability kernel.
- Field-level math is torch + jax only (repo convention). Extension-tier typing (authored
  strict-clean; not on the shared strict CI gate).

## Tests

```bash
pip install -e "packages/omnibias-control[jax,torch,lagrangian,verify,test]"
python -m pytest packages/omnibias-control/tests -q
```

## License

Dual-licensed: AGPL-3.0-or-later OR a commercial licence from Derivon
(`LicenseRef-omnibias-Commercial`). See [`LICENSE`](LICENSE),
[`../../LICENSING.md`](../../LICENSING.md), and
[`../../COMMERCIAL-LICENSE.md`](../../COMMERCIAL-LICENSE.md). Contact
info@derivon.ai for commercial terms.
