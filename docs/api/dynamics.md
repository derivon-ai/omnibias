# omnibias-dynamics

Computer-assisted dynamics on the omnibias *validated* tower: rigorous
(interval / Taylor-model) tools for nonlinear ODEs, built on the QR-Lohner
validated flow, the Newton-Kantorovich / radii-polynomial existence machinery,
and the closed-form variational tower in `omnibias.core.verified`.

- **Validated variational / monodromy flow** — propagate a state *and* its
  fundamental matrix rigorously (the basis for Floquet multipliers / stability).
- **Poincare-section enclosures** — a rigorous return map across a hyperplane.
- **Certified Lyapunov-exponent bounds** — two-sided enclosures of the leading
  exponent from the validated variational flow.
- **Periodic-orbit existence** — a radii-polynomial proof that a *true* periodic
  orbit lives in an explicit ball around a numerical guess.
- **Closed-form-tower jet bridge** — `vector_field_from_sigma_tower` and
  `sigma_oscillator_field` turn an activation's exact derivative tower into the
  `(VectorField, JacobianEnclosure)` pair the validated flow expects (no autodiff),
  and `discrete_periodic_point` proves fixed points / periodic orbits of iterated
  1-D maps via a Krawczyk certificate.

!!! note "Soundness, not speed"
    Every enclosure provably contains the true object over the whole time
    interval, and an existence claim is a proof. Validated integration trades
    speed for certainty; a too-large step or too-chaotic a system makes the
    enclosure blow up, which is reported honestly rather than silently widened.

## Public API

::: omnibias.dynamics
    options:
      show_root_heading: false
      heading_level: 3
      members_order: source

Status: Alpha (`0.1.0a1`).
