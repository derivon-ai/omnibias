# Proof-carrying fluid dynamics

This cookbook certifies the **residual** of an incompressible periodic flow and
adjudicates it through the proof machine. It is the nonlinear companion to
[proof-carrying PDE certificates](proof-carrying-pde.md): Navier–Stokes is
nonlinear, so instead of the linear a-posteriori error theorem we seal the
*evidence* that a sampled field satisfies the momentum, continuity and
pressure-Poisson equations to a stated tolerance.

```python
from omnibias.core.proof import Conjecture
from omnibias.pinn.certified import (
    build_default_machine,
    certified_taylor_green_residual,
)
from omnibias.symbolic.fluid import verify_periodic_flow_residual

# Generate + certify the exact 2-D Taylor-Green vortex (no external data).
cert = certified_taylor_green_residual(64, viscosity=0.1)
assert cert["exact_solution_claim"] is True
assert cert["residual_sup"] < 1e-8          # momentum + continuity + pressure-Poisson

machine = build_default_machine()
verdict = machine.evaluate(
    Conjecture(
        "Taylor-Green periodic residual",
        "navier_stokes_periodic_residual",
        {"certificate": cert},
    )
)
assert verdict.status == "PROVED"
assert verdict.schema_ok and verdict.replay_ok and verdict.honesty_ok

# The independent numpy twin regenerates the flow from the descriptor and re-checks.
report = verify_periodic_flow_residual(cert)
assert report["replay_match"] is True
```

> **Not the global-regularity problem.** `PROVED` here means the *residual* of an exact,
> closed-form model flow (Taylor–Green) is certified below tolerance on the whole
> torus — **not** global-in-time regularity of 3D Navier–Stokes.

The certificate records, for the sampled field on the periodic torus:

```text
||rho (u_t + (u.grad)u) + grad p - mu Lap u - f||_inf   (momentum)
||div u||_inf                                            (continuity)
||-Lap p - rho d_i d_j (u_i u_j)||_inf                   (pressure-Poisson)
```

together with kinetic energy, enstrophy, palinstrophy and a regenerable
`fixture` descriptor.

## Exact fixtures (2-D and 3-D)

| Builder / fixture | Flow | Role |
|---|---|---|
| `certified_taylor_green_residual` | exact 2-D decaying Taylor–Green vortex | laminar correctness baseline |
| `certified_kolmogorov_residual` | exact steady forced shear `u = (A sin ky, 0)` | forced / chaotic-facing base state |
| `beltrami_abc_flow` | exact **3-D** Arnold–Beltrami–Childress decaying flow | 3-D Navier–Stokes companion |

The 3-D ABC flow is an exact unsteady solution of the incompressible
Navier–Stokes equations; its residual certifies to machine zero on the periodic
3-torus, and its independent twin regenerates it from a JSON descriptor. This is
the honest answer to *"can we do 3-D?"* — yes for an **exact analytic** 3-D
solution; it is **not** a global-regularity theorem (`unproven_claim = False`).

## What the proof machine checks

The `navier_stokes_periodic_residual` prover applies the same gates as the other
certified-evidence stacks:

- **schema** — required residual fields present, `residual_tol` positive, and an
  `exact_solution_claim` must be self-consistent with the recorded sups;
- **replay** — `omnibias.symbolic.fluid.verify_periodic_flow_residual`
  regenerates the flow from the descriptor with an independent spectral
  implementation *and* a `np.roll` central finite-difference cross-check
  (`finite_difference_consistent`), confirming the recorded sups match (catching
  under- and over-statement);
- **honesty** — `unproven_claim`, `continuum_navier_stokes_claim`,
  `chaotic_tracking_claim`, `perfect_weather_claim` and
  `turbulence_closure_claim` are all `False`; asserting any of them is downgraded
  to `BLOCKED`.

## Rigorous interval residual (whole-domain)

The FFT residual above is sampled on a grid. The
`navier_stokes_streamfunction_residual` certificate **closes that gap** with a
genuine interval enclosure valid *between* the nodes, on the whole cell:

```python
from omnibias.pinn.certified import certified_shear_streamfunction_residual
from omnibias.symbolic.fluid import verify_streamfunction_residual

# A y-only tanh streamfunction is a rigorously exact steady-Euler shear.
cert = certified_shear_streamfunction_residual(splits=4)   # a v1 make_certificate dict
assert cert["honesty"]["interval_verified"] is True        # genuine interval enclosure
assert cert["payload"]["residual_sup"] < 1e-10             # encloses machine zero on the WHOLE domain
assert cert["payload"]["divergence_sup"] < 1e-10           # div u = psi_xy - psi_yx, structural

# Independent analytic tanh-tower twin (a different algorithm from the interval jet).
assert verify_streamfunction_residual(cert)["replay_match"] is True
```

Incompressibility is enforced *by construction* through the **streamfunction
cage** `u = ∇⊥ψ`; the certified quantity is the steady vorticity-transport
residual `(u·∇)ω − νΔω − f_ω`, computed with the verified `tanh` Cauchy-product
jet. For a general field, call
`certified_streamfunction_residual(field, splits=...)`; raising `splits`
**tightens** the finite enclosure — the interval-analysis signature.

## Genuine rollout diagnostics (real time integration)

`navier_stokes_rollout_diagnostics` integrates the flow forward for real
(pseudo-spectral vorticity–streamfunction, integrating-factor RK2, 2/3 dealiasing)
and seals honest **window diagnostics** — not pointwise truth:

```python
from omnibias.pinn.certified import certified_rollout_diagnostics, fourier_mode_vorticity
from omnibias.symbolic.fluid import verify_rollout_diagnostics

modes = [{"kx": 1, "ky": 0, "amp": 1.0, "phase": 0.0},
         {"kx": 0, "ky": 2, "amp": 0.7, "phase": 0.3}]
_, descriptor = fourier_mode_vorticity(64, modes)   # (field, regenerable descriptor)
cert = certified_rollout_diagnostics(descriptor, viscosity=0.0, dt=2e-3, steps=200)
assert cert["max_divergence"] < 1e-10                 # incompressibility maintained
assert cert["incompressibility_maintained"] is True
assert cert["honesty"]["chaotic_tracking_claim"] is False
assert cert["honesty"]["perfect_weather_claim"] is False
assert verify_rollout_diagnostics(cert)["replay_match"] is True   # independent re-integration
```

## Claim boundary

- The **FFT** residuals (`certified_periodic_flow_residual` and friends) are
  spectral *sampling*, not interval enclosure (`interval_verified=False`). They
  now also carry a band-limited `spectral_l1_residual_bound` (a between-node sup
  bound for the truncated polynomial actually sampled) and a tamper-evident body
  digest (`periodic_residual_digest_ok`), and the twin adds an independent
  finite-difference cross-check — but these are still *evidence for known analytic
  model flows*.
- These FFT sup numbers are **not bit-reproducible across platforms**; "bit-stable"
  refers to the closed-form derivative tower in `omnibias.core`, not a backend FFT
  float.
- The **streamfunction** certificate *is* a genuine whole-domain interval
  enclosure (`interval_verified=True`) for a constructed neural field — still not a
  continuum existence/uniqueness or global-regularity statement.
- Nothing here is perfect weather, high-Reynolds turbulence closure, or pointwise
  long-horizon chaos tracking. See [scope & guarantees](../scope-and-guarantees.md)
  §3.3–3.5.

## Runnable demo

The CI-sized demo lives in
[`examples/certified_fluid_dynamics/`](../../examples/certified_fluid_dynamics/):

```bash
python -m examples.certified_fluid_dynamics.run_demo
python -m examples.certified_fluid_dynamics.run_demo --n 128 --viscosity 0.05
```

No data is downloaded; the optional `--scratch-dir` (or `$OMNIBIAS_SCRATCH`) is a
runtime cache for generated arrays only.
