# omnibias-pinn

Physics-informed neural networks (PINNs) with **closed-form n-th
derivative operators**, built on top of `omnibias-core`. Cross-backend
(PyTorch + JAX) typed fields, ops, hard-conservation cages, prebuilt
PDE residuals, and diagnostics.

The package surfaces three layers:

| Layer | Purpose | Example |
| --- | --- | --- |
| **Fields** (`fields/`) | Structural backends with closed-form derivatives. | `OneLayerVectorField`, `SpectralVectorField`, `ChebyshevVectorField` |
| **Ops** (`ops/`) | User-facing operator surface (functional kernel). | `derivative`, `gradient`, `laplacian`, `biharmonic`, `curl`, `advection` |
| **Cage** (`cage/`) | Strict-conservation layers (hard physical constraints). | `StreamfunctionField`, `VectorPotentialField`, `HelmholtzProjectionField` |

Plus equation-aware modules:

* **Losses** (`losses/`): Sobolev preconditioning, Wang-Perdikaris
  causal weighting, NTK rebalance, entropy-consistent residual, and
  **asymptotic / removable boundary conditions** (`asymptotic_ratio`,
  `asymptotic_bc_loss`, `far_field_decay_loss`) -- the differentiable jet
  `lim` operator surfaced as trainable losses.
* **Equations** (`equations/`): prebuilt PDE residuals
  (NavierStokes, Burgers, Heat, KuramotoSivashinsky, CahnHilliard,
  Biharmonic) returning :class:`NamedTuple` outputs with diagnostics, plus the
  **nonlocal** ones -- CordobaCordobaFontelos (Hilbert transform) and the
  Fredholm / Volterra integral equations.
* **Diagnostics** (`diagnostics/`): backend-agnostic
  ``relative_l2_per_time``, ``forecast_horizon``, ``spectral_fidelity``,
  plus field-level ``derivative_stability`` and ``autograd_phase_check``.

## Pythonic DSL

The canonical user surface is **attribute-based** (Option 1 in the
design memo). Every :class:`FieldState` exposes per-component views
that route to the underlying functional ops:

```python
import torch

from omnibias.pinn._core.components import ComponentSpec
from omnibias.pinn._core.coords import CoordinateSpec
from omnibias.pinn.torch.fields import SpectralVectorField

field = SpectralVectorField(
    coordinate_spec=CoordinateSpec(axes=("x", "y", "z", "t"), time_axis="t"),
    components=ComponentSpec(("u", "v", "w"), groups={"velocity": ("u", "v", "w")}),
    K=8, time_hidden=32, time_depth=1, activation="tanh",
)

coords = torch.rand(16, 4, dtype=torch.float64) * 6.28
state = field(coords)            # FieldState
state.u.dt                       # ∂u/∂t
state.u.dx                       # ∂u/∂x
state.u.lap                      # Δu
state.u.biharm                   # Δ²u
state.velocity.div               # ∇·u
state.velocity.curl              # ∇×u  (3D)
state.velocity.advect()          # (u·∇) u  -- a method, unlike the properties above
```

The same operations are also available as functions
(`omnibias.pinn.torch.ops.derivative(state, "u", axis="t")` etc.) and
the DSL is a thin wrapper -- both routes share a single fastpath
implementation per backend.

## Top-level API

Both the PyTorch and JAX backends expose the same public surface
under their own root namespaces:

* `omnibias.pinn.torch.{fields, ops, cage, losses, equations, diagnostics}`
* `omnibias.pinn.jax.{fields,   ops, cage, losses, equations, diagnostics}`

The shared *schemas* (`CoordinateSpec`, `ComponentSpec`, `FieldState`)
live in `omnibias.pinn._core`.

## Fields

::: omnibias.pinn.torch.fields
    options:
      show_root_heading: false
      heading_level: 3

## Ops

::: omnibias.pinn.torch.ops
    options:
      show_root_heading: false
      heading_level: 3

## Cage

::: omnibias.pinn.torch.cage
    options:
      show_root_heading: false
      heading_level: 3

## Losses

::: omnibias.pinn.torch.losses
    options:
      show_root_heading: false
      heading_level: 3

## Equations

::: omnibias.pinn.torch.equations
    options:
      show_root_heading: false
      heading_level: 3

### Nonlocal residuals: integral equations

Every other equation here is **local** -- a residual at \(x\) reads the field and
its derivatives at \(x\) and nowhere else. `Fredholm` and `Volterra`
(`omnibias.pinn.{torch,jax}.equations.integral`) are not. They are the residuals
of an integral equation of the second kind,

\[
u(x) = f(x) + \lambda \int_\Omega K(x, t)\, u(t)\, d\mu(t),
\]

which couples every point to every other. That changes what an evaluation costs
and what the network must be able to do: it has to be evaluable at the quadrature
nodes, not only at the collocation points. In omnibias that is free, because a
field *is* a function -- `state.field(nodes)` re-evaluates it anywhere.

The two differ in one way that decides which is affordable:

| | Domain | Extra field evaluations per residual |
| --- | --- | --- |
| `Fredholm` | fixed \(\Omega\) | `n_nodes`, shared across the whole batch |
| `Volterra` | \([a, x]\), moves with the point | `batch * n_nodes`, nothing shareable |

`Volterra` stays mesh-free by pulling each interval back to a reference one,
\(t = a + (x - a)s\), and reusing a single fixed rule on \(s \in [0,1]\). That
buys the rule's own convergence order -- Gauss-Legendre, so spectral on a smooth
integrand -- instead of the second order a fixed cumulative-trapezoid grid would
give, which is what keeps `n_nodes` small enough for the cost above to be
tolerable. Its `axis` argument names the causal coordinate and every other one is
frozen at the collocation point's own value, so in a space-time problem it is a
memory term \(\int_0^t K(t,s)\, u(x, s)\, ds\) at fixed \(x\).

Only the integral is quadrature; local terms go through `state.ops.*` and stay
exact closed form. Both residuals are differentiable in the field parameters, the
kernel (so a **learned** kernel is a first-class case) and \(\lambda\), and both
return the nonlocal term alongside the residual, since it is the expensive and
the only approximated half.

Requires the quadrature from `omnibias-measure`:
`pip install "omnibias-pinn[integral]"`. When the equation is *not* coupled to a
PDE and nodal values are enough, the direct solvers in
[`omnibias.measure.{_core,torch,jax}.integraleq`](measure.md) are cheaper and
carry a Fredholm-alternative guard; see
[`docs/examples/measure_integraleq.py`](https://github.com/derivon-ai/omnibias/blob/main/docs/examples/measure_integraleq.py),
which validates both sides against the same analytic oracle.

::: omnibias.pinn.torch.equations.integral
    options:
      show_root_heading: false
      heading_level: 4
      members:
        - Fredholm
        - Volterra
        - fredholm
        - volterra
        - fredholm_residual_samples
        - volterra_residual_samples

## Proof Prep

::: omnibias.pinn.certified
    options:
      show_root_heading: false
      heading_level: 3
      members:
        - candidate_family_catalog
        - generate_fast_candidate_runs
        - replay_candidate_run
        - certify_candidate_run
        - falsify_candidate_run
        - attempt_analytic_proof_for_survivor
        - run_fast_candidate_sprint
        - build_ns_cap_bundle
        - ns_cap_schema_errors
        - certified_transfer_matrix_gap
        - certified_transfer_matrix_gap_schema_errors
        - certified_heat_kernel_transfer_gap
        - certified_heat_kernel_gap_schema_errors
        - certified_symmetric_heat_kernel_gap
        - certified_symmetric_heat_kernel_gap_schema_errors
        - certified_wilson_transfer_gap
        - certified_wilson_transfer_gap_schema_errors
        - heat_kernel_gap_scaling_report
        - heat_kernel_gap_scaling_schema_errors
        - certified_multistep_gap_refinement
        - multistep_gap_refinement_schema_errors
        - certified_effective_mass_curve
        - certified_effective_mass_curve_schema_errors
        - candidate_upgrade_gates
        - vorticity_residual_periodic
        - interval_arithmetic_metadata
        - interval_from_bounds
        - interval_add
        - interval_sub
        - interval_mul
        - interval_div
        - interval_square
        - interval_sqrt
        - interval_trapezoid_bound
        - compactification_map_interval
        - coefficient_interval_boxes
        - certified_tail_bounds_from_artifact
        - continuum_residual_certificates
        - finite_energy_tail_certificate
        - certified_clm_blowup
        - certified_clm_blowup_schema_errors
        - certified_clm_multizero_first_blowup
        - certified_clm_multizero_first_blowup_schema_errors
        - certified_ccf_selfsimilar_blowup_attempt
        - certified_ccf_selfsimilar_blowup_attempt_schema_errors
        - certified_ccf_linearized_operator_bound
        - certified_ccf_linearized_operator_bound_schema_errors
        - certified_euler2d_steady_vortex
        - certified_euler2d_steady_vortex_schema_errors
        - taylor_green_vortex
        - kolmogorov_flow
        - certified_taylor_green_residual
        - certified_kolmogorov_residual
        - certified_periodic_flow_residual
        - certified_periodic_flow_residual_schema_errors
        - periodic_residual_digest_ok
        - beltrami_abc_flow
        - shear_streamfunction
        - cellular_streamfunction
        - streamfunction_from_descriptor
        - certified_streamfunction_residual
        - certified_shear_streamfunction_residual
        - streamfunction_residual_schema_errors
        - fourier_mode_vorticity
        - vorticity_from_descriptor
        - integrate_vorticity_2d
        - certified_rollout_diagnostics
        - rollout_diagnostics_schema_errors
        - certified_sqg_steady_vortex
        - certified_sqg_steady_vortex_schema_errors
        - certified_sqg_selfsimilar_blowup_attempt
        - certified_sqg_selfsimilar_blowup_attempt_schema_errors
        - certified_sqg_linearized_coercivity_attempt
        - certified_sqg_linearized_coercivity_attempt_schema_errors
        - refine_ccf_selfsimilar_profile
        - radii_polynomial_closure
        - default_ccf_collocation_nodes
        - certified_gclm_selfsimilar_blowup
        - certified_gclm_selfsimilar_blowup_schema_errors
        - certified_gclm_gradient_amplification
        - certified_gclm_gradient_amplification_schema_errors
        - axisymmetric_axis_smoothness_certificate
        - build_axisymmetric_interval_report
        - certified_candidate_refinement_report
        - axisymmetric_function_space_metadata
        - assemble_axisymmetric_linearized_operator
        - operator_theoretic_invertibility_certificate
        - componentwise_radii_polynomial_certificate
        - radii_polynomial_certificate
        - norm_divergence_certificate
        - theorem_grade_function_space_contract
        - continuum_banach_invertibility_attempt
        - theorem_grade_radii_polynomial_attempt
        - exact_profile_norm_divergence_attempt
        - regularity_all_data_proof_attempt
        - build_theorem_grade_closure_attempt
        - exact_navier_stokes_equation_contracts
        - theorem_grade_function_space_definitions
        - interval_cap_backend_contract
        - blowup_route_lemma_package
        - regularity_route_lemma_package
        - proof_obligation_bundle
        - blowup_proof_obligation_bundles
        - regularity_proof_obligation_bundles
        - theorem_verifier_record
        - ingest_theorem_verifier_bundle
        - lean_formalization_package
        - external_review_gate
        - build_ns_proof_program_report
        - external_verification_record
        - verify_external_proof_package
        - theorem_claim_gate
        - build_axisymmetric_blowup_closure_report
        - build_blowup_closure_report
        - build_regularity_closure_report
        - build_regularity_inequality_report
        - regularity_counterexample_sweep
        - build_analytic_closure_report
        - build_formal_proof_package
        - build_certificate_manifest

## Diagnostics

::: omnibias.pinn.torch.diagnostics
    options:
      show_root_heading: false
      heading_level: 3

## Discontinuity-capturing PINN (partition)

A single smooth activation network cannot represent a kink / shock / phase
boundary; a **soft partition of unity of smooth sub-solutions** can.
`omnibias.pinn.partition` (a bridge on the [`omnibias-partition`](partition.md)
keystone) provides `PartitionedField`, a genuine PINN field
`u(x) = Σ_l w_l(x) u_l(x)` that plugs into the existing ops and develops an
interface between regions as the gate sharpness `beta -> ∞`. The conservative
(cPINN) demo enforces the PDE per region with an interface-continuity penalty,
beating a single `OneLayerVectorField` on interface error.

**Honesty label.** The blended field's derivatives use the **autodiff product
rule** (the closed-form `sigma`-tower does not cover products of sigmoids); the
sound, certified quantity is the *soft->hard partition gap*
(`omnibias.partition.certify_partition_gap`). The `beta -> ∞` hardening is the
feasibility / temperature sense of "collapse", never the founding `delta -> 0`
bias collapse. Torch only in this first pass.

::: omnibias.pinn.partition.torch
    options:
      show_root_heading: false
      heading_level: 3

## Extensions

Opt-in helpers that wire model-level operators into the
`omnibias.fields.ops_registry` extension point. Importing the package registers
nothing; call the `register_*` helpers to opt in. `register_lim_along` exposes
the closed-form jet `lim` operator as `state.<component>.lim_along`.

::: omnibias.pinn.extensions
    options:
      show_root_heading: false
      heading_level: 3

## Core schemas

::: omnibias.pinn._core
    options:
      show_root_heading: false
      heading_level: 3

## JAX twin

The JAX backend has the same module layout under
`omnibias.pinn.jax`. All cross-backend tests in
`packages/omnibias-pinn/tests/cross_backend/` assert *bit-identical*
results between the two backends (typical tolerances: rtol/atol=1e-12
in float64).
