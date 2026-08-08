# omnibias.pinn.operator (neural operator learning)

Neural operator learning on the omnibias closed-form derivative tower, shipped
inside `omnibias-pinn` as the `omnibias.pinn.operator` submodule. A DeepONet

$$G(u)(y) = b_0 + \sum_k b_k(u)\, t_k(y)$$

is linear in the trunk basis, so every query-coordinate derivative is

$$
\partial^\alpha G(u)(y) = \sum_k b_k(u)\, \partial^\alpha t_k(y)
$$

and the trunk is an omnibias jet network. One trunk jet therefore yields every
mixed partial of the operator output up to a chosen order -- mesh-free, with no
finite differences and no periodic-grid requirement on the query side.

"Operator" in omnibias names three different objects; see
[docs/operator-surface.md](../operator-surface.md) ("Three senses of operator").
This submodule is sense 3 -- neural operator learning (function-space to
function-space maps) -- not `OperatorBlock` and not a field operator like
`grad` / `laplacian`.

## Honest method labels

- **DeepONet query-coordinate derivatives** are **closed form** (trunk jet ×
  branch coefficients). A PDE residual such as $u_t - D u_{xx}$ costs exactly
  one trunk jet.
- **FNO derivatives** are **FFT-based / periodic**: spectral convolution on a
  regular grid. The closed-form claim does **not** transfer to FNO.
- **Residual enclosure** (`enclose_heat_residual` / `certify_heat_residual`) is a
  sound interval bound on the PDE residual over a coefficient box -- **not** a
  solution-error bound.
- **Operator accuracy** (held-out relative $L^2$) is **optimised, not proven**.

The submodule never asserts a continuum / global-regularity claim.

## Core schemas

::: omnibias.pinn.operator
    options:
      show_root_heading: false
      heading_level: 3
      members:
        - ConditioningSpec
        - OperatorSpec
        - SensorGrid
        - sample_fourier_ics
        - branch_coefficient_box
        - enclose_heat_residual
        - certify_heat_residual

## Torch drivers

::: omnibias.pinn.operator.torch
    options:
      show_root_heading: false
      heading_level: 3
      members:
        - DeepONetOperator
        - DeepONetField
        - build_deeponet
        - FNO1d
        - FNO2d
        - SpectralConv1d
        - SpectralConv2d
        - build_fno1d
        - build_fno2d
        - OperatorSlab
        - ParametricOperatorSlab
        - make_heat_slab
        - make_burgers_slab
        - make_ks_slab
        - make_parametric_heat_slab
        - make_parametric_burgers_slab
        - encode_geometry
        - probe_grid
        - data_loss
        - causal_operator_loss
        - heat_residual_loss
        - heat_residual_loss_fd
        - burgers_residual_loss
        - ks_residual_loss
        - ks_residual_loss_fd

## JAX twin

The JAX backend (`omnibias.pinn.operator.jax`) is the parity twin of the torch
surface: torch↔jax agreement is asserted at `rtol=1e-11` under
`jax.config.update("jax_enable_x64", True)` (float64 round-off, not bit-identical
bit patterns). Builder names follow the jax convention (`make_deeponet` /
`make_fno1d`).

::: omnibias.pinn.operator.jax
    options:
      show_root_heading: false
      heading_level: 3
      members:
        - DeepONetOperator
        - DeepONetField
        - make_deeponet
        - FNO1d
        - SpectralConv1d
        - make_fno1d
        - OperatorSlab
        - make_heat_slab
        - make_burgers_slab
        - make_ks_slab
        - data_loss
        - heat_residual_loss
        - heat_residual_loss_fd
        - burgers_residual_loss
        - ks_residual_loss
        - ks_residual_loss_fd

[`docs/examples/pinn_operator_learning.py`](../examples/pinn_operator_learning.py)
is the runnable smoke: order-4 closed-form exactness, the shipped KS residual
on an operator field, an FD-floor smoke, and a family residual certificate.

Status: Alpha submodule (`omnibias.pinn.operator`) of the Beta `omnibias-pinn`
package.
