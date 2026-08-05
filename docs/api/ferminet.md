# omnibias-ferminet

FermiNet bridge for omnibias.

## folx-compatible API

::: omnibias.ferminet.folx_compat
    options:
      show_root_heading: false
      heading_level: 3

## Production bridge

::: omnibias.ferminet.integration
    options:
      show_root_heading: false
      heading_level: 3

## Restricted Tier-2 ansatz

::: omnibias.ferminet.restricted
    options:
      show_root_heading: false
      heading_level: 3

## Padé-Jastrow correlation factor

A closed-form symmetric Padé-Jastrow factor :math:`\exp(J)` that is *additive*
to :math:`\log|\psi|`. `jastrow_value_grad_laplacian` returns the analytic value,
gradient, and Laplacian of :math:`J`; the default cusp slopes are the physical
electron-electron (:math:`\tfrac12` antiparallel, :math:`\tfrac14` parallel) and
electron-nucleus (:math:`-Z`) values. `jastrow_slater_local_kinetic_energy`
combines the Jastrow derivatives with a Slater determinant's kinetic term through
the refactored `tier2_grad_laplacian_log_psi`, correctly accounting for the
cross term in :math:`\lVert\nabla\log|\psi|\rVert^2`.

::: omnibias.ferminet.jastrow
    options:
      show_root_heading: false
      heading_level: 3
