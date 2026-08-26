# Linearizing transforms (02-13)

Named classical maps only: Cole-Hopf / Miura / Bäcklund / Darboux.
Exactness is **to jet truncation order N**. Spec 03-11 Lie-symmetry
*search* stays designed and is not claimed. Full 03-13 pack
birth/death stays designed.

G1 Cole-Hopf jet identity is CI-gated. G3 Burgers win is
**leftover-recorded** unearned (leftover #39): no Cole-Hopf-trained
field versus a direct PINN at matched cost. The previous
`g3_burgers_init` `passed=True` stub is withdrawn. Status is
**shipped**. See theory spec 02-13.

## Core algebra

Pointwise Cole-Hopf (`cole_hopf_u`) plus **factorial-jet** pushforward
(`cole_hopf_jet`, `factorial_jet_multiply` / `factorial_jet_reciprocal`):
heat time-jets map to Burgers jets by Cauchy product only (Taylor
recurrence, no CAS). `verify_cole_hopf_burgers_jet` seals the
plane-wave worked example. Not a Navier-Stokes claim.

::: omnibias.core.transforms_pde
    options:
      show_root_heading: false
      heading_level: 3

## PINN twins

::: omnibias.pinn.transform
    options:
      show_root_heading: false
      heading_level: 3
