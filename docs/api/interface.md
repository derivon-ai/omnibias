# Multi-interface transmission PINN (02-05)

Parallel interfaces with per-interface condition order (value / flux /
curvature). `alpha -> inf` is **interface sharpening**, neither bias
collapse (`delta -> 0`) nor temperature collapse (`beta -> inf`).
Conditions hold to a **stated smoothing tolerance**, not exactly, at
finite `alpha`. Only parallel interfaces are in scope.

Import `Interface` from `omnibias.pinn.interface` (alias
`TransmissionInterface`). Do **not** import
`omnibias.pinn._core.interface.Interface` — that is the XPINN penalty
glue between subnetworks.

G1/G2/G5 are CI-gated. G3 accuracy is **leftover-recorded** unearned
(leftover #37): the smoke measures `MultiInterfaceField` on a
mixed-condition three-layer versus a linear stand-in, but named G3
needs a matched-parameter bake-off against `PartitionedField`,
`FBPINNField`, and a plain MLP over five seeds. G4 hard versus
penalized is **leftover-recorded** unearned (leftover #38): the smoke
measures a hard residual versus a zero-coeff soft field, but named G4
needs an equal-budget train. Those training loops are not wired.
Previous `passed=True` stand-in stubs withdrawn. G3 and G4 are not in
CI `all_passed`. Status is **gated**, not shipped. See theory spec
02-05.

## Algebra

::: omnibias.pinn.interface._core
    options:
      show_root_heading: false
      heading_level: 3

## PyTorch field

::: omnibias.pinn.interface.torch
    options:
      show_root_heading: false
      heading_level: 3

## JAX twin

::: omnibias.pinn.interface.jax
    options:
      show_root_heading: false
      heading_level: 3
