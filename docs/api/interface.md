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

G1/G2/G4/G5 are CI-gated. G3 accuracy is **reported** unearned: the
smoke measures `MultiInterfaceField` on a mixed-condition three-layer
versus a linear stand-in, but named G3 needs a matched-parameter
bake-off against `PartitionedField`, `FBPINNField`, and a plain MLP
over five seeds. That training loop is not wired. Previous
`passed=True` stand-in stub withdrawn. G3 is not in CI `all_passed`.
Status is **gated**, not shipped. See theory spec 02-05.

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
