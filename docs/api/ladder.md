# Hermite ladder nets (02-10)

The gaussian base carries an exact raising and lowering algebra. The
raw tower is **not** the QHO eigenbasis; Rodrigues reweight
(`Normalization.OSCILLATOR`) is required.

G1–G3/G6 are CI-gated. G4 many-body FermiNet variance (2x, five seeds)
is **leftover-recorded** unearned (leftover #21): the 1-D QHO envelope
already contains the ground state, so the named FermiNet run stays
`--full`. Exact `apply_ladder` orbital derivatives versus central FD
are **leftover-recorded** (leftover #45). G5 anharmonic honesty is
**leftover-recorded** (leftover #26): the oscillator ground Rayleigh
loses to a Dirichlet FD grid on `V = x^2/2 + x^4`; the previous
untimed `passed=True` stub is withdrawn. Cost and G5 are not in CI
`all_passed`. Status is **shipped**. See theory spec 02-10.

## Core algebra

::: omnibias.core.ladder
    options:
      show_root_heading: false
      heading_level: 3

## PyTorch module

::: omnibias.torch.architectures.ladder
    options:
      show_root_heading: false
      heading_level: 3

## JAX twin

::: omnibias.jax.architectures.ladder
    options:
      show_root_heading: false
      heading_level: 3
