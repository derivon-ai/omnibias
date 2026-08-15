# Hermite ladder nets (02-10)

The gaussian base carries an exact raising and lowering algebra. The
raw tower is **not** the QHO eigenbasis; Rodrigues reweight
(`Normalization.OSCILLATOR`) is required. Anharmonic G5 may lose;
report it.

G1–G3/G6 are CI-gated. G4 FermiNet is `--full`. Status is **gated**,
not shipped. See theory spec 02-10.

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
