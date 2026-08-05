# omnibias-keras

Keras 3 unified backend for omnibias. The same code runs on TensorFlow,
JAX, or PyTorch via `keras.ops`; closed-form derivative towers share the
polynomial coefficients of the torch and JAX backends through
`omnibias.core.polynomials`, so all backends are bit-identical by
construction.

Select the Keras backend with the `KERAS_BACKEND` environment variable
(`tensorflow` | `jax` | `torch`) *before importing keras*.

## Top-level API

::: omnibias.keras
    options:
      show_root_heading: false
      heading_level: 3

## Activation registry

::: omnibias.keras.activations.registry
    options:
      show_root_heading: false
      heading_level: 3

## OperatorMultiBiasUnit

::: omnibias.keras.unit
    options:
      show_root_heading: false
      heading_level: 3

## Blocks

::: omnibias.keras.blocks
    options:
      show_root_heading: false
      heading_level: 3

## Piecewise & tempered activations

The hard almost-everywhere family and the smooth beta-tempered surrogate
family (see the [activation dictionary](../activations.md)).

::: omnibias.keras.activations.piecewise
    options:
      show_root_heading: false
      heading_level: 3

::: omnibias.keras.activations.tempered
    options:
      show_root_heading: false
      heading_level: 3

## Learnable-temperature blocks

::: omnibias.keras.tempered_blocks
    options:
      show_root_heading: false
      heading_level: 3

## Growable units

::: omnibias.keras.growable
    options:
      show_root_heading: false
      heading_level: 3

## Training utilities

::: omnibias.keras.training.k_scheduler
    options:
      show_root_heading: false
      heading_level: 3

## Fastpath kernels

::: omnibias.keras.fastpath
    options:
      show_root_heading: false
      heading_level: 3
