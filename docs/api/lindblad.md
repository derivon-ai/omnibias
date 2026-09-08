# Open-system Lindblad dynamics (09-32)

A time-independent Gorini–Kossakowski–Sudarshan–Lindblad generator is a
linear semigroup: `rho(t) = exp(t L) rho(0)`, and
`d^n rho / dt^n = L^n rho(t)` from **one** propagator evaluation plus
matrix powers. That is the *linear-semigroup* analogue of the sigma
tower -- one transcendental evaluation regardless of order -- and is
labelled as such, not as the activation tower. Only the qubit Bloch
family and pure dephasing are closed form; general finite `d` is a
certified numerical propagator.

The verified path realifies the complex superoperator once
(`[[Re M, -Im M], [Im M, Re M]]`) and reuses the shipped interval
linear-algebra stack: `interval_matrix_exp`, QR-Lohner flow, and
realified LDL^T positivity. The thermal-qubit steady excited population
is the shipped Fermi occupancy
`occupancy(FermiModel(beta, mu=0), omega)`.

`relaxation` is the seventh named collapse: parameter `relaxation_rate`,
limit `inf`, surviving object `steady_state`. On a pure-dephasing model
every diagonal state is fixed, so einselection can `PROVE` while
relaxation must refuse -- two collapses that disagree on a shared input
are not rebrands.

Two collapse senses are named and must not be conflated: the founding
**bias collapse** (`delta -> 0`) never appears here, and `beta -> inf`
is the founding **temperature collapse** (feasibility sense) of the
thermal occupancy step, not a new slot. The GKSL form is a caller
input, not a Born–Markov derivation.

Status is **shipped**. G1–G7 are CI-gated (`benchmarks/lindblad.py`).
See theory spec
[09-32](https://github.com/derivon-ai/omnibias/blob/main/theory/09-inventions/32-open-system-lindblad-dynamics.md).

Homes: `omnibias.core.lindblad`, `omnibias.core.verified.lindblad`,
`omnibias.core.collapse.relaxation`, plus bit-identical twins
`omnibias.torch.lindblad` and `omnibias.jax.lindblad`, and a qpinn
residual / hard PSD cage.

## Core algebra

::: omnibias.core.lindblad
    options:
      show_root_heading: false
      heading_level: 3

## Verified twin

::: omnibias.core.verified.lindblad
    options:
      show_root_heading: false
      heading_level: 3

## Relaxation collapse

::: omnibias.core.collapse.relaxation
    options:
      show_root_heading: false
      heading_level: 3

## PyTorch twin

::: omnibias.torch.lindblad
    options:
      show_root_heading: false
      heading_level: 3

## JAX twin

::: omnibias.jax.lindblad
    options:
      show_root_heading: false
      heading_level: 3
