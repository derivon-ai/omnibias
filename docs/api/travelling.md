# Soliton tanh-method nets (02-09)

Polynomials in `tanh` are the classical travelling-wave ansatz. This
is tanh **algebra**, not a collapse. A multi-kink sum is not the
n-soliton formula (that is [transforms_pde.md](transforms_pde.md)).

G1/G2/G3/G5 are CI-gated (published tanh-class list, exact rational
zeros, 1e-14 residual, negative control). G4 PINN init-win (5× fewer
steps vs cold start, five seeds) is **leftover-recorded** unearned
(leftover #22): no training loop is wired, so the named 5-seed
step-count win stays `--full`. Algebraic `verify_exact` /
`solve_ansatz` wall plus published-vs-cold residual L1 are
**reported**; the previous untimed `passed=True` stub is withdrawn.
Cost is not in CI `all_passed`. Status is **gated**, not shipped. See
theory spec 02-09.

## Core algebra

::: omnibias.core.tanh_method
    options:
      show_root_heading: false
      heading_level: 3

## PINN twins

::: omnibias.pinn.travelling
    options:
      show_root_heading: false
      heading_level: 3
