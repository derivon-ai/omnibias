---
name: omnibias-backends
description: Extend omnibias activation derivatives or torch/JAX/Keras kernels while preserving coefficient, gradient and parity contracts.
---

1. Read `AGENTS.md` and the affected module; inspect its existing tests.
2. Put coefficients in `omnibias.core.polynomials`, combinatorics in
   `omnibias.core.bell` or `omnibias.core.multi_index`. Keep core pure Python.
3. Update paired torch/JAX implementations together. Keras activation kernels
   use the same core coefficients through `keras.ops`.
4. Preserve Taylor normalization (`f^(k)/k!` or `D^alpha f/alpha!`), parameter
   gradients, JAX tracing, dtype and accumulation order. Request only necessary
   derivatives; full mixed jets have `binomial(dim + order, order)` coefficients.
5. Compare new behavior against an analytic oracle and low-order autodiff;
   verify parameter gradients and existing cross-backend tolerances.
6. Run affected package tests, root parity tests, Ruff and applicable strict typing.

API map: `docs/derivatives.md`. Do not infer performance or universal bit
identity from shared coefficients; use the actual dtype/device and a measured
baseline. Use `get_activation(...).fastpath` for one derivative, `mlp_jet` for
one direction and `mlp_jet_mv` for all requested mixed partials.
