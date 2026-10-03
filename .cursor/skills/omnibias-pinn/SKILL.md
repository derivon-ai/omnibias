---
name: omnibias-pinn
description: Build a PINN residual on omnibias derivative primitives, with explicit coordinates, boundary conditions and differentiable parameter training.
---

1. Start with `docs/pinn.md` and `docs/derivatives.md`; inspect the real API
   signatures. Put application and solver code in its consumer repository.
2. Write the coordinate order, PDE residual, derivative orders, domain and
   boundary/initial conditions explicitly. Choose a manufactured solution as
   the independent test oracle before training.
3. Use directional jets for selected directional derivatives and multivariate
   jets when mixed partials are needed. Convert Taylor coefficients to raw
   derivatives with `jet_to_tower` or `jet_partials`.
4. Keep residuals differentiable in parameters; evaluate collocation and
   boundary losses with the same network. Backpropagate the loss once.
5. Check derivative values and parameter gradients, then train. Report held-out
   residual and solution errors separately from optimization loss.
6. Benchmark time and memory at fixed accuracy, dtype, batch size and derivative
   order. Exact derivative formulas do not certify training convergence or a
   continuous-domain residual bound.

For field objects use `omnibias.fields` and its dispatch contract; the existing
solver consumer is `../omnibias_projects/omnibias-pinn/`.
