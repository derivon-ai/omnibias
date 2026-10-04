# Derivatives for PINNs

omnibias provides closed-form activation derivatives, differentiable Taylor
jets and field operators. Supported neural networks can evaluate high-order
spatial derivatives without constructing a nested autodiff graph for each
order. Parameter training still uses the selected framework's autodiff.

1. [Build a PINN residual](pinn.md): fourth derivatives, boundary conditions and training.
2. [Choose a derivative API](derivatives.md): directional and mixed jets, torch and JAX.
3. [Understand the guarantees](guarantees.md): supported models, precision and complexity.
4. [Find a primitive](packages.md): the 16 distributions and their public entry points.

The monorepo contains reusable infrastructure. Consumer applications and
solvers live under the sibling `omnibias_projects` directory.
