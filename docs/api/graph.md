# omnibias-graph

Differentiable **spectral graph operators** and **continuous combinatorial
relaxations**, bit-identical across the torch and jax backends.

The spectral layer is exact linear algebra on a weighted adjacency matrix: the
combinatorial / normalized / random-walk Laplacians, differentiable Laplacian
eigenmaps (spectral embedding), the graph heat kernel `exp(-t L)`, and the
Rayleigh-Ritz eigenvector relaxation of the ratio / normalized cut. The
relaxation layer turns discrete objects into smooth, temperature-controlled
surrogates: the Sinkhorn projection onto doubly-stochastic matrices,
Gumbel-Sinkhorn permutation matrices, SoftSort differentiable sorting, and a
soft top-k operator. Every relaxation recovers its hard combinatorial object as
the temperature `tau -> 0`.

!!! warning "Scope: differentiable relaxations, not exact combinatorial solvers"
    These operators are *continuous relaxations* and *smooth spectral*
    quantities. Exact NP-hard combinatorial solving — the travelling-salesman
    tour, exact (weighted) max-cut, exact graph isomorphism, SAT / ILP — is
    **out of scope** and guarded by an enforcement test. See the
    [graph-limitation cookbook](../cookbook/graph-limitation.md) and
    [scope & guarantees](../scope-and-guarantees.md) §6. A relaxed cut value is a
    *lower bound* on the discrete optimum; the discrete rounding step is the
    caller's responsibility and is where combinatorial hardness lives.

## Oracles

* **Ring graph** `C_n`: the combinatorial-Laplacian spectrum is
  `lambda_k = 2 - 2 cos(2 pi k / n)`, each eigenpair certified with an
  `omnibias.core.verified` interval enclosure that brackets a true eigenvalue.
* **Two-block SBM**: the Fiedler vector (second-smallest eigenvector) separates
  the planted blocks by sign.
* **SoftSort / soft top-k**: recover `torch.sort` / the hard top-k mask as
  `tau -> 0`; the soft top-k weights sum to exactly `k` at any temperature.

## Ops (torch)

::: omnibias.graph.torch.ops
    options:
      show_root_heading: false
      heading_level: 3

## JAX twin

The JAX backend (`omnibias.graph.jax.ops`) is the bit-identical twin. Laplacians,
eigenvalues, heat kernels, and every relaxation match the torch backend to
`rtol=1e-9` in float64 (cross-backend parity tests). Raw eigen*vectors* of a
degenerate spectrum are only defined up to a rotation within each eigenspace, so
parity is asserted on the eigenvalues, the heat kernel, and the invariant
subspace projector rather than on individual eigenvectors.

Status: Alpha (`0.1.0a1`).
