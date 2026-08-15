# omnibias-graph

> **Differentiable spectral graph operators + continuous combinatorial relaxations, with torch + jax bit-parity.**

`omnibias-graph` provides two families of differentiable operators on graphs and
orderings, each with bit-identical torch and jax backends:

- **Spectral graph ops** — the combinatorial `L = D - A`, symmetric normalized
  `L_sym = I - D^{-1/2} A D^{-1/2}`, and random-walk `L_rw = I - D^{-1} A`
  Laplacians; Laplacian-eigenmaps `spectral_embedding`; the graph heat kernel
  `exp(-t L)`; and the Rayleigh-Ritz `spectral_clustering_relaxation` of the
  ratio / normalized cut.
- **Differentiable relaxations** — `sinkhorn_normalize` (projection onto
  doubly-stochastic matrices), `gumbel_sinkhorn` (relaxed permutation /
  assignment), `soft_sort` (SoftSort), and `soft_top_k`. Each has a temperature
  `tau` that recovers the exact discrete object as `tau -> 0`.

## Status

**Alpha (0.1.0a1).** API may shift between alpha releases.

Gated Face-Net (`omnibias.graph.arrangement`) message-passes on a **sampled
subgraph** of an arrangement tope graph. `beta -> inf` is temperature
collapse, not founding `delta -> 0`. Sound gap, not P vs NP.

## Install

```bash
pip install omnibias-graph[torch]   # PyTorch backend
pip install omnibias-graph[jax]     # JAX backend
pip install omnibias-graph[all]     # both
```

## 30-second tour

```python
import torch
import omnibias.graph.torch.ops as G

# Cycle graph C_8: Laplacian spectrum is 2 - 2 cos(2 pi k / 8).
n = 8
A = torch.zeros(n, n, dtype=torch.float64)
for i in range(n):
    A[i, (i + 1) % n] = A[i, (i - 1) % n] = 1.0

evals, _ = G.laplacian_spectrum(A)          # exact ring eigenvalues
H = G.graph_heat_kernel(A, t=0.5)           # exp(-tL), row-sums preserved
emb = G.spectral_embedding(A, n_components=2)  # Fiedler + next eigenmap

scores = torch.tensor([3.0, 1.0, 2.0, 5.0, 4.0])
sorted_soft = G.soft_sort(scores, temperature=1e-3)   # -> descending sort
mask = G.soft_top_k(scores, k=2, temperature=1e-3)    # -> {5, 4} indicator, sum == 2
```

The `omnibias.graph.jax.ops` namespace mirrors this surface; `tests/` assert
cross-backend parity (`rtol=1e-9`, float64).

## Scope & honesty

These are **differentiable relaxations** and **smooth spectral** quantities, not
exact combinatorial solvers. Exact NP-hard problems (TSP, exact max-cut, exact
graph isomorphism, SAT / ILP) are **out of scope**, documented in
[`docs/cookbook/graph-limitation.md`](../../docs/cookbook/graph-limitation.md)
and guarded by an enforcement test. A relaxed cut value is a lower bound on the
discrete optimum; rounding the relaxed embedding to a discrete partition is the
caller's job and is where combinatorial hardness enters.

The rigorous cross-checks reuse `omnibias.core.verified` (interval eigenvalue
enclosures for the ring-graph oracle).

## License

Apache-2.0. See [`LICENSE`](LICENSE) and [`../../LICENSING.md`](../../LICENSING.md).
You never need a commercial licence for this package.
