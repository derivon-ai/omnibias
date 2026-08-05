# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Entropy, exact sampling, and top-k over the Gibbs path distribution -- omnibias-struct.

Run:

    pip install "omnibias-struct[torch,jax]"
    python docs/examples/path_distributions.py

Any DP relaxation induces a Gibbs distribution ``p_beta(path) proportional to
exp(beta * score(path))`` over the (exponentially many) derivations of its hypergraph. The
distribution operators read exact quantities off that distribution, all on the shared
semiring driver, keeping the two axes apart (``beta -> inf`` tempers the distribution;
``delta -> 0`` differentiates the smooth ones exactly):

* ``path_entropy`` -- the exact Shannon entropy ``H(p_beta) = beta (V_beta - E[score])``
  (closed form from the inside value and the edge marginals); differentiable.
* ``sample_paths`` -- **exact** forward-filtering backward-sampling; the empirical edge
  frequencies converge to the closed-form marginals (no relaxation).
* ``topk_paths`` -- **exact** k-best decode; ``topk_free_energy`` -- the differentiable
  ``lse_beta`` restricted to those top-k scores.

The DP here is shortest/longest path over a small routing DAG lifted onto the driver with
``from_dag`` -- but the operators are generic (they run identically on the CKY, Eisner, and
CTC hypergraphs).
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from omnibias.struct import (  # noqa: E402
    DAG,
    brute_force_entropy,
    brute_force_kbest,
    count_derivations,
    from_dag,
)
from omnibias.struct.jax import path_entropy as path_entropy_jax  # noqa: E402
from omnibias.struct.jax import topk_free_energy as topk_free_energy_jax  # noqa: E402
from omnibias.struct.torch import (  # noqa: E402
    path_entropy,
    sample_paths,
    semiring_marginals,
    topk_free_energy,
    topk_paths,
)

torch.set_default_dtype(torch.float64)

# A small routing DAG: several source(0) -> sink(5) paths through a diamond lattice.
_SCORES = {
    (0, 1): 0.6, (0, 2): 0.2,
    (1, 2): 0.5, (1, 3): 0.9, (2, 3): 0.3, (2, 4): 0.7,
    (3, 4): 0.4, (3, 5): 0.8, (4, 5): 0.5,
}
DAG_ = DAG(num_nodes=6, edges=_SCORES, source=0, sink=5)
GRAPH, EDGE_INDEX = from_dag(DAG_)


def _weights() -> np.ndarray:
    w = np.zeros(GRAPH.num_edges)
    for (u, v), score in _SCORES.items():
        w[EDGE_INDEX[(u, v)]] = score
    return w


def entropy_demo() -> None:
    print("=== 1. path entropy H(p_beta) = beta (V - E[score]) == brute force ===")
    w = _weights()
    n = count_derivations(GRAPH)
    wt = torch.tensor(w, requires_grad=True)
    print(f"  routing DAG: {n} source->sink paths")
    print(f"  {'beta':>6s} {'H (torch)':>11s} {'H (brute)':>11s} {'H (jax)':>11s}")
    for beta in (0.25, 1.0, 4.0, 16.0):
        h = path_entropy(GRAPH, wt, beta)
        h_bf = brute_force_entropy(GRAPH, w, beta)
        h_j = float(path_entropy_jax(GRAPH, jax.numpy.asarray(w), beta))
        print(f"  {beta:6.2f} {float(h.detach()):11.6f} {h_bf:11.6f} {h_j:11.6f}")
        assert abs(float(h.detach()) - h_bf) < 1e-9 and abs(h_j - h_bf) < 1e-9
    (grad,) = torch.autograd.grad(path_entropy(GRAPH, wt, 2.0), wt)
    assert torch.isfinite(grad).all()
    print("  entropy is differentiable; falls from log(N) (hot) toward 0 (cold, unique best).\n")


def sampling_demo() -> None:
    print("=== 2. exact forward-filtering backward-sampling -> closed-form marginals ===")
    w = _weights()
    wt = torch.tensor(w)
    beta = 3.0
    counts, samples = sample_paths(GRAPH, wt, beta, 20000, seed=0)
    empirical = counts.mean(0).numpy()
    closed_form = semiring_marginals(GRAPH, wt, beta).numpy()
    err = float(np.max(np.abs(empirical - closed_form)))
    print(f"  drew {len(samples)} exact samples; max|empirical - closed-form marginal| = {err:.4f}")
    assert err < 0.02  # Monte-Carlo error ~ 1/sqrt(20000)
    print("  Reading: the sampler is exact (no relaxation); frequencies converge to marginals.\n")


def topk_demo() -> None:
    print("=== 3. exact k-best decode + differentiable top-k free energy ===")
    w = _weights()
    wt = torch.tensor(w)
    kb = topk_paths(GRAPH, wt, 3)
    bk = brute_force_kbest(GRAPH, w, 3)
    print("  top-3 path scores (k-best):", [round(s, 4) for s, _ in kb])
    assert [round(s, 10) for s, _ in kb] == [round(s, 10) for s, _ in bk]
    n = count_derivations(GRAPH)
    beta = 4.0
    energies = [float(topk_free_energy(GRAPH, wt, k, beta)) for k in range(1, n + 1)]
    print(f"  top-k free energy k=1..N: {[round(e, 4) for e in energies]}")
    assert all(energies[i] <= energies[i + 1] + 1e-12 for i in range(len(energies) - 1))
    parity = abs(energies[2] - float(topk_free_energy_jax(GRAPH, jax.numpy.asarray(w), 3, beta)))
    print(f"  torch<->jax top-k free-energy parity (k=3): {parity:.2e}")
    assert parity < 1e-10
    print("  Reading: k-best is a hard decode; the free energy smoothly interpolates best->full.\n")


def main() -> None:
    entropy_demo()
    sampling_demo()
    topk_demo()
    print("OK: path entropy == brute force; sampling -> closed-form marginals; k-best exact; "
          "top-k free energy monotone; parity < 1e-10.")


if __name__ == "__main__":
    main()
