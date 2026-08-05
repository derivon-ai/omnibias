# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Distribution operators over the Gibbs path distribution ``p_beta`` (torch).

Bit-identical twin of :mod:`omnibias.struct.jax.distributions` for the differentiable ops.
For the Gibbs distribution ``p_beta(D) proportional to exp(beta * score(D))`` over the
derivations of a :class:`~omnibias.struct._core.semiring.Hypergraph`:

* :func:`path_entropy` -- the exact Shannon entropy ``H(p_beta) = beta (V_beta - E_p[score])``
  in closed form from the inside value and the edge marginals (the first-order
  expectation-semiring quantity); differentiable, ``>= 0``, ``-> log(#derivations)`` as
  ``beta -> 0`` and ``-> log(#argmax)`` as ``beta -> inf``.
* :func:`sample_paths` -- **exact** forward-filtering backward-sampling from ``p_beta`` (not a
  relaxation); its empirical edge frequencies converge to :func:`semiring_marginals`.
* :func:`gumbel_relaxed_sample` -- a Gumbel-perturb-and-MAP **approximate**, differentiable
  relaxed sample (optionally straight-through); honestly labelled, *not* an exact draw.
* :func:`topk_paths` -- **exact** k-best derivations (a hard decode) and
  :func:`topk_free_energy` -- the differentiable ``lse_beta`` restricted to those top-k scores
  (``-> `` best score as ``beta -> inf``, ``-> `` the full soft value as ``k -> #derivations``).
"""

from __future__ import annotations

import torch
from omnibias.struct._core.operators import kbest_derivations, sample_derivations
from omnibias.struct._core.semiring import Hypergraph, best_derivation
from omnibias.struct.torch._logsumexp import logsumexp_beta
from omnibias.struct.torch.semiring import semiring_marginals, semiring_value
from torch import Tensor


def path_entropy(graph: Hypergraph, edge_weights: Tensor, beta: float = 1.0) -> Tensor:
    r"""Exact Shannon entropy (nats) of ``p_beta`` over the derivations of ``graph``.

    Closed form ``H = beta (V_beta - E_p[score]) = beta (value - <marginals, weights>)`` --
    the first-order expectation-semiring entropy. Differentiable in ``edge_weights``; equals
    :func:`omnibias.struct._core.operators.brute_force_entropy` on tiny graphs.
    """
    value = semiring_value(graph, edge_weights, beta)
    marginals = semiring_marginals(graph, edge_weights, beta)
    expected_score = (marginals * edge_weights).sum()
    entropy: Tensor = beta * (value - expected_score)
    return entropy


def sample_paths(
    graph: Hypergraph,
    edge_weights: Tensor,
    beta: float = 1.0,
    num_samples: int = 1,
    *,
    seed: int | None = None,
) -> tuple[Tensor, list[tuple[int, ...]]]:
    r"""Exact forward-filtering backward-sampling of ``num_samples`` derivations from ``p_beta``.

    Returns ``(counts, samples)``: ``counts`` is a ``(num_samples, num_edges)`` tensor of
    per-edge usage counts and ``samples`` the chosen edge-index tuples. Exact (not a
    relaxation); ``counts.mean(0) -> `` :func:`semiring_marginals` as ``num_samples -> inf``.
    Sampling is not differentiable (use :func:`gumbel_relaxed_sample` for a relaxed draw).
    """
    weights = [float(x) for x in edge_weights.detach().reshape(-1).tolist()]
    counts, samples = sample_derivations(graph, weights, beta, num_samples, seed=seed)
    return torch.as_tensor(counts, dtype=edge_weights.dtype, device=edge_weights.device), samples


def gumbel_relaxed_sample(
    graph: Hypergraph,
    edge_weights: Tensor,
    beta: float = 1.0,
    *,
    seed: int | None = None,
    hard: bool = False,
) -> Tensor:
    r"""Gumbel perturb-and-MAP **approximate** relaxed sample (differentiable), as edge scores.

    Adds i.i.d. Gumbel noise to the edge weights and returns the tower-softmax edge selection
    :func:`semiring_marginals` of the perturbed weights; with ``hard=True`` a straight-through
    estimator returns the argmax-derivation indicator in the forward pass and the relaxed
    gradient in the backward. This is a per-edge perturbation, **not** an exact draw from
    ``p_beta`` (use :func:`sample_paths` for that) -- honestly labelled as a relaxation.
    """
    generator = torch.Generator(device=edge_weights.device)
    if seed is not None:
        generator.manual_seed(seed)
    u = torch.rand(edge_weights.shape, dtype=edge_weights.dtype, device=edge_weights.device, generator=generator)
    gumbel = -torch.log(-torch.log(u))
    perturbed = edge_weights + gumbel.detach()
    relaxed: Tensor = semiring_marginals(graph, perturbed, beta)
    if not hard:
        return relaxed
    _score, deriv = best_derivation(graph, [float(x) for x in perturbed.detach().reshape(-1).tolist()])
    hard_ind = torch.zeros_like(edge_weights)
    for e in deriv:
        hard_ind[e] = hard_ind[e] + 1.0
    straight_through: Tensor = hard_ind + (relaxed - relaxed.detach())
    return straight_through


def topk_paths(
    graph: Hypergraph, edge_weights: Tensor, k: int
) -> list[tuple[float, tuple[int, ...]]]:
    r"""Exact ``k``-best derivations ``(score, edge-tuple)`` by score, descending (hard decode).

    Wraps the exact topological k-best DP; not differentiable (the argmax set is a hard
    decode). For a smoothed objective over these paths use :func:`topk_free_energy`.
    """
    weights = [float(x) for x in edge_weights.detach().reshape(-1).tolist()]
    result: list[tuple[float, tuple[int, ...]]] = kbest_derivations(graph, weights, k)
    return result


def topk_free_energy(graph: Hypergraph, edge_weights: Tensor, k: int, beta: float = 1.0) -> Tensor:
    r"""Differentiable ``lse_beta`` restricted to the top-``k`` derivation scores.

    The top-``k`` *set* is chosen by exact hard k-best on the detached weights (piecewise
    constant in the weights); the returned free energy is differentiable through the ``k``
    selected derivations' scores. ``-> `` the best score as ``beta -> inf`` and ``-> `` the
    full :func:`semiring_value` as ``k -> #derivations`` (monotone non-decreasing in ``k``).
    """
    weights = [float(x) for x in edge_weights.detach().reshape(-1).tolist()]
    kbest = kbest_derivations(graph, weights, k)
    scores = torch.stack(
        [
            torch.stack([edge_weights[e] for e in deriv]).sum() if deriv else edge_weights.new_zeros(())
            for _score, deriv in kbest
        ]
    )
    free_energy: Tensor = logsumexp_beta(scores, beta, axis=-1)
    return free_energy


__all__ = [
    "gumbel_relaxed_sample",
    "path_entropy",
    "sample_paths",
    "topk_free_energy",
    "topk_paths",
]
