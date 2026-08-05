# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Differentiable soft dynamic-programming layers (torch).

Bit-identical twin of :mod:`omnibias.struct.jax.soft_dp` (float64). Each ``soft_*``
replaces the hard ``max`` / ``min`` combine of its DP recursion with ``lse_beta``
(:mod:`omnibias.struct.torch._logsumexp`), unrolled so ``autograd`` flows through it. The
``beta -> inf`` annealing is the *feasibility / temperature* collapse (a soft object
hardened to a discrete one, the same axis as ``omnibias-discrete`` / ``-qubo`` /
``-routing``); it is differentiated exactly by the ``delta -> 0``
founding bias collapse tower -- do not conflate the two axes. The ``*_marginals`` functions return
the **closed-form** gradient -- the forward-backward path marginal assembled from the
tower's softmax -- which the tests pin equal to ``autograd``.

**Numerical envelope (measured).** Use float64: the soft *value* stays bit-stable out to
``beta ~ 1e12`` and the closed-form marginals are accurate to ``~1e-9`` for ``beta`` up to
about ``1e6``, degrading gracefully (never ``NaN``) beyond. The unreachable-cell sentinel
``_NEG`` is safe in float64 (``beta * _NEG`` never overflows for practical ``beta``);
float32 breaks both bit-identicality and large-``beta`` stability. The ``*_batched``
helpers map a leading batch dimension through ``torch.func.vmap`` and are bit-identical to
looping the per-example layer.
"""

from __future__ import annotations

import torch
from omnibias.struct._core.trellis import DAG, CTCLattice
from omnibias.struct.torch._logsumexp import logsumexp_beta
from torch import Tensor

_NEG = -1.0e30  # finite sentinel for unreachable CTC lattice cells (avoids -inf grads)


# ---------------------------------------------------------------------------
# Viterbi (linear-chain, max-plus)
# ---------------------------------------------------------------------------


def _chain_forward(emissions: Tensor, transitions: Tensor, start: Tensor, beta: float) -> Tensor:
    alpha_rows = [start + emissions[0]]
    for t in range(1, emissions.shape[0]):
        scores = alpha_rows[-1][:, None] + transitions  # (S_prev, S)
        alpha_rows.append(emissions[t] + logsumexp_beta(scores, beta, axis=0))
    return torch.stack(alpha_rows, dim=0)  # (T, S)


def _chain_backward(emissions: Tensor, transitions: Tensor, beta: float) -> Tensor:
    r"""Backward soft cost-to-go ``B[t, s]`` (excludes the current-cell emission ``E[t, s]``)."""
    bwd_rows: list[Tensor] = [torch.zeros_like(emissions[0])]
    for t in range(emissions.shape[0] - 1, 0, -1):
        nxt = emissions[t] + bwd_rows[0]  # (S',)
        mat = transitions + nxt[None, :]  # (S, S')
        bwd_rows.insert(0, logsumexp_beta(mat, beta, axis=1))
    return torch.stack(bwd_rows, dim=0)  # (T, S)


def _resolve_start(start: Tensor | None, n_states: int, ref: Tensor) -> Tensor:
    if start is None:
        return torch.zeros(n_states, dtype=ref.dtype, device=ref.device)
    return torch.as_tensor(start, dtype=ref.dtype, device=ref.device)


def soft_viterbi(
    emissions: Tensor,
    transitions: Tensor,
    beta: float = 1.0,
    *,
    start: Tensor | None = None,
) -> Tensor:
    r"""Soft Viterbi value ``beta^-1 log sum_paths exp(beta score)`` of a linear chain.

    ``emissions`` is ``(T, S)``, ``transitions`` is ``(S, S)``, optional ``start`` is
    ``(S,)``. Differentiable in all three; ``-> max_path score`` as ``beta -> inf``.
    """
    start_t = _resolve_start(start, emissions.shape[1], emissions)
    alpha = _chain_forward(emissions, transitions, start_t, beta)
    return logsumexp_beta(alpha[-1], beta, axis=-1)


def soft_viterbi_marginals(
    emissions: Tensor,
    transitions: Tensor,
    beta: float = 1.0,
    *,
    start: Tensor | None = None,
) -> Tensor:
    r"""Closed-form state marginals ``gamma[t, s] = P_beta(state s at time t)`` (``(T, S)``).

    Forward-backward with the tower softmax; equals ``d soft_viterbi / d emissions`` and
    sums to ``1`` along the state axis. As ``beta -> inf`` it concentrates on the Viterbi
    path.
    """
    start_t = _resolve_start(start, emissions.shape[1], emissions)
    alpha = _chain_forward(emissions, transitions, start_t, beta)  # (T, S)
    bwd = _chain_backward(emissions, transitions, beta)  # (T, S)
    value = logsumexp_beta(alpha[-1], beta, axis=-1)
    return torch.exp(beta * (alpha + bwd - value))


def soft_viterbi_transition_marginals(
    emissions: Tensor,
    transitions: Tensor,
    beta: float = 1.0,
    *,
    start: Tensor | None = None,
) -> Tensor:
    r"""Closed-form transition-pair marginals ``xi[t, s, s']`` (``(T - 1, S, S)``).

    ``xi[t, s, s'] = P_beta(state s at time t, state s' at time t + 1)``. Each ``(S, S)``
    slab sums to ``1``, and summing over ``t`` gives ``d soft_viterbi / d transitions`` --
    the transition analogue of :func:`soft_viterbi_marginals`. As ``beta -> inf`` it
    concentrates on the consecutive-state pairs of the Viterbi path.
    """
    start_t = _resolve_start(start, emissions.shape[1], emissions)
    n_steps = emissions.shape[0]
    alpha = _chain_forward(emissions, transitions, start_t, beta)  # (T, S)
    bwd = _chain_backward(emissions, transitions, beta)  # (T, S)
    value = logsumexp_beta(alpha[-1], beta, axis=-1)
    xis: list[Tensor] = []
    for t in range(1, n_steps):
        # alpha[t-1][s] + A[s, s'] + (E[t][s'] + B[t][s'])
        term = alpha[t - 1][:, None] + transitions + (emissions[t] + bwd[t])[None, :]
        xis.append(torch.exp(beta * (term - value)))
    return torch.stack(xis, dim=0)  # (T - 1, S, S)


# ---------------------------------------------------------------------------
# Shortest path (DAG, min-plus) -- certified in the max convention (score = -cost)
# ---------------------------------------------------------------------------


def _dag_forward(weights: Tensor, dag: DAG, beta: float) -> list[Tensor | None]:
    alpha: list[Tensor | None] = [None] * dag.num_nodes
    alpha[dag.source] = torch.zeros((), dtype=weights.dtype, device=weights.device)
    for v in range(dag.num_nodes):
        if v == dag.source:
            continue
        preds = [u for u in dag.incoming(v) if alpha[u] is not None]
        if not preds:
            continue
        vals = torch.stack([alpha[u] - weights[u, v] for u in preds])  # type: ignore[operator]
        alpha[v] = logsumexp_beta(vals, beta, axis=-1)
    return alpha


def _dag_backward(weights: Tensor, dag: DAG, beta: float) -> list[Tensor | None]:
    succ: dict[int, list[int]] = {}
    for u, v in dag.edges:
        succ.setdefault(u, []).append(v)
    for vs in succ.values():
        vs.sort()
    bwd: list[Tensor | None] = [None] * dag.num_nodes
    bwd[dag.sink] = torch.zeros((), dtype=weights.dtype, device=weights.device)
    for v in range(dag.num_nodes - 1, -1, -1):
        if v == dag.sink:
            continue
        succs = [w for w in succ.get(v, []) if bwd[w] is not None]
        if not succs:
            continue
        vals = torch.stack([bwd[w] - weights[v, w] for w in succs])  # type: ignore[operator]
        bwd[v] = logsumexp_beta(vals, beta, axis=-1)
    return bwd


def soft_shortest_path(weights: Tensor, dag: DAG, beta: float = 1.0) -> Tensor:
    r"""Soft (softmin) shortest-path cost of a :class:`DAG` for edge ``weights`` ``(n, n)``.

    Returns ``-beta^-1 log sum_paths exp(-beta cost)`` (softmin over ``source -> sink``
    path costs); ``-> min cost`` as ``beta -> inf``. Differentiable in the finite edge
    entries of ``weights``. Certify in the ``max`` convention via the negated value
    (``V* = -min cost``, ``V_beta = -soft cost``).
    """
    alpha = _dag_forward(weights, dag, beta)
    sink_value = alpha[dag.sink]
    if sink_value is None:
        raise ValueError("sink is not reachable from source in this DAG")
    return -sink_value


def soft_shortest_path_marginals(weights: Tensor, dag: DAG, beta: float = 1.0) -> Tensor:
    r"""Closed-form edge-usage marginals ``xi[u, v] = P_beta(edge (u, v) on the path)``.

    An ``(n, n)`` tensor, zero off the edge set. Equals ``d soft_shortest_path / d
    weights`` on the edges; forward-backward with the tower softmax. As ``beta -> inf``
    it concentrates on the shortest path.
    """
    alpha = _dag_forward(weights, dag, beta)
    bwd = _dag_backward(weights, dag, beta)
    value = alpha[dag.sink]
    if value is None:
        raise ValueError("sink is not reachable from source in this DAG")
    xi = torch.zeros_like(weights)
    for u, v in dag.edges:
        au, bv = alpha[u], bwd[v]
        if au is None or bv is None:
            continue
        xi[u, v] = torch.exp(beta * (au - weights[u, v] + bv - value))
    return xi


# ---------------------------------------------------------------------------
# CTC (blank-augmented alignment lattice)
# ---------------------------------------------------------------------------


def soft_ctc(log_probs: Tensor, lattice: CTCLattice, beta: float = 1.0) -> Tensor:
    r"""Soft CTC value ``beta^-1 log sum_alignments exp(beta score)`` for ``log_probs`` ``(T, C)``.

    At ``beta = 1`` this is the standard CTC log-likelihood; as ``beta -> inf`` it anneals
    to the best single-alignment score. Differentiable in ``log_probs`` (autograd through
    the unrolled ``lse_beta`` lattice recursion; its gradient is the alignment marginal).
    """
    ext = lattice.extended_labels()
    m = int(ext.shape[0])
    n_steps = log_probs.shape[0]
    neg = torch.full((), _NEG, dtype=log_probs.dtype, device=log_probs.device)
    f: list[Tensor] = []
    for s in range(m):
        if s == 0:
            f.append(log_probs[0, int(ext[0])])
        elif s == 1:
            f.append(log_probs[0, int(ext[1])])
        else:
            f.append(neg)
    for t in range(1, n_steps):
        nf: list[Tensor] = []
        for s in range(m):
            vals = torch.stack([f[p] for p in lattice.incoming(s)])
            nf.append(log_probs[t, int(ext[s])] + logsumexp_beta(vals, beta, axis=-1))
        f = nf
    ends = [f[m - 1]] + ([f[m - 2]] if m >= 2 else [])
    return logsumexp_beta(torch.stack(ends), beta, axis=-1)


def soft_ctc_marginals(log_probs: Tensor, lattice: CTCLattice, beta: float = 1.0) -> Tensor:
    r"""Closed-form CTC label marginals ``d soft_ctc / d log_probs`` (``(T, C)``).

    Forward-backward over the blank-augmented lattice with the tower softmax; entry
    ``[t, c]`` is the total probability mass of alignments emitting class ``c`` at time
    ``t``, so each row sums to ``1``. Equals ``autograd`` of :func:`soft_ctc` and, as
    ``beta -> inf``, concentrates on the best alignment (:func:`ctc_best_alignment`).
    """
    lp = log_probs
    ext = lattice.extended_labels()
    m = int(ext.shape[0])
    n_steps = lp.shape[0]
    n_classes = int(lp.shape[1])
    neg = torch.full((), _NEG, dtype=lp.dtype, device=lp.device)
    zero = torch.zeros((), dtype=lp.dtype, device=lp.device)
    # forward alpha (includes the current-cell emission)
    f: list[Tensor] = [lp[0, int(ext[s])] if s < 2 else neg for s in range(m)]
    alpha_rows = [torch.stack(f)]
    for t in range(1, n_steps):
        nf = [
            lp[t, int(ext[s])] + logsumexp_beta(torch.stack([f[p] for p in lattice.incoming(s)]), beta, axis=-1)
            for s in range(m)
        ]
        f = nf
        alpha_rows.append(torch.stack(f))
    alpha = torch.stack(alpha_rows, dim=0)  # (T, m)
    # outgoing edges of the lattice (transpose of incoming)
    outgoing: dict[int, list[int]] = {s: [] for s in range(m)}
    for s in range(m):
        for p in lattice.incoming(s):
            outgoing[p].append(s)
    # backward beta (suffix score, excludes the current-cell emission)
    b: list[Tensor] = [zero if s in (m - 1, m - 2) else neg for s in range(m)]
    bwd_rows = [torch.stack(b)]
    for t in range(n_steps - 2, -1, -1):
        nb = [
            logsumexp_beta(
                torch.stack([lp[t + 1, int(ext[s2])] + b[s2] for s2 in outgoing[s]]), beta, axis=-1
            )
            if outgoing[s]
            else neg
            for s in range(m)
        ]
        b = nb
        bwd_rows.insert(0, torch.stack(b))
    bwd = torch.stack(bwd_rows, dim=0)  # (T, m)
    end_vals = [alpha_rows[n_steps - 1][m - 1]] + ([alpha_rows[n_steps - 1][m - 2]] if m >= 2 else [])
    value = logsumexp_beta(torch.stack(end_vals), beta, axis=-1)
    gamma_cell = torch.exp(beta * (alpha + bwd - value))  # (T, m)
    cols: list[Tensor] = []
    for c in range(n_classes):
        members = [s for s in range(m) if int(ext[s]) == c]
        col = torch.zeros(n_steps, dtype=lp.dtype, device=lp.device)
        for s in members:
            col = col + gamma_cell[:, s]
        cols.append(col)
    return torch.stack(cols, dim=1)  # (T, C)


# ---------------------------------------------------------------------------
# Batched twins (leading batch dim via torch.func.vmap; bit-identical to a loop)
# ---------------------------------------------------------------------------


def soft_viterbi_batched(
    emissions: Tensor,
    transitions: Tensor,
    beta: float = 1.0,
    *,
    start: Tensor | None = None,
) -> Tensor:
    r"""Batched :func:`soft_viterbi` -> ``(B,)`` for ``emissions`` ``(B, T, S)``.

    ``transitions`` may be shared ``(S, S)`` or per-example ``(B, S, S)``; ``start`` may be
    ``None``, shared ``(S,)``, or per-example ``(B, S)``. Maps :func:`soft_viterbi` over
    the leading batch axis with ``torch.func.vmap`` -- bit-identical to looping.
    """
    from torch.func import vmap

    t_dim = 0 if transitions.dim() == 3 else None
    if start is None:
        def fwd(e: Tensor, tr: Tensor) -> Tensor:
            return soft_viterbi(e, tr, beta)

        out: Tensor = vmap(fwd, in_dims=(0, t_dim))(emissions, transitions)
        return out
    s_dim = 0 if start.dim() == 2 else None

    def fwd_start(e: Tensor, tr: Tensor, st: Tensor) -> Tensor:
        return soft_viterbi(e, tr, beta, start=st)

    out_s: Tensor = vmap(fwd_start, in_dims=(0, t_dim, s_dim))(emissions, transitions, start)
    return out_s


def soft_shortest_path_batched(weights: Tensor, dag: DAG, beta: float = 1.0) -> Tensor:
    r"""Batched :func:`soft_shortest_path` -> ``(B,)`` for ``weights`` ``(B, n, n)`` (shared ``dag``)."""
    from torch.func import vmap

    def fwd(w: Tensor) -> Tensor:
        return soft_shortest_path(w, dag, beta)

    out: Tensor = vmap(fwd)(weights)
    return out


def soft_ctc_batched(log_probs: Tensor, lattice: CTCLattice, beta: float = 1.0) -> Tensor:
    r"""Batched :func:`soft_ctc` -> ``(B,)`` for ``log_probs`` ``(B, T, C)`` (shared ``lattice``)."""
    from torch.func import vmap

    def fwd(lp: Tensor) -> Tensor:
        return soft_ctc(lp, lattice, beta)

    out: Tensor = vmap(fwd)(log_probs)
    return out


__all__ = [
    "soft_ctc",
    "soft_ctc_batched",
    "soft_ctc_marginals",
    "soft_shortest_path",
    "soft_shortest_path_batched",
    "soft_shortest_path_marginals",
    "soft_viterbi",
    "soft_viterbi_batched",
    "soft_viterbi_marginals",
    "soft_viterbi_transition_marginals",
]
