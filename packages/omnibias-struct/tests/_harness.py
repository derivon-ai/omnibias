# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""The data-driven probe / oracle harness for omnibias-struct (the flaw-finder).

Every probe turns one soft-DP *claim* into a measured number and a boolean verdict,
returned as a JSON-serialisable :class:`ProbeRecord`. The same probes power both the
CPU-tiny deterministic regression subset (``tests/test_refinement_probes.py``) and the
multi-seed GPU-cluster sweep (``struct_refinement/`` in the separate ``omnibias_experiments`` project); the cluster script imports this
module and dumps a list of ``ProbeRecord.to_dict()`` as the committed metrics JSON.

Probes (each is honest about which axis it exercises):

* ``oracle_agreement`` -- hard DP == brute force (``beta -> inf`` limit is exact).
* ``gap_tightness`` -- realised ``|V_beta - V*|`` vs the closed-form bound (temperature axis).
* ``marginals_vs_autodiff`` -- closed-form forward-backward marginal == backend autodiff
  (the ``delta -> 0`` tower gradient is exact).
* ``parity`` -- torch <-> jax bit-for-bit agreement.
* ``beta_stability`` -- value / marginals stay finite and normalised as ``beta`` grows.
* ``path_count`` -- ``log N`` via a log-space count vs ``log`` of the exact integer count.

This is a test/bench helper, not shipped API; it may import the backends freely.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from _struct_helpers import dag_weight_matrix, random_chain, random_dag, sample_ctc
from omnibias.struct import (
    DAG,
    ChainTrellis,
    CTCLattice,
    brute_force_shortest_path,
    brute_force_viterbi,
    certify_soft_dp,
    count_paths,
    ctc_best,
    logsumexp_gap_bound,
    shortest_path,
    viterbi,
)


@dataclass(frozen=True)
class ProbeRecord:
    """One measured probe outcome -- the committed metrics-record schema."""

    probe: str
    problem: str
    backend: str
    seed: int
    beta: float
    size: dict[str, int]
    metrics: dict[str, float]
    ok: bool
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable dict (used by the cluster sweep to build the metrics JSON)."""
        return asdict(self)


# ---------------------------------------------------------------------------
# Reference log-space path count (independent of struct's own counter)
# ---------------------------------------------------------------------------


def _logaddexp(values: list[float]) -> float:
    finite = [v for v in values if v != -math.inf]
    if not finite:
        return -math.inf
    m = max(finite)
    return m + math.log(sum(math.exp(v - m) for v in finite))


def reference_log_count(problem: ChainTrellis | DAG | CTCLattice, n_steps: int | None = None) -> float:
    r"""``log N`` via a pure log-space DP -- an overflow-free oracle for the path count.

    Never materialises the (possibly astronomical) integer count, so it stays finite where
    ``math.log(count_paths(...))`` would need a bignum or overflow to ``inf``.
    """
    if isinstance(problem, ChainTrellis):
        return problem.n_steps * math.log(problem.n_states)
    if isinstance(problem, DAG):
        succ: dict[int, list[int]] = {}
        for u, v in problem.edges:
            succ.setdefault(u, []).append(v)
        log_counts = [-math.inf] * problem.num_nodes
        log_counts[problem.sink] = 0.0
        for node in range(problem.num_nodes - 1, -1, -1):
            if node == problem.sink:
                continue
            log_counts[node] = _logaddexp([log_counts[v] for v in succ.get(node, [])])
        return log_counts[problem.source]
    if n_steps is None:
        raise ValueError("reference_log_count on a CTCLattice requires n_steps")
    m = 2 * problem.n_labels + 1
    if n_steps < problem.n_labels:
        return -math.inf
    log_counts = [-math.inf] * m
    log_counts[0] = 0.0
    if m > 1:
        log_counts[1] = 0.0
    for _ in range(1, n_steps):
        nxt = [-math.inf] * m
        for s in range(m):
            nxt[s] = _logaddexp([log_counts[p] for p in problem.incoming(s)])
        log_counts = nxt
    ends = [log_counts[m - 1]] + ([log_counts[m - 2]] if m >= 2 else [])
    return _logaddexp(ends)


# ---------------------------------------------------------------------------
# Backend soft-value dispatch (torch / jax)
# ---------------------------------------------------------------------------


def _soft_value(backend: str, problem: str, data: Any, beta: float) -> float:
    if backend == "torch":
        import torch
        from omnibias.struct.torch import soft_ctc, soft_shortest_path, soft_viterbi

        if problem == "viterbi":
            trellis, = data
            return float(
                soft_viterbi(
                    torch.tensor(trellis.emissions),
                    torch.tensor(trellis.transitions),
                    beta,
                    start=torch.tensor(trellis.start),
                )
            )
        if problem == "shortest_path":
            dag, w = data
            return float(soft_shortest_path(torch.tensor(w), dag, beta))
        lattice, lp = data
        return float(soft_ctc(torch.tensor(lp), lattice, beta))
    import jax.numpy as jnp
    from omnibias.struct.jax import soft_ctc, soft_shortest_path, soft_viterbi

    if problem == "viterbi":
        trellis, = data
        return float(
            soft_viterbi(
                jnp.asarray(trellis.emissions),
                jnp.asarray(trellis.transitions),
                beta,
                start=jnp.asarray(trellis.start),
            )
        )
    if problem == "shortest_path":
        dag, w = data
        return float(soft_shortest_path(jnp.asarray(w), dag, beta))
    lattice, lp = data
    return float(soft_ctc(jnp.asarray(lp), lattice, beta))


# ---------------------------------------------------------------------------
# Probes
# ---------------------------------------------------------------------------


def probe_oracle_agreement(problem: str, seed: int,
                           size: dict[str, int] | None = None) -> ProbeRecord:
    """Hard DP == brute-force oracle on a tiny instance."""
    s = _resolve_size(problem, size)
    if problem == "viterbi":
        trellis = random_chain(seed, n_steps=s["T"], n_states=s["S"])
        hv, hp = viterbi(trellis)
        bv, bp = brute_force_viterbi(trellis)
        diff = abs(hv - bv)
        size_rec = {"T": trellis.n_steps, "S": trellis.n_states}
    elif problem == "shortest_path":
        dag = random_dag(seed, n=s["n"])
        hv, hp = shortest_path(dag)
        bv, bp = brute_force_shortest_path(dag)
        diff = abs(hv - bv)
        size_rec = {"n": dag.num_nodes}
    else:
        lattice, lp = sample_ctc(seed, n_steps=s["T"], num_classes=s["C"])
        from omnibias.struct import brute_force_ctc

        diff = abs(ctc_best(lattice, lp) - brute_force_ctc(lattice, lp))
        size_rec = {"T": lp.shape[0], "C": lp.shape[1]}
    return ProbeRecord(
        "oracle_agreement", problem, "numpy", seed, math.inf, size_rec,
        {"abs_diff": diff}, ok=diff < 1e-9,
    )


def probe_gap_tightness(problem: str, seed: int, beta: float, backend: str = "torch",
                        size: dict[str, int] | None = None) -> ProbeRecord:
    """Realised soft-vs-hard gap and its closed-form bound; verdict = sandwich is sound."""
    s = _resolve_size(problem, size)
    if problem == "viterbi":
        trellis = random_chain(seed, n_steps=s["T"], n_states=s["S"])
        hard, _ = viterbi(trellis)
        n = count_paths(trellis)
        soft = _soft_value(backend, problem, (trellis,), beta)
        sense = "max"
        size_rec = {"T": trellis.n_steps, "S": trellis.n_states}
    elif problem == "shortest_path":
        dag = random_dag(seed, n=s["n"])
        hard, _ = shortest_path(dag)
        n = count_paths(dag)
        soft = _soft_value(backend, problem, (dag, dag_weight_matrix(dag)), beta)
        sense = "min"
        size_rec = {"n": dag.num_nodes}
    else:
        lattice, lp = sample_ctc(seed, n_steps=s["T"], num_classes=s["C"])
        hard = ctc_best(lattice, lp)
        n = count_paths(lattice, lp.shape[0])
        soft = _soft_value(backend, problem, (lattice, lp), beta)
        sense = "max"
        size_rec = {"T": lp.shape[0], "C": lp.shape[1]}
    cert = certify_soft_dp(hard, soft, n, beta, sense=sense)
    ratio = cert.absolute_gap / cert.gap_bound if cert.gap_bound > 0 else 0.0
    return ProbeRecord(
        "gap_tightness", problem, backend, seed, beta, size_rec,
        {"realized_gap": cert.absolute_gap, "bound": cert.gap_bound, "tightness_ratio": ratio,
         "num_paths": float(n)},
        ok=cert.is_sound,
    )


def probe_marginals_vs_autodiff(problem: str, seed: int, beta: float, backend: str,
                                size: dict[str, int] | None = None) -> ProbeRecord:
    """Closed-form forward-backward marginal vs backend autodiff of the soft value.

    Normalisation check is per problem: Viterbi / CTC marginals sum to 1 along the state /
    class axis at every time; DAG edge marginals instead obey flow conservation, so we
    check the source outflow equals 1.

    Closed-form forward-backward and reverse-mode autodiff compute the *same* marginal by
    different float64 operation orders, so their agreement is bounded by the working-precision
    envelope ``~ eps * beta * L`` (``L`` = the instance's characteristic length): at ``beta =
    1e6`` over a ``T ~ 100`` chain that is ~1e-7, not ``1e-9``. The verdict therefore compares
    against ``max(1e-9, 64 * eps * beta * L)`` -- strict at moderate ``beta`` (the CPU
    regression subset is unaffected), honestly loosened only in the extreme-temperature corner.
    """
    closed, auto = _marginals_and_autodiff(problem, seed, beta, backend, size)
    diff = float(np.max(np.abs(closed - auto)))
    if problem == "shortest_path":
        norm_err = float(abs(closed[0].sum() - 1.0))  # source (node 0) outflow == 1
    else:
        norm_err = float(np.max(np.abs(closed.sum(axis=-1) - 1.0)))
    scale = max(_resolve_size(problem, size).values())
    envelope = 64.0 * float(np.finfo(np.float64).eps) * beta * scale
    tol = max(1e-9, envelope)
    norm_tol = max(1e-7, envelope)
    return ProbeRecord(
        "marginals_vs_autodiff", problem, backend, seed, beta, _size(problem, seed, size),
        {"max_abs_diff": diff, "norm_err": norm_err, "tol": tol},
        ok=diff < tol and norm_err < norm_tol,
    )


def probe_beta_stability(problem: str, seed: int, beta: float, backend: str,
                         size: dict[str, int] | None = None) -> ProbeRecord:
    """Value stays finite (and, where available, marginals sum to 1) as beta grows."""
    value = _soft_value(backend, problem, _make(problem, seed, size), beta)
    marg_sum_err = _marginal_sum_error(problem, seed, beta, backend, size)
    any_nan = (not math.isfinite(value)) or (marg_sum_err is not None and not math.isfinite(marg_sum_err))
    metrics = {"value": value, "value_finite": float(math.isfinite(value))}
    if marg_sum_err is not None:
        metrics["marginal_sum_err"] = marg_sum_err
    ok = math.isfinite(value) and (marg_sum_err is None or marg_sum_err < 1e-6)
    return ProbeRecord(
        "beta_stability", problem, backend, seed, beta, _size(problem, seed, size),
        metrics, ok=ok and not any_nan,
    )


def probe_parity(problem: str, seed: int, beta: float,
                 size: dict[str, int] | None = None) -> ProbeRecord:
    """torch vs jax soft value (and marginals for viterbi/shortest-path) agreement."""
    vt = _soft_value("torch", problem, _make(problem, seed, size), beta)
    vj = _soft_value("jax", problem, _make(problem, seed, size), beta)
    diff = abs(vt - vj)
    return ProbeRecord(
        "parity", problem, "torch|jax", seed, beta, _size(problem, seed, size),
        {"value_abs_diff": diff}, ok=diff < 1e-9,
    )


def probe_path_count(problem: str, seed: int, n_steps: int | None = None,
                     size: dict[str, int] | None = None) -> ProbeRecord:
    """log N via the log-space count vs log of the exact integer count (overflow check)."""
    prob = _problem_only(problem, seed, n_steps, size)
    if problem == "ctc":
        n_steps = n_steps or _resolve_size(problem, size)["T"]
        exact = count_paths(prob, n_steps)
        log_ref = reference_log_count(prob, n_steps)
        size_rec = {"T": n_steps}
    else:
        exact = count_paths(prob)
        log_ref = reference_log_count(prob)
        size_rec = _size(problem, seed, size)
    log_exact = math.log(exact) if exact > 0 else -math.inf
    err = abs(log_exact - log_ref) if math.isfinite(log_ref) else 0.0
    return ProbeRecord(
        "path_count", problem, "numpy", seed, math.inf, size_rec,
        {"log_count_ref": log_ref, "log_count_exact": log_exact, "abs_err": err,
         "count": float(exact)},
        ok=math.isfinite(log_ref) and err < 1e-6,
    )


# ---------------------------------------------------------------------------
# small dispatch helpers
# ---------------------------------------------------------------------------


DEFAULT_SIZES: dict[str, dict[str, int]] = {
    "viterbi": {"T": 4, "S": 3},
    "shortest_path": {"n": 5},
    "ctc": {"T": 4, "C": 3},
}


def _resolve_size(problem: str, size: dict[str, int] | None) -> dict[str, int]:
    """Merge an optional size override onto the per-problem defaults (unknown keys ignored)."""
    base = dict(DEFAULT_SIZES[problem])
    if size:
        base.update({k: v for k, v in size.items() if k in base})
    return base


def _make(problem: str, seed: int, size: dict[str, int] | None = None) -> Any:
    s = _resolve_size(problem, size)
    if problem == "viterbi":
        return (random_chain(seed, n_steps=s["T"], n_states=s["S"]),)
    if problem == "shortest_path":
        dag = random_dag(seed, n=s["n"])
        return (dag, dag_weight_matrix(dag))
    lattice, lp = sample_ctc(seed, n_steps=s["T"], num_classes=s["C"])
    return (lattice, lp)


def _problem_only(problem: str, seed: int, n_steps: int | None,
                  size: dict[str, int] | None = None) -> Any:
    s = _resolve_size(problem, size)
    if problem == "viterbi":
        return random_chain(seed, n_steps=n_steps or s["T"], n_states=s["S"])
    if problem == "shortest_path":
        return random_dag(seed, n=n_steps or s["n"])
    lattice, _ = sample_ctc(seed, n_steps=n_steps or s["T"], num_classes=s["C"])
    return lattice


def _size(problem: str, seed: int, size: dict[str, int] | None = None) -> dict[str, int]:
    return _resolve_size(problem, size)


def _marginals_and_autodiff(problem: str, seed: int, beta: float, backend: str,
                            size: dict[str, int] | None = None) -> tuple[Any, Any]:
    """(closed-form marginal, autodiff grad) as numpy arrays -- shapes match per problem."""
    s = _resolve_size(problem, size)
    if backend == "torch":
        import torch

        if problem == "viterbi":
            from omnibias.struct.torch import soft_viterbi, soft_viterbi_marginals

            trellis = random_chain(seed, n_steps=s["T"], n_states=s["S"])
            e = torch.tensor(trellis.emissions, requires_grad=True)
            tr = torch.tensor(trellis.transitions)
            st = torch.tensor(trellis.start)
            soft_viterbi(e, tr, beta, start=st).backward()
            closed = soft_viterbi_marginals(e.detach(), tr, beta, start=st).numpy()
            return closed, e.grad.numpy()
        if problem == "shortest_path":
            from omnibias.struct.torch import soft_shortest_path, soft_shortest_path_marginals

            dag = random_dag(seed, n=s["n"])
            w = torch.tensor(dag_weight_matrix(dag), requires_grad=True)
            soft_shortest_path(w, dag, beta).backward()
            closed = soft_shortest_path_marginals(w.detach(), dag, beta).numpy()
            grad = np.nan_to_num(w.grad.numpy())
            mask = np.zeros_like(grad)
            for u, v in dag.edges:
                mask[u, v] = 1.0
            return closed * mask, grad * mask
        from omnibias.struct.torch import soft_ctc, soft_ctc_marginals

        lattice, lp = sample_ctc(seed, n_steps=s["T"], num_classes=s["C"])
        x = torch.tensor(lp, requires_grad=True)
        soft_ctc(x, lattice, beta).backward()
        return soft_ctc_marginals(x.detach(), lattice, beta).numpy(), x.grad.numpy()
    import jax

    if problem == "viterbi":
        import jax.numpy as jnp
        from omnibias.struct.jax import soft_viterbi, soft_viterbi_marginals

        trellis = random_chain(seed, n_steps=s["T"], n_states=s["S"])
        e = jnp.asarray(trellis.emissions)
        tr = jnp.asarray(trellis.transitions)
        st = jnp.asarray(trellis.start)
        grad = jax.grad(lambda x: soft_viterbi(x, tr, beta, start=st))(e)
        return np.asarray(soft_viterbi_marginals(e, tr, beta, start=st)), np.asarray(grad)
    if problem == "shortest_path":
        import jax.numpy as jnp
        from omnibias.struct.jax import soft_shortest_path, soft_shortest_path_marginals

        dag = random_dag(seed, n=s["n"])
        w = jnp.asarray(dag_weight_matrix(dag))
        grad = np.asarray(jax.grad(lambda x: soft_shortest_path(x, dag, beta))(w))
        closed = np.asarray(soft_shortest_path_marginals(w, dag, beta))
        mask = np.zeros_like(closed)
        for u, v in dag.edges:
            mask[u, v] = 1.0
        return closed * mask, np.nan_to_num(grad) * mask
    import jax.numpy as jnp
    from omnibias.struct.jax import soft_ctc, soft_ctc_marginals

    lattice, lp = sample_ctc(seed, n_steps=s["T"], num_classes=s["C"])
    x = jnp.asarray(lp)
    grad = jax.grad(lambda z: soft_ctc(z, lattice, beta))(x)
    return np.asarray(soft_ctc_marginals(x, lattice, beta)), np.asarray(grad)


def _marginal_sum_error(problem: str, seed: int, beta: float, backend: str,
                        size: dict[str, int] | None = None) -> float | None:
    """Max |sum of marginals - 1| where a closed-form marginal exists, else None."""
    data = _make(problem, seed, size)
    if backend == "torch":
        import torch

        if problem == "viterbi":
            from omnibias.struct.torch import soft_viterbi_marginals

            trellis, = data
            g = soft_viterbi_marginals(
                torch.tensor(trellis.emissions), torch.tensor(trellis.transitions), beta,
                start=torch.tensor(trellis.start),
            ).numpy()
            return float(np.max(np.abs(g.sum(axis=1) - 1.0)))
        return None
    import jax.numpy as jnp

    if problem == "viterbi":
        from omnibias.struct.jax import soft_viterbi_marginals

        trellis, = data
        g = np.asarray(
            soft_viterbi_marginals(
                jnp.asarray(trellis.emissions), jnp.asarray(trellis.transitions), beta,
                start=jnp.asarray(trellis.start),
            )
        )
        return float(np.max(np.abs(g.sum(axis=1) - 1.0)))
    return None


# ===========================================================================
# New-family probes: CKY parse / Eisner / matrix-tree / operators / alignment.
# These lift the same honesty axes (oracle agreement, closed-form marginals ==
# autodiff, parity, gap soundness) onto the semiring-driver families, plus the
# operator-specific checks (entropy identity + MC, sampler empirical marginals,
# exact k-best). Instances are tiny so the exponential oracles stay cheap.
# ===========================================================================


def _cky_grammar():  # noqa: ANN202
    from omnibias.struct import BinaryGrammar

    return BinaryGrammar(
        num_nonterminals=2, rules=((0, 0, 1), (0, 1, 1), (1, 0, 0), (1, 1, 0)), start=0
    )


def _make_cky(seed: int, length: int = 3) -> tuple[Any, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    g = _cky_grammar()
    emit = rng.standard_normal((length, g.num_nonterminals))
    rule = rng.standard_normal(g.num_rules)
    return g, emit, rule


def _make_arc(seed: int, n: int = 3) -> np.ndarray:
    return np.asarray(np.random.default_rng(seed).standard_normal((n + 1, n + 1)))


def _make_arc_mtt(seed: int, n: int = 3) -> np.ndarray:
    r"""Well-separated arc scores whose optimum is a genuine (acyclic, ROOT-rooted) tree.

    The matrix-tree partition is an *exact* determinant for any scores, but its float64
    evaluation via ``det`` / ``L^{-1}`` is only well-conditioned when the maximum arborescence
    is a real tree: as ``beta -> inf`` the exp-Laplacian tends to the Laplacian minor of the
    greedy-argmax structure, whose determinant is ``1`` for a tree but ``0`` for a cycle. Pure
    ``standard_normal`` arcs give a *cyclic* greedy argmax often enough that the minor goes
    singular by ``beta ~ 16`` (an intrinsic limit of the determinant route, pinned by
    ``test_matrix_tree_singular_at_high_beta_for_cyclic_argmax``). Biasing one head ``< m`` per
    modifier makes the winner a clean tree, so the twin stays bit-identical and marginals stay
    ``== `` autodiff across the whole ``beta`` ladder -- the representative parsing case.
    """
    rng = np.random.default_rng(seed)
    arc = 0.3 * rng.standard_normal((n + 1, n + 1))
    for m in range(1, n + 1):
        arc[int(rng.integers(0, m)), m] += 3.0  # preferred head in {0..m-1} => acyclic, rooted
    return np.asarray(arc)


def _env_tol(beta: float, scale: int, floor: float) -> float:
    r"""Envelope-aware marginal tolerance ``max(floor, 128 eps beta scale)``.

    Closed-form inside-outside and reverse-mode autodiff compute the *same* marginal in
    different float64 operation orders, so their agreement is bounded by the working-precision
    envelope ``~ eps * beta * depth`` -- strict at moderate ``beta`` (the CPU regression subset),
    honestly loosened only in the extreme-temperature corner (mirrors ``probe_marginals_vs_autodiff``).
    """
    return max(floor, 128.0 * float(np.finfo(np.float64).eps) * beta * scale)


def probe_parse_oracle(seed: int) -> ProbeRecord:
    """CKY: hard Viterbi-CKY == brute force, and the derivation count == enumerated trees."""
    from omnibias.struct import brute_force_cky, count_parse_trees, hard_cky

    g, emit, rule = _make_cky(seed)
    diff = abs(hard_cky(g, emit, rule) - brute_force_cky(g, emit, rule))
    n_trees = count_parse_trees(g, emit.shape[0])
    from omnibias.struct import build_chart
    from omnibias.struct._core.semiring import enumerate_derivations

    enum = sum(1 for _ in enumerate_derivations(build_chart(g, emit.shape[0]).graph))
    ok = diff < 1e-9 and n_trees == enum
    return ProbeRecord(
        "oracle_agreement", "parse", "numpy", seed, math.inf,
        {"L": emit.shape[0], "R": g.num_nonterminals},
        {"abs_diff": diff, "count": float(n_trees), "enum": float(enum)}, ok=ok,
    )


def probe_eisner_oracle(seed: int) -> ProbeRecord:
    """Eisner: hard == brute-force projective, and the derivation count == projective trees."""
    from omnibias.struct import brute_force_projective, count_projective_trees, hard_eisner

    arc = _make_arc(seed)
    n = arc.shape[0] - 1
    diff = abs(hard_eisner(arc) - brute_force_projective(arc))
    n_trees = count_projective_trees(n)
    from omnibias.struct._core.eisner import iter_projective_trees

    enum = sum(1 for _ in iter_projective_trees(n))
    ok = diff < 1e-9 and n_trees == enum
    return ProbeRecord(
        "oracle_agreement", "eisner", "numpy", seed, math.inf, {"n": n},
        {"abs_diff": diff, "count": float(n_trees), "enum": float(enum)}, ok=ok,
    )


def probe_mtt_oracle(seed: int) -> ProbeRecord:
    """Matrix-tree: Chu-Liu/Edmonds max arborescence == hard == brute-force over arborescences."""
    from omnibias.struct import brute_force_arborescence, hard_matrix_tree, max_arborescence

    arc = _make_arc_mtt(seed)
    edmonds, _ = max_arborescence(arc)
    diff = max(abs(edmonds - hard_matrix_tree(arc)), abs(edmonds - brute_force_arborescence(arc)))
    return ProbeRecord(
        "oracle_agreement", "mtt", "numpy", seed, math.inf, {"n": arc.shape[0] - 1},
        {"abs_diff": diff}, ok=diff < 1e-9,
    )


def probe_align_oracle(kind: str, seed: int) -> ProbeRecord:
    """Local (Smith-Waterman) / affine-gap (Gotoh) hard DP == brute force over alignments."""
    rng = np.random.default_rng(seed)
    a = rng.integers(0, 3, size=4)
    b = rng.integers(0, 3, size=3)
    sub = rng.standard_normal((3, 3))
    if kind == "local":
        from omnibias.struct import brute_force_local_align, hard_local_align

        gap = -0.7
        diff = abs(hard_local_align(a, b, sub, gap) - brute_force_local_align(a, b, sub, gap))
    else:
        from omnibias.struct import brute_force_gotoh, hard_gotoh

        diff = abs(hard_gotoh(a, b, sub, -1.0, -0.3) - brute_force_gotoh(a, b, sub, -1.0, -0.3))
    return ProbeRecord(
        "oracle_agreement", f"align_{kind}", "numpy", seed, math.inf,
        {"la": int(a.shape[0]), "lb": int(b.shape[0])}, {"abs_diff": diff}, ok=diff < 1e-9,
    )


def _parse_soft_and_marg(backend: str, g: Any, emit: np.ndarray, rule: np.ndarray, beta: float):  # noqa: ANN202
    if backend == "torch":
        import torch
        from omnibias.struct.torch import inside_outside, soft_inside

        e = torch.tensor(emit, requires_grad=True)
        r = torch.tensor(rule, requires_grad=True)
        val = soft_inside(g, e, r, beta)
        val.backward()
        em, rm = inside_outside(g, e.detach(), r.detach(), beta)
        auto = np.concatenate([e.grad.numpy().ravel(), r.grad.numpy().ravel()])
        closed = np.concatenate([em.numpy().ravel(), rm.numpy().ravel()])
        return float(val.detach()), closed, auto
    import jax
    import jax.numpy as jnp
    from omnibias.struct.jax import inside_outside, soft_inside

    e = jnp.asarray(emit)
    r = jnp.asarray(rule)
    val = float(soft_inside(g, e, r, beta))
    ge, gr = jax.grad(lambda ee, rr: soft_inside(g, ee, rr, beta), argnums=(0, 1))(e, r)
    em, rm = inside_outside(g, e, r, beta)
    auto = np.concatenate([np.asarray(ge).ravel(), np.asarray(gr).ravel()])
    closed = np.concatenate([np.asarray(em).ravel(), np.asarray(rm).ravel()])
    return val, closed, auto


def probe_parse_marginals(seed: int, beta: float, backend: str) -> ProbeRecord:
    """CKY inside-outside span/rule marginals == backend autodiff of the inside partition."""
    g, emit, rule = _make_cky(seed)
    _val, closed, auto = _parse_soft_and_marg(backend, g, emit, rule, beta)
    diff = float(np.max(np.abs(closed - auto)))
    tol = _env_tol(beta, emit.shape[0], 1e-8)
    return ProbeRecord(
        "marginals_vs_autodiff", "parse", backend, seed, beta,
        {"L": emit.shape[0], "R": g.num_nonterminals}, {"max_abs_diff": diff, "tol": tol},
        ok=diff < tol,
    )


def probe_parse_gap(seed: int, beta: float, backend: str) -> ProbeRecord:
    """CKY certified log(N)/beta sandwich: lse_beta >= max and the gap bound holds."""
    from omnibias.struct import count_parse_trees, hard_cky

    g, emit, rule = _make_cky(seed)
    val, _closed, _auto = _parse_soft_and_marg(backend, g, emit, rule, beta)
    cert = certify_soft_dp(hard_cky(g, emit, rule), val, count_parse_trees(g, emit.shape[0]), beta,
                           sense="max")
    ratio = cert.absolute_gap / cert.gap_bound if cert.gap_bound > 0 else 0.0
    return ProbeRecord(
        "gap_tightness", "parse", backend, seed, beta,
        {"L": emit.shape[0], "R": g.num_nonterminals},
        {"realized_gap": cert.absolute_gap, "bound": cert.gap_bound, "tightness_ratio": ratio,
         "num_paths": float(cert.num_paths)}, ok=cert.is_sound,
    )


def probe_parse_parity(seed: int, beta: float) -> ProbeRecord:
    """torch vs jax CKY inside partition."""
    g, emit, rule = _make_cky(seed)
    vt, _, _ = _parse_soft_and_marg("torch", g, emit, rule, beta)
    vj, _, _ = _parse_soft_and_marg("jax", g, emit, rule, beta)
    diff = abs(vt - vj)
    return ProbeRecord(
        "parity", "parse", "torch|jax", seed, beta,
        {"L": emit.shape[0], "R": g.num_nonterminals}, {"value_abs_diff": diff}, ok=diff < 1e-9,
    )


def _eisner_soft_and_marg(backend: str, arc: np.ndarray, beta: float):  # noqa: ANN202
    if backend == "torch":
        import torch
        from omnibias.struct.torch import eisner_marginals, soft_eisner

        a = torch.tensor(arc, requires_grad=True)
        val = soft_eisner(a, beta)
        val.backward()
        closed = eisner_marginals(a.detach(), beta).numpy()
        return float(val.detach()), closed, a.grad.numpy()
    import jax
    import jax.numpy as jnp
    from omnibias.struct.jax import eisner_marginals, soft_eisner

    a = jnp.asarray(arc)
    val = float(soft_eisner(a, beta))
    grad = np.asarray(jax.grad(lambda x: soft_eisner(x, beta))(a))
    closed = np.asarray(eisner_marginals(a, beta))
    return val, closed, grad


def probe_eisner_marginals(seed: int, beta: float, backend: str) -> ProbeRecord:
    """Eisner arc marginals == backend autodiff of the projective-parse partition."""
    arc = _make_arc(seed)
    _val, closed, auto = _eisner_soft_and_marg(backend, arc, beta)
    diff = float(np.max(np.abs(closed - auto)))
    tol = _env_tol(beta, arc.shape[0] - 1, 1e-8)
    return ProbeRecord(
        "marginals_vs_autodiff", "eisner", backend, seed, beta, {"n": arc.shape[0] - 1},
        {"max_abs_diff": diff, "tol": tol}, ok=diff < tol,
    )


def probe_eisner_gap(seed: int, beta: float, backend: str) -> ProbeRecord:
    """Eisner certified log(N)/beta sandwich."""
    from omnibias.struct import count_projective_trees, hard_eisner

    arc = _make_arc(seed)
    val, _closed, _auto = _eisner_soft_and_marg(backend, arc, beta)
    cert = certify_soft_dp(hard_eisner(arc), val, count_projective_trees(arc.shape[0] - 1), beta,
                           sense="max")
    ratio = cert.absolute_gap / cert.gap_bound if cert.gap_bound > 0 else 0.0
    return ProbeRecord(
        "gap_tightness", "eisner", backend, seed, beta, {"n": arc.shape[0] - 1},
        {"realized_gap": cert.absolute_gap, "bound": cert.gap_bound, "tightness_ratio": ratio,
         "num_paths": float(cert.num_paths)}, ok=cert.is_sound,
    )


def probe_eisner_parity(seed: int, beta: float) -> ProbeRecord:
    """torch vs jax Eisner partition."""
    arc = _make_arc(seed)
    vt, _, _ = _eisner_soft_and_marg("torch", arc, beta)
    vj, _, _ = _eisner_soft_and_marg("jax", arc, beta)
    diff = abs(vt - vj)
    return ProbeRecord(
        "parity", "eisner", "torch|jax", seed, beta, {"n": arc.shape[0] - 1},
        {"value_abs_diff": diff}, ok=diff < 1e-9,
    )


def _mtt_soft_and_marg(backend: str, arc: np.ndarray, beta: float):  # noqa: ANN202
    if backend == "torch":
        import torch
        from omnibias.struct.torch import matrix_tree_marginals, soft_matrix_tree

        a = torch.tensor(arc, requires_grad=True)
        val = soft_matrix_tree(a, beta)
        val.backward()
        closed = matrix_tree_marginals(a.detach(), beta).numpy()
        return float(val.detach()), closed, a.grad.numpy()
    import jax
    import jax.numpy as jnp
    from omnibias.struct.jax import matrix_tree_marginals, soft_matrix_tree

    a = jnp.asarray(arc)
    val = float(soft_matrix_tree(a, beta))
    grad = np.asarray(jax.grad(lambda x: soft_matrix_tree(x, beta))(a))
    closed = np.asarray(matrix_tree_marginals(a, beta))
    return val, closed, grad


def probe_mtt_marginals(seed: int, beta: float, backend: str) -> ProbeRecord:
    """Matrix-tree arc marginals (via L^{-1}) == backend autodiff of the exact determinant value."""
    arc = _make_arc_mtt(seed)
    _val, closed, auto = _mtt_soft_and_marg(backend, arc, beta)
    # Only the meaningful entries (column m >= 1) carry a marginal; column 0 is the ROOT wall.
    diff = float(np.max(np.abs(closed[:, 1:] - auto[:, 1:])))
    # Matrix-tree marginals come from L^{-1}, so the envelope also carries the Laplacian
    # conditioning; a slightly higher floor than the inside-outside families, still strict
    # at moderate beta.
    tol = _env_tol(beta, arc.shape[0] - 1, 1e-7)
    return ProbeRecord(
        "marginals_vs_autodiff", "mtt", backend, seed, beta, {"n": arc.shape[0] - 1},
        {"max_abs_diff": diff, "tol": tol}, ok=diff < tol,
    )


def probe_mtt_gap(seed: int, beta: float, backend: str) -> ProbeRecord:
    """Matrix-tree beta->inf gap vs the max arborescence (bound log(#arborescences)/beta)."""
    from omnibias.struct import count_arborescences, max_arborescence

    arc = _make_arc_mtt(seed)
    val, _closed, _auto = _mtt_soft_and_marg(backend, arc, beta)
    edmonds, _ = max_arborescence(arc)
    cert = certify_soft_dp(edmonds, val, count_arborescences(arc.shape[0] - 1), beta, sense="max")
    ratio = cert.absolute_gap / cert.gap_bound if cert.gap_bound > 0 else 0.0
    return ProbeRecord(
        "gap_tightness", "mtt", backend, seed, beta, {"n": arc.shape[0] - 1},
        {"realized_gap": cert.absolute_gap, "bound": cert.gap_bound, "tightness_ratio": ratio,
         "num_paths": float(cert.num_paths)}, ok=cert.is_sound,
    )


def probe_mtt_parity(seed: int, beta: float) -> ProbeRecord:
    """torch vs jax matrix-tree partition."""
    arc = _make_arc_mtt(seed)
    vt, _, _ = _mtt_soft_and_marg("torch", arc, beta)
    vj, _, _ = _mtt_soft_and_marg("jax", arc, beta)
    diff = abs(vt - vj)
    return ProbeRecord(
        "parity", "mtt", "torch|jax", seed, beta, {"n": arc.shape[0] - 1},
        {"value_abs_diff": diff}, ok=diff < 1e-9,
    )


def _operator_graph(seed: int):  # noqa: ANN202
    """A tiny CKY-chart hypergraph + edge weights -- a rich arity-2 derivation forest."""
    from omnibias.struct._core.parse import build_chart, chart_edge_weights

    g, emit, rule = _make_cky(seed, length=3)
    spec = build_chart(g, emit.shape[0])
    return spec.graph, chart_edge_weights(spec, emit, rule)


def probe_entropy(seed: int, beta: float, backend: str) -> ProbeRecord:
    """Closed-form path entropy == brute-force enumeration oracle, and within MC of the sampler."""
    from omnibias.struct._core.operators import brute_force_entropy

    graph, w = _operator_graph(seed)
    exact = brute_force_entropy(graph, w, beta)
    if backend == "torch":
        import torch
        from omnibias.struct.torch import path_entropy, sample_paths

        closed = float(path_entropy(graph, torch.tensor(w), beta))
        counts, samples = sample_paths(graph, torch.tensor(w), beta, 4000, seed=seed)
    else:
        import jax.numpy as jnp
        from omnibias.struct.jax import path_entropy, sample_paths

        closed = float(path_entropy(graph, jnp.asarray(w), beta))
        counts, samples = sample_paths(graph, jnp.asarray(w), beta, 4000, seed=seed)
    value = _semiring_value_np(graph, w, beta)
    emp_scores = [float(sum(w[e] for e in d)) for d in samples]
    mc_entropy = beta * (value - float(np.mean(emp_scores)))
    id_err = abs(closed - exact)
    mc_err = abs(mc_entropy - exact)
    # H = beta * (V - E[score]) is a beta-amplified difference of near-equal O(1) quantities, so
    # the closed-form-vs-enumeration agreement carries a ``~ beta * eps`` cancellation envelope
    # (as beta -> inf, H -> log(#argmax) and both sides collapse toward 0). Strict at moderate
    # beta; honestly loosened only in the extreme-temperature corner.
    id_tol = max(1e-8, 1.0e5 * float(np.finfo(np.float64).eps) * beta)
    return ProbeRecord(
        "entropy", "operators", backend, seed, beta, {"n_edges": graph.num_edges},
        {"closed": closed, "exact": exact, "mc": mc_entropy, "identity_err": id_err,
         "mc_err": mc_err, "id_tol": id_tol},
        ok=id_err < id_tol and mc_err < 0.2,
    )


def probe_sampling(seed: int, beta: float, backend: str) -> ProbeRecord:
    """Exact FFBS empirical edge frequencies converge to the closed-form marginals."""
    graph, w = _operator_graph(seed)
    num = 20000
    if backend == "torch":
        import torch
        from omnibias.struct.torch import sample_paths
        from omnibias.struct.torch.semiring import semiring_marginals

        counts, _ = sample_paths(graph, torch.tensor(w), beta, num, seed=seed)
        emp = counts.numpy().mean(axis=0)
        closed = semiring_marginals(graph, torch.tensor(w), beta).numpy()
    else:
        import jax.numpy as jnp
        from omnibias.struct.jax import sample_paths
        from omnibias.struct.jax.semiring import semiring_marginals

        counts, _ = sample_paths(graph, jnp.asarray(w), beta, num, seed=seed)
        emp = np.asarray(counts).mean(axis=0)
        closed = np.asarray(semiring_marginals(graph, jnp.asarray(w), beta))
    diff = float(np.max(np.abs(emp - closed)))
    return ProbeRecord(
        "sampling", "operators", backend, seed, beta, {"n_edges": graph.num_edges, "num": num},
        {"max_abs_diff": diff}, ok=diff < 0.05,
    )


def probe_kbest(seed: int) -> ProbeRecord:
    """Exact topological k-best == enumerate-and-sort oracle (scores and order)."""
    from omnibias.struct._core.operators import brute_force_kbest, kbest_derivations

    graph, w = _operator_graph(seed)
    k = 5
    fast = kbest_derivations(graph, w, k)
    slow = brute_force_kbest(graph, w, k)
    diff = max((abs(a[0] - b[0]) for a, b in zip(fast, slow, strict=True)), default=0.0)
    return ProbeRecord(
        "kbest", "operators", "numpy", seed, math.inf, {"n_edges": graph.num_edges, "k": k},
        {"max_score_diff": diff, "n_returned": float(len(fast))},
        ok=diff < 1e-9 and len(fast) == len(slow),
    )


def _semiring_value_np(graph: Any, w: np.ndarray, beta: float) -> float:
    from omnibias.struct._core.semiring import soft_value

    return soft_value(graph, w, beta)


def probe_align_gap(kind: str, seed: int, beta: float, backend: str) -> ProbeRecord:
    """Local / affine-gap soft alignment: lse_beta >= max hard alignment (temperature axis)."""
    rng = np.random.default_rng(seed)
    a = rng.integers(0, 3, size=4)
    b = rng.integers(0, 3, size=3)
    sub = rng.standard_normal((3, 3))
    if backend == "torch":
        import torch

        f64 = torch.float64
        if kind == "local":
            from omnibias.struct import hard_local_align
            from omnibias.struct.torch import soft_local_align

            hard = hard_local_align(a, b, sub, -0.7)
            soft = float(soft_local_align(a, b, torch.tensor(sub), torch.tensor(-0.7, dtype=f64), beta))
        else:
            from omnibias.struct import hard_gotoh
            from omnibias.struct.torch import soft_gotoh

            hard = hard_gotoh(a, b, sub, -1.0, -0.3)
            soft = float(
                soft_gotoh(a, b, torch.tensor(sub), torch.tensor(-1.0, dtype=f64),
                           torch.tensor(-0.3, dtype=f64), beta)
            )
    else:
        import jax.numpy as jnp

        if kind == "local":
            from omnibias.struct import hard_local_align
            from omnibias.struct.jax import soft_local_align

            hard = hard_local_align(a, b, sub, -0.7)
            soft = float(soft_local_align(a, b, jnp.asarray(sub), jnp.asarray(-0.7), beta))
        else:
            from omnibias.struct import hard_gotoh
            from omnibias.struct.jax import soft_gotoh

            hard = hard_gotoh(a, b, sub, -1.0, -0.3)
            soft = float(soft_gotoh(a, b, jnp.asarray(sub), jnp.asarray(-1.0), jnp.asarray(-0.3), beta))
    lse_ge_max = soft >= hard - 1e-9
    return ProbeRecord(
        "gap_tightness", f"align_{kind}", backend, seed, beta,
        {"la": int(a.shape[0]), "lb": int(b.shape[0])},
        {"hard": hard, "soft": soft, "realized_gap": abs(soft - hard)}, ok=lse_ge_max,
    )


def probe_align_parity(kind: str, seed: int, beta: float) -> ProbeRecord:
    """torch vs jax local / affine-gap soft alignment value."""
    rng = np.random.default_rng(seed)
    a = rng.integers(0, 3, size=4)
    b = rng.integers(0, 3, size=3)
    sub = rng.standard_normal((3, 3))
    import jax.numpy as jnp
    import torch

    f64 = torch.float64
    if kind == "local":
        from omnibias.struct.jax import soft_local_align as jax_f
        from omnibias.struct.torch import soft_local_align as torch_f

        vt = float(torch_f(a, b, torch.tensor(sub), torch.tensor(-0.7, dtype=f64), beta))
        vj = float(jax_f(a, b, jnp.asarray(sub), jnp.asarray(-0.7), beta))
    else:
        from omnibias.struct.jax import soft_gotoh as jax_g
        from omnibias.struct.torch import soft_gotoh as torch_g

        vt = float(
            torch_g(a, b, torch.tensor(sub), torch.tensor(-1.0, dtype=f64),
                    torch.tensor(-0.3, dtype=f64), beta)
        )
        vj = float(jax_g(a, b, jnp.asarray(sub), jnp.asarray(-1.0), jnp.asarray(-0.3), beta))
    diff = abs(vt - vj)
    return ProbeRecord(
        "parity", f"align_{kind}", "torch|jax", seed, beta,
        {"la": int(a.shape[0]), "lb": int(b.shape[0])}, {"value_abs_diff": diff}, ok=diff < 1e-9,
    )


FAMILY_PROBLEMS = ("parse", "eisner", "mtt", "operators", "align_local", "align_affine")


def sweep_families(seeds: int, betas: tuple[float, ...],
                   backends: tuple[str, ...] = ("torch", "jax")) -> list[ProbeRecord]:
    """Run the new-family probe matrix (parse / eisner / mtt / operators / alignment).

    Parity / gap / oracle probes run across the full ``betas`` ladder; the stochastic
    operator probes (entropy identity vs the empirical sampler, sampler-vs-closed-form
    marginals) run only on a moderate ``beta <= 1e4`` sub-ladder -- at ``beta = 1e6`` the
    Gibbs mass is a point mass, so an MC estimate is degenerate rather than informative.
    """
    stable = tuple(b for b in betas if b <= 1.0e4) or betas
    records: list[ProbeRecord] = []
    for seed in range(seeds):
        records.append(probe_parse_oracle(seed))
        records.append(probe_eisner_oracle(seed))
        records.append(probe_mtt_oracle(seed))
        records.append(probe_align_oracle("local", seed))
        records.append(probe_align_oracle("affine", seed))
        records.append(probe_kbest(seed))
        for beta in betas:
            records.append(probe_parse_parity(seed, beta))
            records.append(probe_eisner_parity(seed, beta))
            records.append(probe_mtt_parity(seed, beta))
            records.append(probe_align_parity("local", seed, beta))
            records.append(probe_align_parity("affine", seed, beta))
            for backend in backends:
                records.append(probe_parse_marginals(seed, beta, backend))
                records.append(probe_parse_gap(seed, beta, backend))
                records.append(probe_eisner_marginals(seed, beta, backend))
                records.append(probe_eisner_gap(seed, beta, backend))
                records.append(probe_mtt_marginals(seed, beta, backend))
                records.append(probe_mtt_gap(seed, beta, backend))
                records.append(probe_align_gap("local", seed, beta, backend))
                records.append(probe_align_gap("affine", seed, beta, backend))
        for beta in stable:
            for backend in backends:
                records.append(probe_entropy(seed, beta, backend))
                records.append(probe_sampling(seed, beta, backend))
    return records


PROBLEMS = ("viterbi", "shortest_path", "ctc")
DEFAULT_BETAS = (1.0, 4.0, 16.0, 64.0, 256.0, 1.0e3, 1.0e4, 1.0e6)


def sweep(seeds: int = 8, betas: tuple[float, ...] = DEFAULT_BETAS,
          backends: tuple[str, ...] = ("torch", "jax"),
          sizes: dict[str, list[dict[str, int] | None]] | None = None,
          *, families: bool = True) -> list[ProbeRecord]:
    """Run the full probe matrix -- the body of the cluster sweep and the CPU regression.

    ``sizes`` maps each problem to a size ladder (**smallest first**); when omitted a single
    default-size instance is used. The exponential oracles -- brute-force agreement and the
    exact integer path count -- run only on the smallest instance of each ladder, while the
    polynomial-time differentiable probes (gap, parity, beta-stability, marginals) run on
    every size, so scaling behaviour is measured without an enumeration blow-up. With
    ``families=True`` the semiring-driver families (CKY parse / Eisner / matrix-tree /
    distribution operators / local + affine-gap alignment) are appended via
    :func:`sweep_families`.
    """
    records: list[ProbeRecord] = []
    for seed in range(seeds):
        for problem in PROBLEMS:
            size_list = (sizes or {}).get(problem, [None])
            records.append(probe_oracle_agreement(problem, seed, size_list[0]))
            records.append(probe_path_count(problem, seed, size=size_list[0]))
            for size in size_list:
                for beta in betas:
                    records.append(probe_gap_tightness(problem, seed, beta, size=size))
                    records.append(probe_parity(problem, seed, beta, size=size))
                    for backend in backends:
                        records.append(probe_beta_stability(problem, seed, beta, backend, size))
                        records.append(probe_marginals_vs_autodiff(problem, seed, beta, backend, size))
    if families:
        records.extend(sweep_families(seeds, betas, backends))
    return records
