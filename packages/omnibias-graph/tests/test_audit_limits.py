# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
r"""Enforcement: omnibias-graph ships only differentiable relaxations.

Exact NP-hard combinatorial solving is deliberately out of scope. These tests
fail if such a surface is silently added, or if the honesty note is deleted --
mirroring the out-of-thesis enforcement in ``omnibias.geometry.gauge`` / ``omnibias-boolean``
and the ``rsa-limitation`` audit boundary.
"""

from __future__ import annotations

import omnibias.graph

FORBIDDEN = {
    "tsp_solve",
    "traveling_salesman",
    "max_cut",
    "exact_max_cut",
    "min_cut_exact",
    "graph_isomorphism",
    "is_isomorphic",
    "sat_solve",
    "ilp_solve",
    "integer_program",
    "chromatic_number",
    "graph_coloring",
    "maximum_clique",
    "vertex_cover",
    "hamiltonian_cycle",
    "subgraph_isomorphism",
}


def _ops_all(backend: str) -> set[str]:
    mod = __import__(f"omnibias.graph.{backend}.ops", fromlist=["__all__"])
    return set(mod.__all__)


def test_no_np_hard_solver_exported_torch() -> None:
    import pytest

    pytest.importorskip("torch")
    assert _ops_all("torch").isdisjoint(FORBIDDEN)


def test_no_np_hard_solver_exported_jax() -> None:
    import pytest

    pytest.importorskip("jax")
    assert _ops_all("jax").isdisjoint(FORBIDDEN)


def test_backends_export_identical_surface() -> None:
    import pytest

    pytest.importorskip("torch")
    pytest.importorskip("jax")
    assert _ops_all("torch") == _ops_all("jax")


def test_package_docstring_records_scope() -> None:
    doc = (omnibias.graph.__doc__ or "").lower()
    # yes-if framing: relaxations are supported and certified routing lives elsewhere
    assert "relaxation" in doc
    assert "routing" in doc
    # the one honest limit (exactness boundary) is stated, not hidden
    assert "p = np" in doc
    assert "exact" in doc
