# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""The committed operator coverage matrix is complete, vocabulary-clean and in step with its page.

``scripts/generate_coverage_matrix.py`` builds every cell against a dense autodiff oracle (see that file and
``docs/operator-coverage.md``); running it is farm work.  This test reads the committed JSON, which is cheap, and
fails when a cell is a ``mismatch`` or an ``error``, when a status is outside the documented vocabulary, when a
refusal carries no reason, or when ``docs/operator-coverage.md`` is not what the JSON renders to.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "generate_coverage_matrix.py"


@pytest.fixture(scope="module")
def gen():
    spec = importlib.util.spec_from_file_location("generate_coverage_matrix", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def docs(gen):
    found = gen.load_matrices()
    if not found:
        pytest.skip("no committed matrix JSON")
    return found


def test_both_backends_are_committed(docs):
    assert set(docs) == {"jax", "torch"}
    for backend, doc in docs.items():
        assert doc["backend"] == backend
        assert doc["format"] == 1


def test_no_failing_cells_and_only_documented_statuses(gen, docs):
    for backend, doc in docs.items():
        for sec, row, col, c in gen.iter_cells(doc):
            where = f"{backend}/{sec}/{row}/{col}"
            assert c["status"] in gen.STATUSES, where
            assert c["status"] not in ("mismatch", "error"), f"{where}: {c}"


def test_every_served_cell_was_checked_against_the_oracle(gen, docs):
    served = {"structured", "reduced", "fallback", "composed"}
    for backend, doc in docs.items():
        for sec, row, col, c in gen.iter_cells(doc):
            where = f"{backend}/{sec}/{row}/{col}"
            if c["status"] in served:
                assert c.get("algorithm"), where
                assert c.get("error") is not None, where
                assert c["error"] <= gen.TOL, f"{where}: {c['error']}"


def test_refusals_and_gaps_carry_a_reason(gen, docs):
    for backend, doc in docs.items():
        for sec, row, col, c in gen.iter_cells(doc):
            if c["status"] in ("refused", "not_provided"):
                assert c.get("note"), f"{backend}/{sec}/{row}/{col}"


def test_sections_and_rows_are_the_documented_ones(docs):
    sections = {"activations", "architectures", "derivatives", "residuals", "vector_fields"}
    architectures = {
        "mlp_tanh",
        "residual_mlp",
        "fourier_features",
        "gated_pirate",
        "attention",
        "kan_edges",
    }
    for backend, doc in docs.items():
        assert set(doc["sections"]) == sections, backend
        assert architectures <= set(doc["sections"]["architectures"]), backend
        assert len(doc["sections"]["activations"]) >= 40, backend
    assert set(docs["jax"]["sections"]["activations"]) == set(
        docs["torch"]["sections"]["activations"]
    )


def test_page_is_the_rendering_of_the_json(gen, docs):
    assert (ROOT / "docs" / "operator-coverage.md").read_text() == gen.build_page(
        docs
    ), "run `python scripts/generate_coverage_matrix.py render` and commit the page"
