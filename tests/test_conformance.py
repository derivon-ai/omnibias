# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""``omnibias.core.conformance`` against any engine: the reference, a broken one, and the JAX and PyTorch front ends."""

from __future__ import annotations

import numpy as np
import pytest
from omnibias.core import conformance as C


def test_the_numpy_reference_engine_conforms_to_the_public_fixtures():
    results = C.run_correctness(C.reference_engine, tolerance=1e-13)
    summary = C.summarize(results)
    assert summary["cases"] == 6 and summary["failed"] == [], summary
    assert {r.quantity for r in results} == {"lap", "divergence", "logdet", "solve"}


def test_a_wrong_engine_and_a_raising_engine_fail_with_their_reasons():
    def off_by_1e6(q, c):
        want = C.reference_engine(q, c)
        return want + 1e-6 * np.maximum(1.0, np.abs(want))  # entrywise relative error 1e-6

    wrong = C.run_correctness(off_by_1e6)
    assert not any(r.passed for r in wrong)
    assert all(r.rel_error == pytest.approx(1e-6, rel=1e-3) for r in wrong)

    def broken(q, c):
        if q == "solve":
            raise NotImplementedError("no solve")
        return C.reference_engine(q, c)

    res = C.run_correctness(broken)
    failed = [r for r in res if not r.passed]
    assert [r.quantity for r in failed] == ["solve"] and "NotImplementedError: no solve" in failed[
        0
    ].error
    nan = C.run_correctness(
        lambda q, c: np.full_like(C.reference_engine(q, c), np.nan), quantities=["lap"]
    )
    assert not any(r.passed for r in nan)


def test_dispatch_regret_reports_median_worst_and_the_gate():
    timings = {"a": {"x": 1.0, "y": 2.0}, "b": {"x": 3.0, "y": 3.3}, "c": {"x": 1.0, "y": 1.0}}
    good = C.dispatch_regret(timings, {"a": "x", "b": "x", "c": "y"})
    assert good["regret"] == {"a": 1.0, "b": 1.0, "c": 1.0} and good["within_limits"]
    bad = C.dispatch_regret(timings, {"a": "y", "b": "y", "c": "x"})
    assert bad["worst"] == 2.0 and bad["median"] == pytest.approx(1.1) and not bad["within_limits"]
    with pytest.raises(ValueError, match="was not timed"):
        C.dispatch_regret(timings, {"a": "z", "b": "x", "c": "x"})


def test_the_fixture_loader_refuses_another_version(tmp_path):
    (tmp_path / "f.json").write_text('{"version": 2, "cases": []}')
    with pytest.raises(ValueError, match="version"):
        C.load_cases(tmp_path / "f.json")


def test_jax_front_end_conforms():
    jax = pytest.importorskip("jax")
    from omnibias.jax.operators import conformance_engine

    before = jax.config.read("jax_enable_x64")
    jax.config.update(
        "jax_enable_x64", True
    )  # the engine refuses float32 rather than conform silently
    try:
        summary = C.summarize(C.run_correctness(conformance_engine(), tolerance=1e-11))
    finally:
        jax.config.update("jax_enable_x64", before)
    assert summary["failed"] == [], summary


def test_torch_front_end_conforms():
    pytest.importorskip("torch")
    from omnibias.torch.operators import conformance_engine

    summary = C.summarize(C.run_correctness(conformance_engine(), tolerance=1e-11))
    assert summary["failed"] == [], summary
