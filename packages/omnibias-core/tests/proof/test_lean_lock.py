# SPDX-License-Identifier: Apache-2.0
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import pytest
from omnibias.core.proof.lean_lock import generated_lean_obligation


def test_generated_obligation_restores_on_failure_and_absent_file(tmp_path):
    generated = tmp_path / "Generated.lean"
    with pytest.raises(RuntimeError), generated_lean_obligation(tmp_path, generated, "temporary"):
        assert generated.read_text() == "temporary"
        raise RuntimeError("build failure")
    assert not generated.exists()
    generated.write_text("original")
    with generated_lean_obligation(tmp_path, generated, "new"):
        assert generated.read_text() == "new"
    assert generated.read_text() == "original"


def test_threads_cannot_overwrite_another_active_obligation(tmp_path):
    generated = tmp_path / "Generated.lean"
    generated.write_text("original")

    def worker(label):
        with generated_lean_obligation(tmp_path, generated, label):
            time.sleep(0.025)
            assert generated.read_text() == label

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(worker, ["first", "second", "third", "fourth"]))
    assert generated.read_text() == "original"


def test_separate_processes_share_the_project_lock(tmp_path):
    generated = tmp_path / "Generated.lean"
    generated.write_text("original")
    code = """
import sys, time
from pathlib import Path
from omnibias.core.proof.lean_lock import generated_lean_obligation
root=Path(sys.argv[1]); generated=root/'Generated.lean'; label=sys.argv[2]
with generated_lean_obligation(root, generated, label):
    time.sleep(0.15)
    assert generated.read_text()==label
"""
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(
                subprocess.run,
                [sys.executable, "-c", code, str(tmp_path), label],
                capture_output=True,
                text=True,
                check=True,
            )
            for label in ("first", "second")
        ]
        for future in futures:
            future.result(timeout=20)
    assert generated.read_text() == "original"
