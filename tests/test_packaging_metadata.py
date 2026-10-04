# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Exercise Warehouse constraints that local README rendering does not enforce."""
import importlib.util
import io
import tarfile
import zipfile
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "check_packaging", Path(__file__).resolve().parents[1] / "scripts/check_packaging.py"
)
assert spec and spec.loader
packaging = importlib.util.module_from_spec(spec)
spec.loader.exec_module(packaging)


@pytest.mark.parametrize("length,accepted", [(512, True), (513, False), (550, False)])
def test_summary_limit_in_both_distribution_formats(tmp_path, length, accepted):
    payload = ("Metadata-Version: 2.4\nName: omnibias-example\nVersion: 0.1.0a1\n"
               + "Summary: " + "x" * length + "\n\nREADME\n").encode()
    wheel = tmp_path / "example.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("omnibias_example-0.1.0a1.dist-info/METADATA", payload)
    sdist = tmp_path / "example.tar.gz"
    with tarfile.open(sdist, "w:gz") as archive:
        entry = tarfile.TarInfo("example/PKG-INFO")
        entry.size = len(payload)
        archive.addfile(entry, io.BytesIO(payload))
    assert (packaging._wheel_offenders(wheel) == []) is accepted
    assert (packaging._sdist_offenders(sdist) == []) is accepted
