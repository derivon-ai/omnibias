# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Promotion is tied to a reviewed commit and exact, recoverable artifact bytes."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from urllib.error import URLError

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("release_artifacts", ROOT / "scripts/release_artifacts.py")
assert spec and spec.loader
release = importlib.util.module_from_spec(spec)
sys.path.insert(0, str(ROOT / "scripts"))
try:
    spec.loader.exec_module(release)
finally:
    sys.path.pop(0)

SHA = "a" * 40


def good_run():
    return {"head_sha": SHA, "head_branch": "main", "event": "workflow_dispatch",
            "status": "completed", "conclusion": "success", "path": release.WORKFLOW,
            "repository": {"full_name": release.REPOSITORY}}


def test_successful_preparation():
    release.verify_run(good_run(), [{"name": "build primitive distributions", "conclusion": "success"}], SHA)


@pytest.mark.parametrize("key,value", [("head_sha", "b" * 40), ("head_branch", "other"),
    ("event", "pull_request"), ("conclusion", "failure"), ("path", "untrusted.yml"),
    ("repository", {"full_name": "other/repository"})])
def test_wrong_preparation_refused(key, value):
    with pytest.raises(ValueError):
        release.verify_run({**good_run(), key: value}, [], SHA)


def test_publish_run_cannot_substitute_for_preparation():
    with pytest.raises(ValueError, match="build"):
        release.verify_run(good_run(), [{"name": "build primitive distributions", "conclusion": "skipped"}], SHA)


def record(name="a.whl", sha="abc", **kw):
    return {"filename": name, "digests": {"sha256": sha}, **kw}


def test_partial_upload_recovers_only_missing_files():
    assert release.missing_files({"a.whl": "abc", "a.tar.gz": "def"}, [record()]) == ["a.tar.gz"]
    assert release.missing_files({"a.whl": "abc"}, [record()]) == []


@pytest.mark.parametrize("existing", [[record(sha="wrong")], [record(yanked=True)],
                                      [record("unexpected.whl")], [record(), record()]])
def test_conflicting_index_artifacts_are_refused(existing):
    with pytest.raises(ValueError, match="conflicts"):
        release.missing_files({"a.whl": "abc"}, existing)


def fixture_cohort(tmp_path, monkeypatch):
    project = {"name": "omnibias-core", "version": "0.5.0rc1"}
    monkeypatch.setattr(release, "projects", lambda root: {"core": project})
    for suffix in ["-py3-none-any.whl", ".tar.gz"]:
        (tmp_path / ("omnibias_core-0.5.0rc1" + suffix)).write_bytes(b"reviewed artifact")
    data = release.manifest(tmp_path, tmp_path, SHA)
    (tmp_path / release.MANIFEST).write_text(json.dumps(data))
    return data


def test_manifest_detects_modified_bytes(tmp_path, monkeypatch):
    data = fixture_cohort(tmp_path, monkeypatch)
    assert release.verify_files(tmp_path, tmp_path, SHA) == data
    next(tmp_path.glob("*.whl")).write_bytes(b"tampered")
    with pytest.raises(ValueError, match="hashes"):
        release.verify_files(tmp_path, tmp_path, SHA)


def test_manifest_rejects_extra_files(tmp_path, monkeypatch):
    fixture_cohort(tmp_path, monkeypatch)
    (tmp_path / "unknown").touch()
    with pytest.raises(ValueError, match="Unexpected"):
        release.verify_files(tmp_path, tmp_path, SHA)


def test_dependency_must_already_be_complete_on_target(monkeypatch):
    monkeypatch.setattr(release, "projects", lambda root: {
        "torch": {"name": "omnibias-torch", "dependencies": ["omnibias-core>=0.5.0rc1"],
                  "optional-dependencies": {"all": ["omnibias-torch"]}}})
    data = {"packages": {"core": {"name": "omnibias-core", "version": "0.5.0rc1",
                                  "files": {"a.whl": "abc", "a.tar.gz": "def"}}}}
    monkeypatch.setattr(release, "index_project", lambda *a: {"releases": {"0.5.0rc1": [record()]}})
    with pytest.raises(ValueError, match="before torch"):
        release.verify_dependencies(ROOT, ["torch"], "testpypi", data)
    monkeypatch.setattr(release, "index_project", lambda *a: {"releases": {"0.5.0rc1": [record(), record("a.tar.gz", "def")]}})
    release.verify_dependencies(ROOT, ["torch"], "testpypi", data)


def test_resume_requires_matching_files_and_explicit_flag(tmp_path, monkeypatch):
    data = fixture_cohort(tmp_path, monkeypatch)
    files = data["packages"]["core"]["files"]
    name = next(iter(files))
    monkeypatch.setattr(release, "index_project", lambda *a: {"releases": {"0.5.0rc1": [record(name, files[name])]}})
    output = tmp_path / "upload"
    with pytest.raises(ValueError, match="resume"):
        release.stage(ROOT, tmp_path, output, data, ["core"], "testpypi", False)
    matrix = release.stage(ROOT, tmp_path, output, data, ["core"], "testpypi", True)
    assert matrix == [{"package": "core", "environment": "testpypi-core"}]
    assert {p.name for p in (output / "core").iterdir()} == set(files) - {name}


def test_index_outage_is_not_an_empty_release(tmp_path, monkeypatch):
    data = fixture_cohort(tmp_path, monkeypatch)
    def outage(*args):
        raise URLError("offline")
    monkeypatch.setattr(release, "index_project", outage)
    with pytest.raises(URLError):
        release.stage(ROOT, tmp_path, tmp_path / "upload", data, ["core"], "pypi", True)


def test_repair_reuses_only_identical_package_trees(monkeypatch):
    monkeypatch.setattr(release, "projects", lambda root: {
        "core": {"name": "omnibias-core", "version": "0.5.0rc1"},
        "partition": {"name": "omnibias-partition", "version": "0.1.0a2"},
    })
    monkeypatch.setattr(release.subprocess, "run", lambda *a, **kw: None)
    monkeypatch.setattr(release.subprocess, "check_output", lambda command, **kw:
                        "packages/omnibias-partition/pyproject.toml\n" if command[-1].endswith("partition") else "")
    checked = []
    monkeypatch.setattr(release, "ensure_unpublished", lambda *args: checked.append(args))
    assert release.repair_plan(ROOT, "b" * 40, SHA) == (["core"], ["partition"])
    assert checked == [("omnibias-partition", "0.1.0a2", index) for index in ("testpypi", "pypi")]


@pytest.mark.parametrize("error", [ValueError("version occupied"), URLError("index unavailable")])
def test_repair_never_rebuilds_an_occupied_or_unknown_version(monkeypatch, error):
    monkeypatch.setattr(release, "projects", lambda root: {"core": {"name": "omnibias-core", "version": "0.5.0rc1"}})
    monkeypatch.setattr(release.subprocess, "run", lambda *a, **kw: None)
    monkeypatch.setattr(release.subprocess, "check_output", lambda *a, **kw: "changed")
    def unavailable(*args):
        raise error
    monkeypatch.setattr(release, "ensure_unpublished", unavailable)
    with pytest.raises(type(error)):
        release.repair_plan(ROOT, "b" * 40, SHA)


def test_repair_rejects_unrelated_history(monkeypatch):
    def unrelated(*args, **kwargs):
        raise release.subprocess.CalledProcessError(1, "git merge-base")
    monkeypatch.setattr(release.subprocess, "run", unrelated)
    with pytest.raises(release.subprocess.CalledProcessError):
        release.repair_plan(ROOT, "b" * 40, SHA)


def test_repair_keeps_original_bytes_and_records_origin(tmp_path, monkeypatch):
    available = {"core": {"name": "omnibias-core", "version": "0.5.0rc1"},
                 "partition": {"name": "omnibias-partition", "version": "0.1.0a2"}}
    monkeypatch.setattr(release, "projects", lambda root: available)
    monkeypatch.setattr(release, "gh", lambda *args: json.dumps({"head_sha": "b" * 40}))
    def write_artifacts(directory, project, payload):
        directory.mkdir(parents=True, exist_ok=True)
        for suffix in ("-py3-none-any.whl", ".tar.gz"):
            name = project["name"].replace("-", "_") + "-" + project["version"] + suffix
            (directory / name).write_bytes(payload)
    verified = []
    def fetch(run_id, directory, commit):
        verified.append((run_id, commit))
        for project in available.values():
            write_artifacts(directory, project, b"original signed bytes")
        return release.manifest(tmp_path, directory, commit)
    monkeypatch.setattr(release, "fetch_prepared", fetch)
    monkeypatch.setattr(release, "repair_plan", lambda *args: (["core"], ["partition"]))
    built = []
    def build(command, **kwargs):
        built.append(command[3])
        write_artifacts(Path(command[-1]), available["partition"], b"corrected metadata")
    monkeypatch.setattr(release.subprocess, "run", build)
    directory = tmp_path / "dist"
    release.build_cohort(tmp_path, directory, SHA, "123")
    data = release.verify_files(tmp_path, directory, SHA)
    assert verified == [("123", "b" * 40)]
    assert built == [str(tmp_path / "packages/omnibias-partition")]
    assert data["reuse"] == {"run_id": "123", "commit": "b" * 40, "packages": ["core"]}
    assert next(directory.glob("omnibias_core-*.whl")).read_bytes() == b"original signed bytes"
    assert next(directory.glob("omnibias_partition-*.whl")).read_bytes() == b"corrected metadata"


def test_repair_requires_verified_previous_artifacts(tmp_path, monkeypatch):
    monkeypatch.setattr(release, "gh", lambda *args: json.dumps({"head_sha": "b" * 40}))
    def unverified(*args):
        raise ValueError("Invalid signature")
    monkeypatch.setattr(release, "fetch_prepared", unverified)
    with pytest.raises(ValueError, match="signature"):
        release.build_cohort(ROOT, tmp_path / "dist", SHA, "123")
    assert not list((tmp_path / "dist").iterdir())
