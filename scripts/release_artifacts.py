# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Prepare immutable release cohorts and promote only verified artifact bytes."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from packaging.requirements import Requirement
from packaging.version import Version
from release_preflight import (
    ROOT,
    ensure_unpublished,
    index_project,
    projects,
    publisher_environment,
    require_prerelease,
    select,
)

REPOSITORY = "derivon-ai/omnibias"
WORKFLOW = ".github/workflows/release.yml"
MANIFEST = "release-manifest.json"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def manifest(root: Path, directory: Path, commit: str, reuse: dict | None = None) -> dict:
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("Expected a full source commit")
    cohort = {}
    for short, project in projects(root).items():
        prefix = project["name"].replace("-", "_") + "-" + project["version"]
        files = [prefix + "-py3-none-any.whl", prefix + ".tar.gz"]
        cohort[short] = {
            "name": project["name"], "version": project["version"],
            "files": {name: digest(directory / name) for name in files},
        }
    data = {"schema": 1, "repository": REPOSITORY, "commit": commit, "packages": cohort}
    if reuse is not None:
        if (set(reuse) != {"run_id", "commit", "packages"}
                or not isinstance(reuse["run_id"], str) or not reuse["run_id"].isdecimal()
                or not isinstance(reuse["commit"], str)
                or not re.fullmatch(r"[0-9a-f]{40}", reuse["commit"])
                or not isinstance(reuse["packages"], list)
                or not all(isinstance(short, str) for short in reuse["packages"])
                or reuse["packages"] != sorted(set(reuse["packages"]))
                or not set(reuse["packages"]) <= set(cohort)):
            raise ValueError("Invalid artifact reuse provenance")
        data["reuse"] = reuse
    return data


def verify_files(root: Path, directory: Path, commit: str) -> dict:
    data = json.loads((directory / MANIFEST).read_text())
    expected = manifest(root, directory, commit, data.get("reuse"))
    if data != expected:
        raise ValueError("Prepared manifest differs from source metadata or artifact hashes")
    names = {name for p in data["packages"].values() for name in p["files"]} | {MANIFEST}
    if {p.name for p in directory.iterdir()} != names:
        raise ValueError("Unexpected files in prepared artifact bundle")
    return data


def repair_plan(root: Path, previous_commit: str, commit: str) -> tuple[list[str], list[str]]:
    """Reuse only identical package trees from an ancestor; rebuild only unused versions."""
    subprocess.run(["git", "merge-base", "--is-ancestor", previous_commit, commit],
                   cwd=root, check=True)
    reused, rebuilt = [], []
    for short, project in sorted(projects(root).items()):
        path = f"packages/{project['name']}"
        changed = subprocess.check_output(
            ["git", "diff", "--name-only", previous_commit, commit, "--", path],
            cwd=root, text=True,
        ).strip()
        if changed:
            for repository in ("testpypi", "pypi"):
                ensure_unpublished(project["name"], project["version"], repository)
            rebuilt.append(short)
        else:
            reused.append(short)
    return reused, rebuilt


def build_cohort(root: Path, directory: Path, commit: str, reuse_run_id: str = "") -> None:
    directory.mkdir(parents=True, exist_ok=True)
    if any(directory.iterdir()):
        raise ValueError("Build output directory must be empty")
    available = projects(root)
    rebuilt = sorted(available)
    reuse = None
    if reuse_run_id:
        if not reuse_run_id.isdecimal():
            raise ValueError("Reuse requires a numeric preparation run ID")
        run = json.loads(gh("api", f"repos/{REPOSITORY}/actions/runs/{reuse_run_id}"))
        previous_commit = run["head_sha"]
        previous = directory.parent / "artifacts" / "previous-primitive-dist"
        # fetch_prepared verifies repository/workflow/ref, every hash, and all
        # attestations. It also rejects changes to cohort names or versions.
        data = fetch_prepared(reuse_run_id, previous, previous_commit)
        reused, rebuilt = repair_plan(root, previous_commit, commit)
        for short in reused:
            for name in data["packages"][short]["files"]:
                shutil.copyfile(previous / name, directory / name)
        reuse = {"run_id": reuse_run_id, "commit": previous_commit, "packages": reused}
    for short in rebuilt:
        subprocess.run([sys.executable, "-m", "build", str(root / "packages" / available[short]["name"]),
                        "--outdir", str(directory.resolve())], cwd=root, check=True)
    data = manifest(root, directory, commit, reuse)
    (directory / MANIFEST).write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    verify_files(root, directory, commit)
    print(json.dumps({"rebuilt": rebuilt, "reuse": reuse}), flush=True)


def verify_run(run: dict, jobs: list[dict], commit: str) -> None:
    expected = {
        "head_sha": commit, "head_branch": "main", "event": "workflow_dispatch",
        "status": "completed", "conclusion": "success", "path": WORKFLOW,
    }
    if any(run.get(key) != value for key, value in expected.items()):
        raise ValueError("Preparation must be a successful main-branch workflow at this exact commit")
    if run.get("repository", {}).get("full_name") != REPOSITORY:
        raise ValueError("Preparation belongs to another repository")
    if not any(j.get("name") == "build primitive distributions" and j.get("conclusion") == "success" for j in jobs):
        raise ValueError("Run did not successfully build the prepared cohort")


def verify_dependencies(root: Path, names: list[str], repository: str, data: dict) -> None:
    """Publication requires complete, exact cohort dependencies already on the index."""
    available = projects(root)
    cohort = {p["name"]: p for p in data["packages"].values()}
    cache = {}
    for short in names:
        project = available[short]
        requirements = list(project.get("dependencies", []))
        requirements += [r for group in project.get("optional-dependencies", {}).values() for r in group]
        for raw in requirements:
            req = Requirement(raw)
            if not req.name.startswith("omnibias-") or req.name == project["name"]:
                continue
            dependency = cohort[req.name]
            if not req.specifier.contains(Version(dependency["version"]), prereleases=True):
                raise ValueError(f"{short}: prepared dependency does not satisfy {raw}")
            if req.name not in cache:
                cache[req.name] = index_project(req.name, repository)
            published = cache[req.name]
            existing = [] if published is None else published.get("releases", {}).get(dependency["version"], [])
            if missing_files(dependency["files"], existing):
                raise ValueError(f"Publish complete dependency {req.name} before {short} on {repository}")


def missing_files(expected: dict[str, str], existing: list[dict]) -> list[str]:
    seen = set()
    for file in existing:
        name = file["filename"]
        if name in seen or name not in expected or file.get("yanked") or file.get("digests", {}).get("sha256") != expected[name]:
            raise ValueError(f"Existing release artifact conflicts with prepared bytes: {name}")
        seen.add(name)
    return sorted(set(expected) - seen)


def gh(*args: str) -> str:
    return subprocess.check_output(["gh", *args], text=True)


def fetch_prepared(run_id: str, directory: Path, commit: str) -> dict:
    if not run_id.isdecimal():
        raise ValueError("prepared_run_id must be a numeric workflow run ID")
    run = json.loads(gh("api", f"repos/{REPOSITORY}/actions/runs/{run_id}"))
    jobs = json.loads(gh("api", f"repos/{REPOSITORY}/actions/runs/{run_id}/jobs?per_page=100"))["jobs"]
    verify_run(run, jobs, commit)
    directory.mkdir(parents=True, exist_ok=True)
    if any(directory.iterdir()):
        raise ValueError("Prepared download directory must be empty")
    gh("run", "download", run_id, "--repo", REPOSITORY, "--name", "primitive-dist", "--dir", str(directory))
    data = verify_files(ROOT, directory, commit)
    def attest(path: Path) -> None:
        gh("attestation", "verify", str(path), "--repo", REPOSITORY,
           "--signer-workflow", REPOSITORY + "/" + WORKFLOW,
           "--source-ref", "refs/heads/main", "--source-digest", commit,
           "--deny-self-hosted-runners")
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(attest, sorted(directory.iterdir())))
    return data


def stage(root: Path, directory: Path, output: Path, data: dict, names: list[str], repository: str, resume: bool) -> list[dict]:
    verify_dependencies(root, names, repository, data)
    matrix = []
    for short in names:
        package = data["packages"][short]
        published = index_project(package["name"], repository)
        existing = [] if published is None else published.get("releases", {}).get(package["version"], [])
        missing = missing_files(package["files"], existing)
        if existing and not resume:
            raise ValueError(f"{short}: version occupied; use resume only for hash-verified recovery")
        if not missing:
            print(f"Already complete with matching hashes: {short}")
            continue
        target = output / short
        target.mkdir(parents=True, exist_ok=False)
        for name in missing:
            shutil.copyfile(directory / name, target / name)
        matrix.append({"package": short, "environment": publisher_environment(short, repository)})
    return matrix


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["build", "verify", "manifest", "promote"])
    parser.add_argument("--directory", type=Path, default=Path("dist"))
    parser.add_argument("--output", type=Path, default=Path("upload"))
    args = parser.parse_args()
    commit = os.environ["GITHUB_SHA"]
    if args.mode == "build":
        build_cohort(ROOT, args.directory, commit, os.environ.get("REUSE_PREPARED_RUN_ID", ""))
        return
    if args.mode == "verify":
        verify_files(ROOT, args.directory, commit)
        return
    if args.mode == "manifest":
        data = manifest(ROOT, args.directory, commit)
        (args.directory / MANIFEST).write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
        verify_files(ROOT, args.directory, commit)
        return
    if os.environ.get("GITHUB_REF") != "refs/heads/main":
        raise ValueError("Publication is restricted to main")
    repository = os.environ["RELEASE_REPOSITORY"]
    if repository not in {"pypi", "testpypi"}:
        raise ValueError("Choose an explicit publication index")
    names = select(ROOT, os.environ.get("REQUESTED_PACKAGES", ""), "", "workflow_dispatch")
    require_prerelease(ROOT, names)
    data = fetch_prepared(os.environ["PREPARED_RUN_ID"], args.directory, commit)
    matrix = stage(ROOT, args.directory, args.output, data, names, repository,
                   os.environ.get("RESUME_UPLOAD") == "true")
    with Path(os.environ["GITHUB_OUTPUT"]).open("a") as output:
        print("matrix=" + json.dumps({"include": matrix}), file=output)
        print("pending=" + str(bool(matrix)).lower(), file=output)


if __name__ == "__main__":
    main()
