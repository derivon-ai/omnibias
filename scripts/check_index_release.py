# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Download and verify an entire prepared cohort from one explicit package index."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import urlopen

from release_artifacts import MANIFEST, ROOT, missing_files, verify_files
from release_preflight import index_project


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", required=True, choices=["pypi", "testpypi"])
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    commit = json.loads((args.prepared / MANIFEST).read_text())["commit"]
    data = verify_files(ROOT, args.prepared, commit)
    report = []
    for package in data["packages"].values():
        published = index_project(package["name"], args.repository)
        files = [] if published is None else published.get("releases", {}).get(package["version"], [])
        if missing_files(package["files"], files):
            raise ValueError(f"Incomplete release: {package['name']}")
        for record in files:
            url = urlsplit(record["url"])
            if url.scheme != "https" or url.hostname not in {"files.pythonhosted.org", "test-files.pythonhosted.org"}:
                raise ValueError("Unexpected index artifact origin")
            name = record["filename"]
            target = args.output / ("wheels" if name.endswith(".whl") else "sdists") / name
            target.parent.mkdir(parents=True, exist_ok=True)
            with urlopen(record["url"], timeout=60) as response:
                payload = response.read()
            if hashlib.sha256(payload).hexdigest() != package["files"][name]:
                raise ValueError(f"Downloaded bytes differ: {name}")
            target.write_bytes(payload)
            report.append({"filename": name, "url": record["url"], "sha256": package["files"][name]})
        print(f"Verified {package['name']} {package['version']} on {args.repository}", flush=True)
    (args.output / "index-artifacts.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
