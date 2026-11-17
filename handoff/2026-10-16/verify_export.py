"""Verify a handoff ZIP using only Python's standard library; never extract it."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import stat
import time
import zipfile
from pathlib import Path, PurePosixPath


def safe_name(name: str) -> bool:
    parts = PurePosixPath(name).parts
    return (
        bool(parts)
        and not name.startswith("/")
        and "\\" not in name
        and all(part not in {"", ".", ".."} and ":" not in part for part in name.split("/"))
    )


def verify(path: Path) -> dict:
    started = time.time()
    verified_bytes = 0
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or not all(safe_name(n) for n in names):
            raise ValueError("Duplicate or unsafe archive paths")
        manifests = [n for n in names if n.endswith("/MANIFEST.json")]
        if len(manifests) != 1:
            raise ValueError("Expected exactly one export manifest")
        manifest_name = manifests[0]
        if archive.getinfo(manifest_name).file_size > 128 * 1024 * 1024:
            raise ValueError("Manifest exceeds 128 MiB")
        manifest = json.loads(archive.read(manifest_name))
        if manifest.get("schema") != "omnibias-research-transfer/v1":
            raise ValueError("Unsupported manifest schema")
        rows = manifest["files"]
        expected = {r["archive_path"] for r in rows}
        if len(expected) != len(rows) or expected | {manifest_name} != set(names):
            raise ValueError("Manifest does not cover the exact archive membership")
        unstable = []
        for index, row in enumerate(rows, 1):
            name, size, expected_sha = row["archive_path"], row["bytes"], row["sha256"]
            if type(size) is not int or size < 0 or not re.fullmatch(r"[0-9a-f]{64}", expected_sha):
                raise ValueError(f"Invalid record: {name}")
            info = archive.getinfo(name)
            if info.is_dir() or stat.S_ISLNK(info.external_attr >> 16) or info.file_size != size:
                raise ValueError(f"Size/type mismatch: {name}")
            digest = hashlib.sha256()
            count = 0
            with archive.open(name) as source:
                while chunk := source.read(1024 * 1024):
                    digest.update(chunk)
                    count += len(chunk)
            if count != size or digest.hexdigest() != expected_sha:
                raise ValueError(f"Checksum mismatch: {name}")
            verified_bytes += count
            if row.get("stable_during_copy") is False:
                unstable.append(name)
            if index % 10000 == 0:
                print(json.dumps({"verified_files": index, "bytes": verified_bytes}), flush=True)
    return {
        "schema": "omnibias-research-transfer-verification/v1",
        "archive": str(path.resolve()),
        "integrity_passed": True,
        "files_verified": len(rows),
        "bytes_verified": verified_bytes,
        "unstable_origin_entries": unstable,
        "snapshot_origin_consistency_passed": not unstable,
        "elapsed_s": time.time() - started,
        "numerical_or_benchmark_validation": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    result = verify(args.archive)
    rendered = json.dumps(result, indent=2) + "\n"
    if args.report:
        with args.report.open("x", encoding="utf-8") as output:
            output.write(rendered)
    print(rendered, end="")
    if not result["snapshot_origin_consistency_passed"]:
        raise SystemExit("Origin changed during export; inspect unstable entries")


if __name__ == "__main__":
    main()
