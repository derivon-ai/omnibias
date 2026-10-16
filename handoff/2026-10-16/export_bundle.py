"""Export a clean Git checkpoint and research tree without running numerical code.

Live numbered attempt files at/after --main-cutoff are excluded. Append-only
process manifests are copied as complete, cutoff-filtered line prefixes. The
result is a filewise snapshot, not an atomic or scientifically audited result.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import stat
import subprocess
import time
import zipfile
from pathlib import Path

PREFIX = "omnibias-gpu-handoff-20261016"
SKIP_DIRS = {"__pycache__", ".git", ".venv", ".pytest_cache", ".mypy_cache", ".ruff_cache"}


def utc() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def excluded_live(relative: Path, cutoff: int) -> bool:
    parts = relative.parts
    if len(parts) >= 3 and parts[:2] == ("completion_execution_20261013", "matched"):
        match = re.match(r"^(\d{5})(?:\.|$)", parts[2])
        return match is not None and int(match[1]) >= cutoff
    return False


def signature(value: os.stat_result) -> list[int]:
    return [value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns]


def iter_files(root: Path):
    for parent, dirs, names in os.walk(root, followlinks=False):
        dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS)
        for name in sorted(names):
            path = Path(parent) / name
            metadata = path.lstat()
            if stat.S_ISREG(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
                yield path
            else:
                raise ValueError(f"Unsupported filesystem entry: {path}")


def copy_file(archive: zipfile.ZipFile, source: Path, arcname: str, *, cutoff: int) -> dict:
    target = source.resolve(strict=True)
    link = os.readlink(source) if source.is_symlink() else None
    metadata = target.stat()
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"Not a regular file: {source}")
    prefix_log = source.name.endswith("process_manifest.jsonl") and (
        source.parent.name == "completion_execution_20261013"
    )
    digest = hashlib.sha256()
    copied = 0
    skipped_rows = 0
    with target.open("rb") as reader:
        before = os.fstat(reader.fileno())
        with archive.open(arcname, "w", force_zip64=True) as writer:
            if prefix_log:
                # Snapshot only the original extent, never chase an appending file.
                remaining = before.st_size
                while remaining:
                    line = reader.readline(remaining)
                    remaining -= len(line)
                    if not line.endswith(b"\n"):
                        skipped_rows += 1
                        break
                    row = json.loads(line)
                    if type(row.get("index")) is not int:
                        raise ValueError(f"Unrecognized process-manifest row: {source}")
                    # Each original schedule has separate indices. Only the main
                    # protocol prefix uses main_cutoff; other complete rows stay.
                    if source.name == "protocol_process_manifest.jsonl" and row["index"] >= cutoff:
                        skipped_rows += 1
                        continue
                    writer.write(line)
                    digest.update(line)
                    copied += len(line)
            else:
                remaining = before.st_size
                while remaining:
                    chunk = reader.read(min(1024 * 1024, remaining))
                    if not chunk:
                        raise OSError(f"File truncated during export: {source}")
                    writer.write(chunk)
                    digest.update(chunk)
                    copied += len(chunk)
                    remaining -= len(chunk)
            after = os.fstat(reader.fileno())
    final_target = source.resolve(strict=True)
    final_path_metadata = target.stat()
    stable = (
        final_target == target
        and signature(before) == signature(after) == signature(final_path_metadata)
        and (link is None or os.readlink(source) == link)
    )
    if prefix_log:
        # Only appends are permissible; copied complete lines are independent
        # frozen bytes. Record the scope instead of claiming a whole-file freeze.
        stable = (
            final_target == target
            and before.st_dev == after.st_dev == final_path_metadata.st_dev
            and before.st_ino == after.st_ino == final_path_metadata.st_ino
            and after.st_size >= before.st_size
            and final_path_metadata.st_size >= before.st_size
        )
    return {
        "archive_path": arcname,
        "origin": str(source),
        "bytes": copied,
        "sha256": digest.hexdigest(),
        "stable_during_copy": stable,
        "origin_signature_before": signature(before),
        "origin_signature_after": signature(after),
        "snapshot_policy": "complete-process-manifest-prefix"
        if prefix_log
        else "fixed-original-extent",
        "omitted_process_manifest_rows": skipped_rows,
        "materialized_original_symlink": link,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--research", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--main-cutoff", type=int, required=True)
    parser.add_argument("--cutoff-observed-utc", required=True)
    args = parser.parse_args()
    repo, research, output = args.repo.resolve(), args.research.resolve(), args.output.resolve()
    if not 0 <= args.main_cutoff <= 10548:
        raise ValueError("Invalid original main-schedule cutoff")
    if research == output or research in output.parents:
        raise ValueError("Output must be outside the research input tree")
    if git(repo, "status", "--porcelain"):
        raise ValueError("Commit the checkpoint before export; working tree must be clean")
    commit = git(repo, "rev-parse", "HEAD")
    branch = git(repo, "branch", "--show-current")
    if not branch:
        raise ValueError("Named checkpoint branch required")
    output.mkdir(parents=True, exist_ok=False)
    started = utc()
    state = {
        "schema": "omnibias-transfer-checkpoint/v1",
        "created_utc": started,
        "git_commit": commit,
        "git_branch": branch,
        "remote": git(repo, "remote", "get-url", "origin"),
        "repo_origin": str(repo),
        "research_origin": str(research),
        "laptop_schedule": {
            "status_at_preparation": "running",
            "local_exec_session_id": 24105,
            "main_records": 10548,
            "cache_records": 36,
            "pde_records": 234,
            "filename_only_cutoff": args.main_cutoff,
            "cutoff_observed_utc": args.cutoff_observed_utc,
            "cutoff_meaning": "Existence under serial-prefix assumption; NOT pass/status/terminal/audit acceptance",
            "future_and_in_progress_attempts_included": False,
            "all_schedules_terminal": False,
            "restart_or_replacement": False,
        },
        "target": {
            "scheduler": "LSF",
            "gpus": "about20 shared A100 allocations,20–80GB; actual target binding pending",
        },
        "current_final_candidate_acceptance": {
            "numerical": False,
            "scientific": False,
            "resource": False,
            "performance": False,
            "public_integration": False,
        },
        "packaging_contention": {
            "start_utc": started,
            "end": "After independent archive verification; see transfer_receipt.json beside ZIP",
            "scope": "Export runs during old laptop schedule; overlapping observations cannot be assumed quiet final evidence",
            "execution_policy": "single CPU7, nice19, idle I/O priority where supported",
        },
        "omission_policy": {
            "excluded_directory_names": sorted(SKIP_DIRS),
            "live_main_attempts": f"completion_execution_20261013/matched indices >= {args.main_cutoff}",
            "future_outputs": "Created after inventory/cutoff are not part of this export",
            "environments": "Installed virtualenv/CUDA/tool binaries outside research tree are not exported; use lock/environment records",
            "external_projects": "Sibling omnibias_projects repos are not included except any existing frozen copies inside research",
            "symlinks": "Research-internal file symlinks are materialized; original target recorded per file",
        },
    }
    (output / "export_state.json").write_text(json.dumps(state, indent=2) + "\n")
    bundle = output / "omnibias.bundle"
    subprocess.run(
        ["git", "-C", str(repo), "bundle", "create", str(bundle), f"refs/heads/{branch}"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(repo), "bundle", "verify", str(bundle)], check=True, capture_output=True
    )
    research_files = []
    omissions = []
    for path in iter_files(research):
        relative = path.relative_to(research)
        if excluded_live(relative, args.main_cutoff):
            omissions.append(
                {"path": str(relative), "reason": "live-main-attempt-at-or-after-cutoff"}
            )
            continue
        if path.is_symlink() and research not in path.resolve(strict=True).parents:
            raise ValueError(f"External research symlink must be reviewed: {path}")
        research_files.append(
            (path, f"{PREFIX}/research/bias-collapse-research/{relative.as_posix()}")
        )
    (output / "excluded_live_files.json").write_text(json.dumps(omissions, indent=2) + "\n")
    handoff_root = repo / "handoff" / "2026-10-16"
    handoff_files = [
        (p, f"{PREFIX}/handoff/{p.name}") for p in sorted(handoff_root.iterdir()) if p.is_file()
    ]
    inputs = (
        handoff_files
        + [
            (bundle, f"{PREFIX}/git/omnibias.bundle"),
            (output / "export_state.json", f"{PREFIX}/export_state.json"),
            (output / "excluded_live_files.json", f"{PREFIX}/excluded_live_files.json"),
        ]
        + sorted(research_files, key=lambda r: r[1])
    )
    archive_path = output / f"{PREFIX}.zip"
    rows = []
    total_bytes = 0
    clock = time.monotonic()
    with zipfile.ZipFile(
        archive_path, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True
    ) as archive:
        for index, (source, name) in enumerate(inputs, 1):
            row = copy_file(archive, source, name, cutoff=args.main_cutoff)
            rows.append(row)
            total_bytes += row["bytes"]
            if index % 5000 == 0 or time.monotonic() - clock >= 30:
                print(
                    json.dumps(
                        {
                            "files": index,
                            "total_files": len(inputs),
                            "uncompressed_bytes": total_bytes,
                            "utc": utc(),
                        }
                    ),
                    flush=True,
                )
                clock = time.monotonic()
        manifest = {
            "schema": "omnibias-research-transfer/v1",
            "git_commit": commit,
            "git_branch": branch,
            "copy_started_utc": started,
            "copy_finished_utc": utc(),
            "snapshot_scope": "Filewise export; incomplete original laptop schedule; no new numerical/performance acceptance",
            "files": rows,
            "files_count": len(rows),
            "total_uncompressed_bytes": total_bytes,
            "unstable_origin_entries": [
                r["archive_path"] for r in rows if not r["stable_during_copy"]
            ],
            "excluded_live_files": len(omissions),
        }
        rendered = json.dumps(manifest, separators=(",", ":"), ensure_ascii=True).encode() + b"\n"
        archive.writestr(f"{PREFIX}/MANIFEST.json", rendered)
    (output / "MANIFEST.json").write_bytes(rendered)
    (output / "build_summary.json").write_text(
        json.dumps({k: v for k, v in manifest.items() if k != "files"}, indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                "archive": str(archive_path),
                "files": len(rows),
                "bytes": total_bytes,
                "unstable": manifest["unstable_origin_entries"],
            }
        ),
        flush=True,
    )
    if manifest["unstable_origin_entries"]:
        raise SystemExit("Origin changed while copying; do not present as a sealed export")


if __name__ == "__main__":
    main()
