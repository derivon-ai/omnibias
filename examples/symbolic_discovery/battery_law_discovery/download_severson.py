# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Download helper for the Severson/TRI fast-charging battery dataset.

The preferred public mirror is the BSEBench Hugging Face dataset:
``bsebench-org/severson-2019-raw``. The script intentionally keeps the
dependency optional so the rest of the demo can run from manually downloaded
`.mat` files.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def download_severson(out: Path, repo_id: str = "bsebench-org/severson-2019-raw") -> Path:
    """Download Severson raw `.mat` files into `out`.

    Returns the local directory containing the downloaded snapshot.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover - exercised only without optional dep
        raise SystemExit(
            "Missing optional dependency `huggingface_hub`.\n"
            "Install it with `pip install huggingface_hub`, or manually download "
            "the Severson `.mat` files from https://data.matr.io/1 and pass "
            "`--data-dir` to run_demo.py."
        ) from exc

    out.mkdir(parents=True, exist_ok=True)
    local_dir = snapshot_download(repo_id, repo_type="dataset", local_dir=str(out))
    return Path(local_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/severson_2019"))
    parser.add_argument("--repo-id", default="bsebench-org/severson-2019-raw")
    args = parser.parse_args()

    local_dir = download_severson(args.out, repo_id=args.repo_id)
    mats = sorted(local_dir.rglob("*.mat"))
    print(f"Downloaded Severson snapshot to {local_dir}")
    print(f"Found {len(mats)} .mat files")
    for mat in mats:
        print(f"  {mat}")


if __name__ == "__main__":
    main()
