# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Check package-owned README assets and render actual wheel long descriptions."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import zipfile
from email.parser import BytesParser
from pathlib import Path

from PIL import Image
from validate_wheels import discover, wheel_readme


def check(projects: list[Path]) -> None:
    hashes: dict[str, Path] = {}
    for project in projects:
        readme = (project / "README.md").read_text()
        assets = project / "docs/visuals"
        assert "story.gif)" in readme, f"{project}: embed its animation directly"
        assert "poster.png)" in readme, f"{project}: link a static alternative"
        provenance = json.loads((assets / "provenance.json").read_text())
        assert provenance["source"], f"{project}: explain where the values come from"
        assert (
            provenance["scene_sha256"]
            == hashlib.sha256((assets / "scene.py").read_bytes()).hexdigest()
        ), f"{project}: regenerate visuals after changing the scene"
        for name in ("story.gif", "story-mobile.gif"):
            path = assets / name
            payload = path.read_bytes()
            digest = hashlib.sha256(payload).hexdigest()
            assert digest == provenance["sha256"][name], f"{path}: hash mismatch"
            assert digest not in hashes, f"Duplicated animation: {path} and {hashes.get(digest)}"
            hashes[digest] = path
            assert len(payload) < 1.5 * 1024 * 1024, f"{path}: oversized animation"
            with Image.open(path) as image:
                assert image.n_frames > 1, f"{path}: animation has only one frame"
                for frame in range(image.n_frames):
                    image.seek(frame)
                    assert image.info["duration"] >= 100, f"{path}: flashing frames"
        for name in (
            "poster.png",
            "poster-mobile.png",
            "poster.svg",
            "poster-mobile.svg",
            "scene.py",
        ):
            assert (assets / name).is_file(), f"{project}: missing {name}"
    print(f"{len(projects)} distinct README presentations checked")


def render_wheels(wheels: Path, output: Path) -> None:
    from readme_renderer.markdown import render

    output.mkdir(parents=True, exist_ok=True)
    for wheel in sorted(wheels.glob("*.whl")):
        with zipfile.ZipFile(wheel) as archive:
            name = next(n for n in archive.namelist() if n.endswith(".dist-info/METADATA"))
            metadata = BytesParser().parsebytes(archive.read(name))
            assert metadata["Description-Content-Type"].startswith("text/markdown")
            source = wheel_readme(metadata)
            html = render(source)
            assert html and "<img " in html, f"{wheel}: long description lost its visual"
            # Package releases must reference immutable, public visual assets.
            if metadata["Name"] in {p.name for p in discover(None)[0]}:
                images = re.findall(r'<img[^>]+src="([^"]+)"', html)
                assert images and all(
                    re.match(
                        r"https://raw\.githubusercontent\.com/derivon-ai/omnibias/[0-9a-f]{40}/",
                        url,
                    )
                    for url in images
                ), f"{wheel}: image URL is not immutable"
            (output / (metadata["Name"] + ".html")).write_text(
                '<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width"><style>body{max-width:1000px;margin:32px auto;padding:20px;font:16px/1.6 system-ui;overflow-wrap:anywhere}img{max-width:100%;height:auto}pre{overflow:auto}table{display:block;overflow:auto}</style>'
                + html
            )
    print("Rendered wheel long descriptions with PyPI's Markdown renderer")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--projects-root", type=Path)
    parser.add_argument("--wheels", type=Path)
    parser.add_argument("--output", type=Path, default=Path("artifacts/readme-rendered"))
    args = parser.parse_args()
    primitives, consumers = discover(args.projects_root)
    check(primitives + consumers)
    if args.wheels:
        render_wheels(args.wheels, args.output)


if __name__ == "__main__":
    main()
