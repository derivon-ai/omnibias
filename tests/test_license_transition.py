# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Historical grants, prospective boundaries and attribution preservation."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import tomllib

ROOT = Path(__file__).resolve().parents[1]


def test_historical_inventory_preserves_published_apache_grants() -> None:
    baseline = json.loads((ROOT / "docs/license-transition-baseline.json").read_text())
    entries = {entry["name"]: entry for entry in baseline["distributions"]}
    assert len(entries) == len(baseline["distributions"]) == 44
    for short in ("core", "torch", "jax", "keras", "fields", "ferminet", "pinn", "geometry"):
        assert entries[f"omnibias-{short}"]["license"] == "Apache-2.0"
    changed = {"omnibias-" + name for name in ("curvature", "graph", "partition", "struct")}
    for manifest in (ROOT / "packages").glob("*/pyproject.toml"):
        project = tomllib.loads(manifest.read_text())["project"]
        original = entries[project["name"]]
        if project["name"] in changed:
            assert original["license"] == "Apache-2.0"
            assert project["license"] == "AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial"
            for notice in ("NOTICE", "COMMERCIAL-LICENSE.md"):
                assert notice in project["license-files"]
                assert (manifest.parent / notice).is_file()
        else:
            assert project["license"] == original["license"]


def test_header_rewrite_keeps_every_copyright_notice() -> None:
    spec = importlib.util.spec_from_file_location("license_headers", ROOT / "scripts/license_headers.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original = ("#!/usr/bin/env python3\n# SPDX-License-Identifier: Apache-2.0\n"
                "# Copyright (C) 2024 Original Author\n# Copyright (C) 2026 Derivon\n"
                "# Additional attribution stays here.\nvalue = 1\n")
    updated = module.rewrite(original, "AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial")
    assert updated.startswith("#!/usr/bin/env python3\n# SPDX-License-Identifier:")
    assert "# Copyright (C) 2024 Original Author\n# Copyright (C) 2026 Derivon\n" in updated
    assert updated.endswith("# Additional attribution stays here.\nvalue = 1\n")
    assert module.rewrite(updated, "AGPL-3.0-or-later OR LicenseRef-omnibias-Commercial") == updated
