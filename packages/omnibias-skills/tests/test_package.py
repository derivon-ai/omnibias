# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Smoke tests for the omnibias-skills package scaffold."""

from __future__ import annotations

import omnibias.skills as skills

_EXPECTED_SKILLS = {
    "omnibias-backends",
    "omnibias-fields-pinn",
    "omnibias-geometry",
    "omnibias-curvature-optim",
    "omnibias-verify",
    "omnibias-symbolic",
}


def test_version() -> None:
    assert skills.__version__ == "0.1.0a1"


def test_public_api() -> None:
    for name in ("install_skills", "check_skills", "uninstall_skills", "bundled_skills"):
        assert name in skills.__all__
        assert hasattr(skills, name)


def test_bundled_skills_present() -> None:
    infos = skills.bundled_skills()
    assert {info.name for info in infos} == _EXPECTED_SKILLS
    # every bundled skill carries a non-empty description (used for discovery)
    assert all(info.description.strip() for info in infos)


def test_no_backend_dependency() -> None:
    # The installer is pure standard library; importing it must not pull a backend.
    import sys

    assert "torch" not in sys.modules
    assert "jax" not in sys.modules
