# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Keep source mathematics intact when compiling the research record."""

from pathlib import Path

import pytest

from scripts.compile_session_findings import appendix


@pytest.mark.parametrize(
    "literal",
    [
        "    -3[s^2+d+4-u](xx^T+zz^T)\n",
        "\t-3[s^2+d+4-u](xx^T+zz^T)\n",
        "```text\n[x](y)\n```\n",
        "~~~text\n[x](y)\n~~~\n",
        "\\[\n[x](y)\n\\]\n",
        "$$\n[x](y)\n$$\n",
        "`[x](y)`\n",
        "\\([x](y)\\)\n",
        "$[x](y)$\n",
    ],
)
def test_mathematical_and_code_literals_are_preserved(literal: str) -> None:
    assert appendix(literal, Path("/tmp/source/note.md")) == literal


def test_source_links_and_heading_levels_are_normalized() -> None:
    result = appendix(
        "# Heading\n\n[local](paper.md) [web](https://example.org/paper)\n",
        Path("/tmp/source/note.md"),
    )
    assert result == (
        "### Heading\n\n[local](/tmp/source/paper.md) "
        "[web](https://example.org/paper)\n"
    )
