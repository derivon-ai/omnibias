# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""The published derivative-order ceiling must match the live registry.

``docs/scope-and-guarantees.md`` sec 2 is labelled "reading guidance for AI
agents", so when it drifts from the code it does not merely go stale -- it
teaches every agent and reader the wrong ceiling. It had drifted: ``silu`` /
``gelu`` / ``mish`` were listed as capped at ``n = 1`` "distributional (Dirac at
the kink)" long after they gained exact all-orders Leibniz towers, and none of
the three has a kink at all.

This guard parses the table and checks every claim against what the fastpath
actually does, on **both** backends, so the doc is pinned to the code rather
than to someone's memory of it.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

DOC = REPO / "docs" / "scope-and-guarantees.md"

#: Probe depth. Anything still answering here is reported as unbounded; the
#: towers that genuinely stop do so at 2 or 3, far below this.
PROBE_ORDER = 9

#: Sentinel for "answers at every probed order".
UNBOUNDED = "unbounded"

#: Points chosen to straddle every kink in the piecewise family (0 excluded:
#: it is exactly where the a.e. convention is a convention).
PROBE_POINTS = (-2.4, -1.3, -0.4, 0.7, 2.1, 6.5)


def _table_rows() -> list[tuple[str, str, int | str]]:
    """``(family_cell, activation, claimed_ceiling)`` from the sec-2 table."""
    text = DOC.read_text(encoding="utf-8")
    section = text.split("## 2. The derivative-order ceiling", 1)
    assert len(section) == 2, "section 2 heading moved; update this guard"
    body = section[1].split("\n## ", 1)[0]

    rows: list[tuple[str, str, int | str]] = []
    for line in body.splitlines():
        if not line.startswith("|") or line.startswith("|---"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) != 3 or cells[1] == "Max closed-form order `n`":
            continue
        family, claim = cells[0], cells[1]
        ceiling: int | str
        if UNBOUNDED in claim.lower():
            ceiling = UNBOUNDED
        else:
            digits = re.findall(r"\d+", claim)
            assert digits, f"unparseable ceiling cell: {claim!r}"
            ceiling = int(digits[0])
        for name in re.findall(r"`([^`]+)`", family):
            rows.append((family, name, ceiling))
    return rows


DOC_ROWS = _table_rows()


def _measure(fastpath, zeros) -> int | str:
    """Highest order the fastpath answers, or ``UNBOUNDED``."""
    top = -1
    for n in range(PROBE_ORDER + 1):
        try:
            out = fastpath(zeros, n)
        except NotImplementedError:
            break
        if out is None:
            break
        top = n
    return UNBOUNDED if top >= PROBE_ORDER else top


def _torch_ceilings() -> dict[str, int | str]:
    torch = pytest.importorskip("torch")
    from omnibias.torch.activations import registry

    z = torch.tensor(PROBE_POINTS, dtype=torch.float64)
    return {
        name: _measure(spec.fastpath, z)
        for name, spec in registry._REGISTRY.items()
        if spec.fastpath is not None
    }


def _jax_ceilings() -> dict[str, int | str]:
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from omnibias.jax import activations

    z = jnp.asarray(PROBE_POINTS, dtype=jnp.float64)
    return {
        name: _measure(spec.fastpath, z)
        for name, spec in activations._REGISTRY.items()
        if spec.fastpath is not None
    }


def test_the_table_actually_parses_into_claims() -> None:
    """A guard that silently parses nothing would pass forever."""
    assert len(DOC_ROWS) > 25, f"only parsed {len(DOC_ROWS)} claims; parser broke"
    assert any(c == UNBOUNDED for _, _, c in DOC_ROWS)
    assert any(c == 2 for _, _, c in DOC_ROWS)
    assert any(c == 3 for _, _, c in DOC_ROWS)


@pytest.mark.parametrize("name,claimed", [(n, c) for _, n, c in DOC_ROWS])
def test_every_documented_ceiling_matches_torch(name: str, claimed: int | str) -> None:
    measured = _torch_ceilings()
    assert name in measured, f"docs list `{name}`, which no torch spec registers"
    assert measured[name] == claimed, (
        f"docs claim `{name}` caps at {claimed}, torch fastpath gives {measured[name]}"
    )


@pytest.mark.parametrize("name,claimed", [(n, c) for _, n, c in DOC_ROWS])
def test_every_documented_ceiling_matches_jax(name: str, claimed: int | str) -> None:
    measured = _jax_ceilings()
    assert name in measured, f"docs list `{name}`, which no jax spec registers"
    assert measured[name] == claimed, (
        f"docs claim `{name}` caps at {claimed}, jax fastpath gives {measured[name]}"
    )


def test_the_two_backends_agree_on_every_ceiling() -> None:
    """Bit-identical twins must not disagree about where a tower stops."""
    torch_c, jax_c = _torch_ceilings(), _jax_ceilings()
    shared = sorted(set(torch_c) & set(jax_c))
    assert len(shared) > 40, f"only {len(shared)} shared specs; a backend lost some"
    mismatched = {n: (torch_c[n], jax_c[n]) for n in shared if torch_c[n] != jax_c[n]}
    assert not mismatched, f"backend ceilings disagree: {mismatched}"


def test_the_capped_families_really_are_capped() -> None:
    """The negative half: a claimed cap must genuinely raise past its ceiling.

    Without this, replacing every cell with "unbounded" would pass.
    """
    torch = pytest.importorskip("torch")
    from omnibias.torch.activations import registry

    z = torch.tensor(PROBE_POINTS, dtype=torch.float64)
    capped = [(n, c) for _, n, c in DOC_ROWS if c != UNBOUNDED]
    assert capped, "no capped families left to check; the negative half went vacuous"
    for name, ceiling in capped:
        spec = registry._REGISTRY[name]
        assert spec.fastpath is not None
        spec.fastpath(z, int(ceiling))  # the ceiling itself must work
        with pytest.raises(NotImplementedError):
            spec.fastpath(z, int(ceiling) + 1)


def test_negative_orders_are_rejected_by_every_fastpath() -> None:
    """The derivative-tower contract: ``n < 0`` is a ValueError, not a cap."""
    torch = pytest.importorskip("torch")
    from omnibias.torch.activations import registry

    z = torch.tensor(PROBE_POINTS, dtype=torch.float64)
    for _name, spec in sorted(registry._REGISTRY.items()):
        if spec.fastpath is None:
            continue
        with pytest.raises(ValueError):
            spec.fastpath(z, -1)
