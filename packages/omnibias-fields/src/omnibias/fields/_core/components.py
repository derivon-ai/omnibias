# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Component (output-channel) specification for vector PINN fields.

A :class:`ComponentSpec` is the structural identity of the output side of
a typed PINN field. It records:

- ``names``: the ordered list of scalar component names (e.g.
  ``("u", "v", "w", "p")`` for 3D NS primitive variables).
- ``groups``: named subsets of components (e.g.
  ``{"velocity": ("u", "v", "w")}``) used by :class:`VectorView` for
  vector-level operators (curl, div, advect, ...).

The spec is hashable / equality-comparable so it can be used as a dict
key (operator-registry caching) and ``copy.replace``-able by users that
need to extend an existing spec.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass


def _normalise_names(names: Sequence[str]) -> tuple[str, ...]:
    out: list[str] = []
    seen: set[str] = set()
    for n in names:
        if not isinstance(n, str):
            raise TypeError(
                f"component name must be str, got {type(n).__name__}: {n!r}"
            )
        if not n:
            raise ValueError("component names must be non-empty strings")
        if n in seen:
            raise ValueError(f"duplicate component name: {n!r}")
        seen.add(n)
        out.append(n)
    return tuple(out)


def _normalise_groups(
    groups: Mapping[str, Sequence[str]] | None,
    names: tuple[str, ...],
) -> dict[str, tuple[str, ...]]:
    if groups is None:
        return {}
    out: dict[str, tuple[str, ...]] = {}
    name_set = set(names)
    for gname, members in groups.items():
        if not isinstance(gname, str) or not gname:
            raise ValueError(f"group name must be non-empty str: {gname!r}")
        if gname in name_set:
            raise ValueError(
                f"group name {gname!r} clashes with component name"
            )
        norm = tuple(members)
        if len(norm) == 0:
            raise ValueError(f"group {gname!r} is empty")
        seen: set[str] = set()
        for m in norm:
            if not isinstance(m, str):
                raise TypeError(
                    f"group member must be str, got {type(m).__name__}: {m!r}"
                )
            if m not in name_set:
                raise ValueError(
                    f"group {gname!r} member {m!r} not in components {names!r}"
                )
            if m in seen:
                raise ValueError(
                    f"group {gname!r} has duplicate member: {m!r}"
                )
            seen.add(m)
        out[gname] = norm
    return out


@dataclass(frozen=True)
class ComponentSpec:
    """Frozen ordered list of scalar component names plus named groups.

    Parameters
    ----------
    names
        Ordered names of the scalar components produced by the field. Must
        be non-empty, all distinct, all non-empty strings.
    groups
        Optional mapping from group name to a tuple of component names.
        Each group name must not collide with any component name and each
        member must be a known component.

    Examples
    --------
    >>> spec = ComponentSpec(
    ...     names=("u", "v", "w", "p"),
    ...     groups={"velocity": ("u", "v", "w")},
    ... )
    >>> spec.names
    ('u', 'v', 'w', 'p')
    >>> spec.groups["velocity"]
    ('u', 'v', 'w')
    """

    names: tuple[str, ...]
    groups: tuple[tuple[str, tuple[str, ...]], ...]

    def __init__(
        self,
        names: Sequence[str],
        *,
        groups: Mapping[str, Sequence[str]] | None = None,
    ) -> None:
        norm_names = _normalise_names(names)
        if not norm_names:
            raise ValueError("ComponentSpec.names must be non-empty")
        norm_groups = _normalise_groups(groups, norm_names)
        groups_t = tuple((k, v) for k, v in norm_groups.items())
        object.__setattr__(self, "names", norm_names)
        object.__setattr__(self, "groups", groups_t)

    @property
    def n_components(self) -> int:
        return len(self.names)

    def index(self, name: str) -> int:
        """Resolve a component name to its integer position in ``names``."""
        if name not in self.names:
            raise KeyError(
                f"component {name!r} not in {self.names!r}"
            )
        return self.names.index(name)

    def has(self, name: str) -> bool:
        return name in self.names or any(g == name for g, _ in self.groups)

    def is_component(self, name: str) -> bool:
        return name in self.names

    def is_group(self, name: str) -> bool:
        return any(g == name for g, _ in self.groups)

    def group_members(self, name: str) -> tuple[str, ...]:
        for g, members in self.groups:
            if g == name:
                return members
        raise KeyError(f"group {name!r} not in {[g for g, _ in self.groups]!r}")

    def all_known_names(self) -> tuple[str, ...]:
        return self.names + tuple(g for g, _ in self.groups)

    def __len__(self) -> int:
        return len(self.names)

    def __contains__(self, name: object) -> bool:
        if not isinstance(name, str):
            return False
        return self.has(name)

    def __iter__(self):
        return iter(self.names)

    def __repr__(self) -> str:
        bits = [f"names={self.names!r}"]
        if self.groups:
            gd = {g: list(m) for g, m in self.groups}
            bits.append(f"groups={gd!r}")
        return f"ComponentSpec({', '.join(bits)})"

    def asdict(self) -> dict:
        return {
            "names": list(self.names),
            "groups": {g: list(m) for g, m in self.groups},
        }


__all__ = ["ComponentSpec"]
