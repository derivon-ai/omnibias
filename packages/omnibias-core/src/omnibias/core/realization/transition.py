# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Portable, versioned structural proposals. Acceptance is a downstream concern."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from hashlib import sha256
from typing import Any


@dataclass(frozen=True)
class SlotUpdate:
    name: str
    index: int
    value: float | tuple[float, ...]


@dataclass(frozen=True)
class TransitionProposal:
    source_digest: str
    source_version: int
    slot_ids: tuple[int, ...]
    updates: tuple[SlotUpdate, ...]
    kind: str
    coordinate_identity: str
    error_budget: float = 0.0
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.error_budget < 0 or self.schema_version != 1:
            raise ValueError('invalid transition budget or schema')
        if len(set(self.slot_ids)) != len(self.slot_ids):
            raise ValueError('slot IDs must be unique')

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, separators=(',', ':'), allow_nan=False)

    @classmethod
    def from_json(cls, text: str) -> TransitionProposal:
        data: dict[str, Any] = json.loads(text)
        data['slot_ids'] = tuple(data['slot_ids'])
        data['updates'] = tuple(SlotUpdate(item['name'], item['index'],
                                         tuple(item['value']) if isinstance(item['value'], list)
                                         else item['value']) for item in data['updates'])
        return cls(**data)


def snapshot_digest(snapshot: dict[str, Any]) -> str:
    return sha256(json.dumps(snapshot, sort_keys=True, separators=(',', ':'),
                             allow_nan=False).encode()).hexdigest()


__all__ = ['SlotUpdate', 'TransitionProposal', 'snapshot_digest']
