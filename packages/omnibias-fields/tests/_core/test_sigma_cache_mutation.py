# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""SigmaCache in-place mutation (stale-read) guard.

The guard keys off torch's ``Tensor._version`` counter, so these tests need a
real backend tensor and live outside the pure-Python substrate module. The
contract: ``z`` (and the ``coords`` it derives from) is immutable for the life
of the cache; mutating it in place must raise on the next read rather than
silently returning stale ``sigma^(n)(z)``.
"""

from __future__ import annotations

import pytest
import torch
from omnibias.fields._core import SigmaCache


def test_sigma_cache_detects_inplace_mutation_on_get_or_compute() -> None:
    z = torch.linspace(-1.0, 1.0, 8)
    cache: SigmaCache[torch.Tensor] = SigmaCache(z=z)
    cache.get_or_compute(0, lambda n: z * 1.0)  # fill from pristine z
    z.add_(1.0)  # in-place mutation -> cached sigma is now stale
    with pytest.raises(RuntimeError, match="mutated in place"):
        cache.get_or_compute(1, lambda n: z * 2.0)


def test_sigma_cache_detects_inplace_mutation_on_get() -> None:
    z = torch.ones(4)
    cache: SigmaCache[torch.Tensor] = SigmaCache(z=z)
    cache.put(0, z.clone())
    z.mul_(2.0)
    with pytest.raises(RuntimeError, match="mutated in place"):
        cache.get(0)


def test_sigma_cache_allows_unmutated_torch_z() -> None:
    z = torch.arange(5.0)
    cache: SigmaCache[torch.Tensor] = SigmaCache(z=z)
    first = cache.get_or_compute(0, lambda n: z + 1.0)
    again = cache.get_or_compute(0, lambda n: z + 999.0)  # cached: build skipped
    assert torch.equal(first, again)
    assert torch.equal(cache.get(0), z + 1.0)


def test_sigma_cache_noop_guard_for_immutable_jax_z() -> None:
    jnp = pytest.importorskip("jax.numpy")
    z = jnp.arange(5.0)
    cache: SigmaCache = SigmaCache(z=z)  # jax arrays immutable -> no tripwire
    out = cache.get_or_compute(0, lambda n: z + 1.0)
    assert out is cache.get(0)


def test_sigma_cache_noop_guard_for_non_tensor_z() -> None:
    cache: SigmaCache[str] = SigmaCache(z="z")  # no _version -> guard inert
    assert cache.get_or_compute(0, lambda n: "s0") == "s0"
    assert cache.get(0) == "s0"
