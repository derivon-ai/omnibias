# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Data-parallel Gram, Gauss-Newton step and NTK diagonal agree with the single-device programs.

Runs in a subprocess with four virtual CPU devices, because the device count is fixed when JAX starts.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap

import pytest

pytest.importorskip("jax")

SCRIPT = textwrap.dedent(
    """
    import json
    import numpy as np
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from jax.sharding import Mesh
    from omnibias.jax.experimental import (
        deep_residual_gauss_newton, deep_residual_gram, deep_residual_ntk_diagonal)
    from omnibias.jax.experimental.distributed import (
        distributed_residual_gauss_newton, distributed_residual_gram,
        distributed_residual_ntk_diagonal, shard_points)

    mesh = Mesh(np.array(jax.devices()), ("data",))
    rng = np.random.default_rng(0)
    D, H = 5, 6
    sizes = [D, H, H, 1]
    acts = ["tanh", "tanh", None]
    layers = [
        (jnp.asarray(rng.standard_normal((b, a)) / np.sqrt(a)),
         jnp.asarray(0.1 * rng.standard_normal(b)), acts[i])
        for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:]))
    ]
    X = jnp.asarray(rng.standard_normal((8, D)))
    Xb = jnp.asarray(rng.standard_normal((4, D)))
    interior = lambda u, g, lap, x: lap[:, 0] + 0.5 * u[:, 0] - jnp.sin(x[:, 0])
    boundary = lambda u, g, lap, x: u[:, 0] - x.sum(-1)
    blocks = [(X, interior, 2), (Xb, boundary, 0)]
    sharded = [(shard_points(X, mesh), interior, 2), (shard_points(Xb, mesh), boundary, 0)]
    rel = lambda a, b: float(jnp.max(jnp.abs(a - b)) / jnp.max(jnp.abs(a)))
    out = {"devices": len(jax.devices())}
    for strategy in ("auto", "streams", "explicit"):
        r1, K1 = deep_residual_gram(blocks, layers, strategy=strategy)
        r2, K2 = distributed_residual_gram(sharded, layers, mesh, strategy=strategy)
        out["gram_" + strategy] = max(rel(r1, r2), rel(K1, K2))
    r1, K1, s1 = deep_residual_gauss_newton(blocks, layers, 1e-2)
    r2, K2, s2 = distributed_residual_gauss_newton(sharded, layers, 1e-2, mesh)
    out["step"] = max(max(rel(a, b), rel(c, d)) for (a, c), (b, d) in zip(s1, s2))
    rd1, d1 = deep_residual_ntk_diagonal(blocks, layers)
    rd2, d2 = distributed_residual_ntk_diagonal(sharded, layers, mesh)
    out["ntk"] = max(rel(rd1, rd2), rel(d1, d2))
    try:
        distributed_residual_gram([(X[:7], interior, 2)], layers, mesh)
        out["indivisible"] = "accepted"
    except ValueError as err:
        out["indivisible"] = "refused"
    print("RESULT" + json.dumps(out))
    """
)


def test_sharded_programs_equal_the_single_device_programs():
    env = dict(os.environ)
    env.update(
        XLA_FLAGS="--xla_force_host_platform_device_count=4",
        JAX_PLATFORMS="cpu",
        PYTHONHASHSEED="0",
    )
    proc = subprocess.run(
        [sys.executable, "-c", SCRIPT],
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    line = next(ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT"))
    out = json.loads(line[len("RESULT") :])
    assert out["devices"] == 4
    for key in ("gram_auto", "gram_streams", "gram_explicit", "step", "ntk"):
        assert out[key] < 1e-12, (key, out[key])
    assert out["indivisible"] == "refused"


CACHE_SCRIPT = textwrap.dedent(
    """
    import json
    import numpy as np
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from jax.sharding import Mesh
    from omnibias.jax.experimental import deep_residual_gauss_newton, deep_residual_gram
    from omnibias.jax.experimental import distributed as dist

    mesh = Mesh(np.array(jax.devices()), ("data",))
    rng = np.random.default_rng(1)
    D, H = 4, 5
    sizes = [D, H, H, 1]
    acts = ["tanh", "tanh", None]
    def make(scale):
        g = np.random.default_rng(2)
        return [
            (jnp.asarray(scale * g.standard_normal((b, a)) / np.sqrt(a)),
             jnp.asarray(0.1 * g.standard_normal(b)), acts[i])
            for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:]))
        ]
    X = jnp.asarray(rng.standard_normal((8, D)))
    interior = lambda u, g, lap, x: lap[:, 0] + 0.5 * u[:, 0] - jnp.sin(x[:, 0])
    blocks = [(dist.shard_points(X, mesh), interior, 2)]
    ref = [(X, interior, 2)]
    rel = lambda a, b: float(jnp.max(jnp.abs(a - b)) / jnp.max(jnp.abs(a)))
    out = {}
    dist._PROGRAMS.clear()
    errs = []
    for scale in (1.0, 1.3, 0.7):  # a new set of weights each call: one program, correct values each time
        layers = make(scale)
        r1, K1 = deep_residual_gram(ref, layers)
        r2, K2 = dist.distributed_residual_gram(blocks, layers, mesh)
        errs.append(max(rel(r1, r2), rel(K1, K2)))
        if scale == 1.0:
            out["programs_after_first"] = len(dist._PROGRAMS)
    out["gram_errors"] = errs
    out["programs_after_three"] = len(dist._PROGRAMS)
    out["compiles"] = int(next(iter(dist._PROGRAMS.values()))._cache_size())
    errs = []
    for damping in (1e-2, 1e-1):
        layers = make(1.0)
        r1, K1, s1 = deep_residual_gauss_newton(ref, layers, damping)
        r2, K2, s2 = dist.distributed_residual_gauss_newton(blocks, layers, damping, mesh)
        errs.append(max(max(rel(a, b), rel(c, d)) for (a, c), (b, d) in zip(s1, s2)))
    out["step_errors"] = errs
    out["programs_total"] = len(dist._PROGRAMS)
    # a different structure (width) is a different program; the limit bounds the cache
    dist._PROGRAM_LIMIT = 2
    for width in (5, 6, 7):
        g = np.random.default_rng(3)
        lays = [(jnp.asarray(g.standard_normal((width, D))), None, "tanh"), (jnp.asarray(g.standard_normal((1, width))), None, None)]
        dist.distributed_residual_gram(blocks, lays, mesh)
    out["bounded"] = len(dist._PROGRAMS)
    # under an outer jit the layers are tracers: the program is still found by structure
    layers = make(1.0)
    f = jax.jit(lambda W0: dist.distributed_residual_gram(blocks, [(W0, *layers[0][1:])] + layers[1:], mesh))
    r3, K3 = f(layers[0][0])
    r1, K1 = deep_residual_gram(ref, layers)
    out["jit_error"] = max(rel(r1, r3), rel(K1, K3))
    print("RESULT" + json.dumps(out))
    """
)


def test_the_sharded_program_is_built_once_per_structure_and_takes_the_weights_as_arguments():
    env = dict(os.environ)
    env.update(
        XLA_FLAGS="--xla_force_host_platform_device_count=4",
        JAX_PLATFORMS="cpu",
        PYTHONHASHSEED="0",
    )
    proc = subprocess.run(
        [sys.executable, "-c", CACHE_SCRIPT],
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    line = next(ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT"))
    out = json.loads(line[len("RESULT") :])
    assert out["programs_after_first"] == out["programs_after_three"] == 1
    assert out["compiles"] == 1  # three weight sets, one compilation
    assert max(out["gram_errors"]) < 1e-12 and max(out["step_errors"]) < 1e-12
    assert out["programs_total"] == 2  # the Gram and the Gauss-Newton program
    assert out["bounded"] == 2
    assert out["jit_error"] < 1e-12
