# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Data-parallel Gram, Gauss-Newton step and NTK diagonal over ``torch.distributed`` ranks equal the single-process ones.

Two gloo processes on the CPU, spawned in a subprocess so that the test session keeps no process group.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap

import pytest

pytest.importorskip("torch")

SCRIPT = textwrap.dedent(
    """
    import json, os, sys
    import numpy as np
    import torch
    import torch.distributed as dist
    import torch.multiprocessing as mp

    def work(rank, world, port, out):
        os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
        dist.init_process_group("gloo", rank=rank, world_size=world)
        from omnibias.torch.experimental import residual_gram as rg
        from omnibias.torch.experimental.distributed import (
            distributed_residual_gauss_newton, distributed_residual_gram,
            distributed_residual_ntk_diagonal)
        g = torch.Generator().manual_seed(0)
        D, H = 5, 6
        sizes = [D, H, H, 1]
        acts = ["tanh", "tanh", None]
        layers = [(torch.randn(b, a, generator=g, dtype=torch.float64) / a ** 0.5,
                   0.1 * torch.randn(b, generator=g, dtype=torch.float64), acts[i])
                  for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:]))]
        X = torch.randn(8, D, generator=g, dtype=torch.float64)
        Xb = torch.randn(4, D, generator=g, dtype=torch.float64)
        interior = lambda u, gr, lap, x: lap[:, 0] + 0.5 * u[:, 0] - torch.sin(x[:, 0])
        boundary = lambda u, gr, lap, x: u[:, 0] - x.sum(-1)
        full = [(X, interior, 2), (Xb, boundary, 0)]
        part = lambda a: a.chunk(world)[rank]
        local = [(part(X), interior, 2), (part(Xb), boundary, 0)]
        rel = lambda a, b: float((a - b).abs().max() / a.abs().max())
        res = {}
        for strategy in ("auto", "streams", "explicit"):
            r1, K1 = rg.deep_residual_gram(full, layers, strategy=strategy)
            r2, K2 = distributed_residual_gram(local, layers, strategy=strategy)
            res["gram_" + strategy] = max(rel(r1, r2), rel(K1, K2))
        r1, K1, s1 = rg.deep_residual_gauss_newton(full, layers, 1e-2)
        r2, K2, s2 = distributed_residual_gauss_newton(local, layers, 1e-2)
        res["step"] = max(max(rel(a, c), rel(b, d)) for (a, b), (c, d) in zip(s1, s2))
        rd1, d1 = rg.deep_residual_ntk_diagonal(full, layers)
        rd2, d2 = distributed_residual_ntk_diagonal(local, layers)
        res["ntk"] = max(rel(rd1, rd2), rel(d1, d2))
        if rank == 0:
            open(out, "w").write(json.dumps(res))
        dist.destroy_process_group()

    if __name__ == "__main__":
        out, port = sys.argv[1], int(sys.argv[2])
        mp.spawn(work, args=(2, port, out), nprocs=2, join=True)
    """
)


def test_ranks_reproduce_the_single_process_programs(tmp_path):
    script = tmp_path / "run.py"
    script.write_text(SCRIPT)
    out = tmp_path / "out.json"
    port = 29500 + (os.getpid() % 400)
    proc = subprocess.run(
        [sys.executable, str(script), str(out), str(port)],
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
        env={**os.environ, "OMP_NUM_THREADS": "1"},
    )
    assert proc.returncode == 0, proc.stderr[-2500:]
    res = json.loads(out.read_text())
    for key in ("gram_auto", "gram_streams", "gram_explicit", "step", "ntk"):
        assert res[key] < 1e-12, (key, res[key])
