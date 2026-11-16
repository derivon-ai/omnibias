# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Derivon
"""Generate the cross-backend operator fixtures ``packages/omnibias-core/src/omnibias/core/fixtures/cross_backend_operators_v1.json``.

Every case is a small dense network whose float64 weights and points are stored exactly (``float.hex``).  The reference
values are computed here in ``DPS``-digit ``mpmath`` arithmetic by central differences of the network itself, so no
derivative formula of this package is involved: the Laplacian, ``|grad u|^2``, the divergence, ``log|det(I + J)|``, the
solve ``(I + J)^{-1} rhs`` and the residual Gram ``K = J J^T`` with the damped Gauss-Newton step of a Poisson plus
boundary residual.  ``tests/test_cross_backend_fixtures.py`` replays them on JAX and PyTorch, and ``omnibias.core.conformance`` serves them to any engine.

usage: python scripts/generate_cross_backend_fixtures.py [--check]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mpmath as mp
import numpy as np

DPS = 90
HX1, HX2, HTHETA = mp.mpf("1e-18"), mp.mpf("1e-16"), mp.mpf("1e-14")
DAMPING = "0.01"
OUT = (
    Path(__file__).resolve().parents[1]
    / "packages/omnibias-core/src/omnibias/core/fixtures/cross_backend_operators_v1.json"
)

CASES = [
    {
        "name": "tanh_3_5_5_1",
        "activation": "tanh",
        "dims": [3, 5, 5, 1],
        "seed": 11,
        "interior": 3,
        "boundary": 2,
    },
    {
        "name": "silu_4_6_1",
        "activation": "silu",
        "dims": [4, 6, 1],
        "seed": 12,
        "interior": 3,
        "boundary": 2,
    },
    {
        "name": "tanh_2_4_4_4_1",
        "activation": "tanh",
        "dims": [2, 4, 4, 4, 1],
        "seed": 13,
        "interior": 2,
        "boundary": 2,
    },
    {
        "name": "tanh_field_3_6_3",
        "activation": "tanh",
        "dims": [3, 6, 3],
        "seed": 14,
        "interior": 3,
        "boundary": 0,
    },
]


def _act(name: str, z):
    if name == "tanh":
        return mp.tanh(z)
    if name == "silu":
        return z / (1 + mp.exp(-z))
    raise ValueError(name)


def _forward(layers, name, x):
    a = list(x)
    for k, (W, b) in enumerate(layers):
        z = [sum(W[o][i] * a[i] for i in range(len(a))) + b[o] for o in range(len(b))]
        a = z if k == len(layers) - 1 else [_act(name, v) for v in z]
    return a


def _hex(a):
    arr = np.asarray(a, dtype=np.float64)
    if arr.ndim == 0:
        return float(arr).hex()
    return [_hex(v) for v in arr]


def _mp_layers(layers):
    return [
        ([[mp.mpf(float(v)) for v in row] for row in W], [mp.mpf(float(v)) for v in b])
        for W, b in layers
    ]


def _shift(x, k, h):
    y = list(x)
    y[k] = y[k] + h
    return y


def _first(f, x, k):
    return (f(_shift(x, k, HX1)) - f(_shift(x, k, -HX1))) / (2 * HX1)


def _second(f, x, k):
    return (f(_shift(x, k, HX2)) - 2 * f(x) + f(_shift(x, k, -HX2))) / HX2**2


def _lap(layers, name, x, comp=0):
    return sum(_second(lambda z: _forward(layers, name, z)[comp], x, k) for k in range(len(x)))


def _jacobian(layers, name, x):
    out = len(layers[-1][1])
    return mp.matrix(
        [
            [_first(lambda z, c=c: _forward(layers, name, z)[c], x, k) for k in range(len(x))]
            for c in range(out)
        ]
    )


def _params(layers):
    out = []
    for W, b in layers:
        out.extend(v for row in W for v in row)
        out.extend(b)
    return out


def _with_params(layers, vec):
    out, pos = [], 0
    for W, b in layers:
        nW = [[vec[pos + o * len(W[0]) + i] for i in range(len(W[0]))] for o in range(len(W))]
        pos += len(W) * len(W[0])
        nb = list(vec[pos : pos + len(b)])
        pos += len(b)
        out.append((nW, nb))
    return out


def _residual_rows(layers, name, interior, boundary):
    rows = [_lap(layers, name, x) + mp.sin(x[0]) for x in interior]
    rows += [_forward(layers, name, x)[0] - sum(x) for x in boundary]
    return rows


def _s(v) -> str:
    return mp.nstr(v, 40, strip_zeros=False)


def build_case(case: dict) -> dict:
    rng = np.random.default_rng(case["seed"])
    dims = case["dims"]
    layers = [
        (
            (rng.standard_normal((b, a)) * 0.6 / np.sqrt(a)).astype(np.float64),
            (0.2 * rng.standard_normal(b)).astype(np.float64),
        )
        for a, b in zip(dims[:-1], dims[1:], strict=True)
    ]
    D = dims[0]
    interior = rng.standard_normal((case["interior"], D))
    boundary = rng.standard_normal((case["boundary"], D)) if case["boundary"] else np.zeros((0, D))
    rhs = rng.standard_normal((case["interior"], dims[-1]))
    mpl = _mp_layers(layers)
    name = case["activation"]
    xs = [[mp.mpf(float(v)) for v in row] for row in interior]
    ref: dict = {}
    if dims[-1] == 1:
        ref["u"] = [_s(_forward(mpl, name, x)[0]) for x in xs]
        ref["grad_sq"] = [
            _s(sum(_first(lambda z: _forward(mpl, name, z)[0], x, k) ** 2 for k in range(D)))
            for x in xs
        ]
        ref["lap"] = [_s(_lap(mpl, name, x)) for x in xs]
        bx = [[mp.mpf(float(v)) for v in row] for row in boundary]
        r0 = _residual_rows(mpl, name, xs, bx)
        theta = _params(mpl)
        N = len(r0)
        J = mp.matrix(N, len(theta))
        for p in range(len(theta)):
            plus, minus = list(theta), list(theta)
            plus[p] += HTHETA
            minus[p] -= HTHETA
            rp = _residual_rows(_with_params(mpl, plus), name, xs, bx)
            rm = _residual_rows(_with_params(mpl, minus), name, xs, bx)
            for n in range(N):
                J[n, p] = (rp[n] - rm[n]) / (2 * HTHETA)
        K = J * J.T
        alpha = mp.lu_solve(K + mp.mpf(DAMPING) * mp.eye(N), mp.matrix(r0))
        step = J.T * alpha
        ref["residual"] = {
            "r": [_s(v) for v in r0],
            "K": [[_s(K[i, j]) for j in range(N)] for i in range(N)],
            "damping": DAMPING,
            "step": [_s(step[i]) for i in range(len(theta))],
        }
    else:
        Js = [_jacobian(mpl, name, x) for x in xs]
        ref["divergence"] = [_s(sum(J[i, i] for i in range(D))) for J in Js]
        ref["logdet"] = [_s(mp.log(abs(mp.det(mp.eye(D) + J)))) for J in Js]
        ref["solve"] = [
            [_s(v) for v in mp.lu_solve(mp.eye(D) + J, mp.matrix([mp.mpf(float(v)) for v in r]))]
            for J, r in zip(Js, rhs, strict=True)
        ]
    return {
        "name": case["name"],
        "activation": name,
        "dims": dims,
        "layers": [{"W": _hex(W), "b": _hex(b)} for W, b in layers],
        "interior": _hex(interior),
        "boundary": _hex(boundary),
        "rhs": _hex(rhs),
        "reference": ref,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="fail when the committed fixture is stale")
    args = ap.parse_args()
    mp.mp.dps = DPS
    doc = {
        "version": 1,
        "dps": DPS,
        "protocol": "central differences in mpmath; exact float64 weights and points as hex floats",
        "cases": [build_case(c) for c in CASES],
    }
    text = json.dumps(doc, indent=1) + "\n"
    if args.check:
        return 0 if OUT.exists() and OUT.read_text() == text else 1
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text)
    print(f"wrote {OUT} ({len(text)} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
